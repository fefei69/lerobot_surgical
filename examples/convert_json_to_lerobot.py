"""
Script Json to Lerobot.

# --raw-dir     Corresponds to the directory of your JSON dataset
# --repo-id     Your unique repo ID on Hugging Face Hub
# --robot_type  The type of the robot used in the dataset (e.g., Unitree_G1_Dex3, Unitree_Z1_Dual, Unitree_G1_Dex3)
# --push_to_hub Whether or not to upload the dataset to Hugging Face Hub (true or false)

python unitree_lerobot/utils/convert_unitree_json_to_lerobot.py \
    --raw-dir $HOME/datasets/g1_grabcube_double_hand \
    --repo-id your_name/g1_grabcube_double_hand \
    --robot_type Unitree_G1_Dex3 \ 
    --push_to_hub
"""
import os
import cv2
import tqdm
import tyro
import json
import glob
import dataclasses
import shutil
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Literal, List, Dict, Optional

from lerobot.common.constants import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from action_utils import *
from utils import uniform_sampling_numpy


# migrate from unitree_lerobot.utils.constants import ROBOT_CONFIGS
@dataclasses.dataclass(frozen=True)
class RobotConfig:
    motors: List[str]
    cameras: List[str]
    camera_to_image_key:Dict[str, str]
    json_state_data_name: List[str]
    json_action_data_name: List[str]

dVRK_CONFIG = RobotConfig(
    motors=[
        "psm_yaw_joint",
        "psm_pitch_end_joint",
        "psm_main_insertion_joint",
        "psm_tool_roll_joint",
        "psm_tool_pitch_joint",
        "psm_tool_yaw_joint",
        "psm_tool_gripper_joint",
        
    ],
    cameras=[
        "cam_left",
        "cam_right",
    ],
    camera_to_image_key = {'left_image': 'cam_left', 'right_image': 'cam_right'},
    json_state_data_name = ['psm_cutter_js', 'psm_retraction_js'],
    json_action_data_name = ['left_arm', 'right_arm']
)

ROBOT_CONFIGS = {
    "DVRK": dVRK_CONFIG,
}

@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10*5
    image_writer_threads: int = 5*5
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()


class JsonDataset:
    def __init__(self, data_dirs: Path, robot_type: str) -> None:
        """
        Initialize the dataset for loading and processing HDF5 files containing robot manipulation data.
        
        Args:
            data_dirs: Path to directory containing training data
        """
        assert data_dirs is not None, "Data directory cannot be None"
        assert robot_type is not None, "Robot type cannot be None"
        self.data_dirs = data_dirs
        self.json_file = 'data.json'

        # load Q: perspective transformation matrix to reconstruct point clouds from disparities
        self.Q = np.load("Q.npy")

        # Initialize paths and cache
        self._init_paths()
        self._init_cache()
        self.json_state_data_name = ROBOT_CONFIGS[robot_type].json_state_data_name
        self.json_action_data_name = ROBOT_CONFIGS[robot_type].json_action_data_name
        self.camera_to_image_key = ROBOT_CONFIGS[robot_type].camera_to_image_key


    def _init_paths(self) -> None:
        """Initialize episode and task paths."""

        self.episode_paths = []
        self.task_paths = []
        for task_path in glob.glob(os.path.join(self.data_dirs, '*')):
            if os.path.isdir(task_path):
                episode_paths = glob.glob(os.path.join(task_path, '*'))
                if episode_paths:
                    self.task_paths.append(task_path)
                    self.episode_paths.append(episode_paths[1]) # only take the data.json path, ignore colors/
        
        self.episode_paths = sorted(self.episode_paths)
        self.episode_ids = list(range(len(self.episode_paths)))


    def __len__(self) -> int:
        """Return the number of episodes in the dataset."""
        return len(self.episode_paths)


    def _init_cache(self) -> List:
        """Initialize data cache if enabled."""

        self.episodes_data_cached = []
        for episode_path in tqdm.tqdm(self.episode_paths, desc="Loading Cache Json"):
            # json_path = os.path.join(episode_path, self.json_file)
            json_path = episode_path
            with open(json_path, 'r', encoding='utf-8') as jsonf:
                self.episodes_data_cached.append(json.load(jsonf))

        print(f"==> Cached {len(self.episodes_data_cached)} episodes")

        return self.episodes_data_cached


    def _extract_data(self, episode_data: Dict, key: str, parts: List[str]) -> np.ndarray:
        """
        Extract data from episode dictionary for specified parts.
        
        Args:
            episode_data: Dictionary containing episode data
            key: Data key to extract ('states' or 'actions')
            parts: List of parts to include ('left_arm', 'right_arm')
            
        Returns:
            Concatenated numpy array of the requested data
        """
        result = []
        for sample_data in episode_data['data']:
            data_array = np.array([], dtype=np.float32)
            for part in parts:
                if part in sample_data[key] and sample_data[key][part] is not None:
                    qpos = np.array(sample_data[key][part]['qpos'], dtype=np.float32)
                    data_array = np.concatenate([data_array, qpos])
            result.append(data_array)
        return np.array(result)


    def _parse_images(self, episode_path: str, episode_data) -> dict[str, list[np.ndarray]]:
        """Load and stack images for a given camera key."""

        images = defaultdict(list)
        
        keys = episode_data["data"][0]['colors'].keys()
        cameras = [key for key in keys if "depth" not in key]
        for camera in cameras:
            image_key = self.camera_to_image_key.get(camera)
            if image_key is None:
                continue

            for sample_data in episode_data['data']:
                relative_path = sample_data['colors'].get(camera)
                if not relative_path:
                    continue

                image_path = os.path.join(episode_path, relative_path)
                if camera == "left_image":
                    # derive relative / absolute depth path
                    rel_depth = (
                        Path(relative_path)
                        .with_name(Path(relative_path).name       # left_image_000000.jpg → left_depth_000000.jpg
                                .replace("left_image_", "left_depth_"))
                        .with_suffix(".png")                  # jpg → png
                    )
                else:
                    rel_depth = (
                        Path(relative_path)
                        .with_name(Path(relative_path).name       # right_image_000000.jpg → right_depth_000000.jpg
                                .replace("right_image_", "right_depth_"))
                        .with_suffix(".png")                  # jpg → png
                    )
                depth_path = Path(episode_path) / str(rel_depth).replace("colors", "depths")
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Image path does not exist: {image_path}")

                image = cv2.imread(depth_path)
                # image = cv2.imread(image_path)
                if image is None:
                    if images[image_key]:
                        # copy() so later in‑place ops don’t mutate previous frame
                        images[image_key].append(images[image_key][-1].copy())
                        continue
                    else:
                        raise RuntimeError(
                            f"First depth frame missing: {depth_path}"
                        )

                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                images[image_key].append(image_rgb)

        return images
    
    def _reconstruct_point_clouds(self, episode_path: str):
        """
        Reconstruct point clouds from disparities.
        
        Args:
            episode_path: Path to the episode directory
            
        Returns:
            Point clouds as a numpy array of shape (T, 4096, 3)
        """
        point_clouds = []
        disp_path = os.path.join(episode_path, "disparities", "disparities_episode.npy")
        disps = np.load(disp_path, allow_pickle=True) if os.path.exists(disp_path) else None
        if disps is None:
            raise FileNotFoundError(f"Disparity file not found: {disp_path}")
        for disp in disps:
            # Assuming depth_img is a single-channel image with depth values
            points = cv2.reprojectImageTo3D(disp.squeeze(0), self.Q).reshape(-1, 3)  # Reshape to (N, 3)
            points = uniform_sampling_numpy(points[None, ...], 4096).squeeze(0)  # (4096, 3)
            point_clouds.append(points)
        # Convert to numpy array of shape (T, 4096, 3)
        point_clouds = np.array(point_clouds, dtype=np.float32)
        # some sanity checks
        if point_clouds.shape[0] == 0:
            raise ValueError("No valid depth images found in the episode data.")
        if point_clouds.shape[1] != 4096:
            raise ValueError(f"Expected 4096 points per frame, got {point_clouds.shape[1]} points.")
        return point_clouds
    
    
    def _process_retraction_episode(
        self,
        episode_data: Dict,
        concat_states: bool = True,
        rel_ee_actions: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns (states, actions) for one episode.

        states  : (T, 7)  -> current [qpos(6) | gripper]
        actions : (T-1,7) -> *target* joint pos at t+1 (absolute), last sample dropped
        """
        # ---------- Collect joint + gripper per frame ----------
        dissection_tar = []
        qtraj = []
        qvel = []
        qeffort = []
        ee_frames = []
        gripper = []
        action = []
        for sample in episode_data["data"]:
            # robot states
            js   = sample["states"]["psm_retraction_js"]
            # dissection target points in pixel coordinates
            dissection_tar_ = sample["states"]["dissection_target"]["points"]
            dissection_tar_ = np.array(dissection_tar_).flatten().astype(np.int32)
            # end-effector frames 
            ee_frames_pos = sample["states"]["psm_retraction_ee"]["psm_retraction_pos"]
            ee_frames_quat = sample["states"]["psm_retraction_ee"]["psm_retraction_quat"]
            ee_frames_ = np.concatenate([ee_frames_pos, ee_frames_quat])
            # states
            qpos_ = js["qpos"]          # 6-D list
            qvel_ = js["qvel"]          # 6-D list
            qeffort_ = js["qeffort"]    # 6-D list
            grip = js["gripper"]        # scalar
            if concat_states:
                qtraj.append(qpos_ + [grip] + qvel_ + qeffort_)
            else:
                qtraj.append(qpos_ + [grip])
                qvel.append(qvel_)  
                qeffort.append(qeffort_)  
                
            ee_frames.append(ee_frames_) 
            dissection_tar.append(dissection_tar_)
            gripper.append(grip)
            if rel_ee_actions:
                action = None
            else:
                action.append(qpos_ + [grip]) # use current joint positions as action

        states = np.asarray(qtraj, dtype=np.float32)           # (T,7)
        qvel = np.asarray(qvel, dtype=np.float32)             # (T,7)
        qeffort = np.asarray(qeffort, dtype=np.float32)       # (T,7)
        
        ee_frames = np.asarray(ee_frames, dtype=np.float32)   # (T,7)
        dissection_tar = np.asarray(dissection_tar, dtype=np.float32)  # (T, 8)
        gripper = np.asarray(gripper, dtype=np.float32)       # (T,1)

        if rel_ee_actions:
            actions_ = tool_centric_relative_actions(torch.from_numpy(ee_frames))  # (T-1, 4, 4)
            actions = se3_to_10d_actions(actions_, torch.from_numpy(gripper[:-1]))  # (T-1, 10)
        else:
            action = np.asarray(action, dtype=np.float32)         # (T,7)
            # ---------- Create one-step-ahead targets --------------
            # Drop the final frame because it has no “next” pose.
            actions = action[1:].copy()                            # (T-1,7)

        if concat_states:
            # Drop the last frame of states, actions, qvel, qeffort, dissection_tar
            return states[:-1], actions, dissection_tar[:-1]
        else:
            return states[:-1], actions, qvel[:-1], qeffort[:-1], dissection_tar[:-1]



    def get_item(self, index: Optional[int] = None, concat_states: bool = False) -> Dict:
        """Get a training sample from the dataset.  """
            
        file_path = np.random.choice(self.episode_paths) if index is None else self.episode_paths[index]
        episode_data = self.episodes_data_cached[index]

        if concat_states:
            # Load state and action data (retraction only)
            state, action, dissection_tar = self._process_retraction_episode(episode_data, concat_states=True)
        else:
            state, action, qvel, qeffort, dissection_tar = self._process_retraction_episode(episode_data, concat_states=False)
            
        episode_length = len(state)

        # Ensure state and action have the same length
        state_dim = state.shape[1] if len(state.shape) == 2 else state.shape[0]
        action_dim = action.shape[1] if len(action.shape) == 2 else state.shape[0]

        # Load task description
        task = episode_data.get('text', {}).get('goal', "")
        
        # Load camera images
        cameras = self._parse_images(file_path[:-9], episode_data) #ignore the last 9 characters which is data.json

        # Reconstruct point clouds (T, 4096, 3)
        point_clouds = self._reconstruct_point_clouds(file_path[:-9]) 
        # Extract camera configuration
        cam_height, cam_width = next(img for imgs in cameras.values() if imgs for img in imgs).shape[:2]
        data_cfg = {
            'camera_names': list(cameras.keys()),
            'cam_height': cam_height,
            'cam_width': cam_width,
            'state_dim': state_dim,
            'action_dim': action_dim,
        }
        if concat_states:
            return {'episode_index': index,
                    'episode_length': episode_length,
                    'state': state, 
                    'dissection_tar': dissection_tar,
                    'point_clouds': point_clouds,
                    'action': action,
                    'cameras': cameras,
                    'task': task,
                    'data_cfg':data_cfg}
        else:
            return {'episode_index': index,
                    'episode_length': episode_length,
                    'state': state, 
                    'dissection_tar': dissection_tar,
                    'point_clouds': point_clouds,
                    'action': action,
                    'qvel': qvel,
                    'qeffort': qeffort,
                    'cameras': cameras,
                    'task': task,
                    'data_cfg':data_cfg}


def create_empty_dataset(
    repo_id: str,
    robot_type: str,
    mode: Literal["video", "image"] = "video",
    *,
    has_point_cloud: bool = False,
    has_velocity: bool = False,
    has_effort: bool = False,
    has_dissection_target: bool = False,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
    FPS: int = 30,
) -> LeRobotDataset:
    
    motors = ROBOT_CONFIGS[robot_type].motors
    cameras = ROBOT_CONFIGS[robot_type].cameras

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (19,), # 6 joints + 1 gripper + 6 velocity + 6 effort
            "names": [
                motors,
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": (7,), # 3 pos + 6D orientation + gripper  # 6 joints + 1 gripper (relative joint values)
            "names": [
                motors,
            ],
        },
    }

    if has_point_cloud:
        features["observation.point_cloud"] = {
            "dtype": "float32",
            "shape": (4096, 3),  # 3D points, 4096 points per frame
            "names": [
                "x",
                "y",
                "z",
            ],
        }

    if has_velocity:
        features["observation.velocity"] = {
            "dtype": "float32",
            "shape": (6,), # without gripper
            "names": [
                motors,
            ],
        }

    if has_effort:
        features["observation.effort"] = {
            "dtype": "float32",
            "shape": (6,), # without gripper
            "names": [
                motors,
            ],
        }

    if has_dissection_target:
        features["observation.dissection_tar"] = {
            "dtype": "float32",
            "shape": (8,), # without gripper
            "names": [
                "first_point_px",
                "first_point_py",
                "second_point_px",
                "second_point_py",
                "third_point_px",
                "third_point_py",
                "fourth_point_px",
                "fourth_point_py",
            ],
        }

    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": (3, 480, 640),
            "names": [
                "channels",
                "height",
                "width",
            ],
        }

    if Path(HF_LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(HF_LEROBOT_HOME / repo_id)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=FPS,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )

def process_disscection_target(
    dissection_tar: np.ndarray,
) -> np.ndarray:
    """
    Process dissection target to ensure it has exactly 8 points.
    If more than 8 points are provided, truncate to the first 8.
    """
    if dissection_tar.shape[0] > 8:
        # discard the first point 
        dissection_tar = dissection_tar[:8]
        print(f"Warning: Dissection target has more than 8 points, truncating to 8 points.")
    return dissection_tar

def populate_dataset(
    dataset: LeRobotDataset,
    raw_dir: Path,
    robot_type: str,
    concat_states: bool = True,
) -> LeRobotDataset:
    
    json_dataset = JsonDataset(raw_dir, robot_type)
    for i in tqdm.tqdm(range(len(json_dataset))):
        episode = json_dataset.get_item(i, concat_states=concat_states)
        state = episode["state"]
        action = episode["action"]
        cameras = episode["cameras"]
        task = episode["task"]
        episode_length = episode["episode_length"]
        dissection_tar = episode["dissection_tar"]
        point_clouds = episode["point_clouds"]

        if concat_states==False:
            qvel = episode["qvel"]
            qeffort = episode["qeffort"]

        num_frames = episode_length
        for i in range(num_frames):
            if concat_states:
                frame = {
                    "observation.state": state[i],
                    "action": action[i],
                    "task": task,
                    "observation.dissection_tar": process_disscection_target(dissection_tar[i]),
                    "observation.point_cloud": point_clouds[i],
                } 
            else:
                # With separate velocity and effort
                frame = {
                    "observation.state": state[i],
                    "action": action[i],
                    "task": task,
                    "observation.velocity": qvel[i],
                    "observation.effort": qeffort[i],
                    "observation.dissection_tar": process_disscection_target(dissection_tar[i]),
                }
                
            for camera, img_array in cameras.items():
                frame[f"observation.images.{camera}"] = img_array[i]
            dataset.add_frame(frame)

        dataset.save_episode()

    return dataset


def json_to_lerobot(
    raw_dir: Path,
    repo_id: str,
    robot_type: str,        # Unitree_Z1_Dual, Unitree_G1_Gripper, Unitree_G1_Dex3
    *,
    push_to_hub: bool = False,
    mode: Literal["video", "image"] = "video",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
):

    if (HF_LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(HF_LEROBOT_HOME / repo_id)

    dataset = create_empty_dataset(
        repo_id,
        robot_type=robot_type,
        mode=mode, 
        has_point_cloud=True, 
        has_effort=False, # False means effort is not stored separately
        has_velocity=False, # False means velocity is not stored separately
        has_dissection_target=True,
        dataset_config=dataset_config,
        FPS=30,  # Assuming a default FPS of 30
    )
    dataset = populate_dataset(
        dataset,
        raw_dir,
        robot_type=robot_type,
    )

    if push_to_hub:
        dataset.push_to_hub(upload_large_folder = True)


def local_push_to_hub(
        repo_id: str,
        root_path: Path,):

    dataset = LeRobotDataset(repo_id = repo_id, root = root_path)
    dataset.push_to_hub(upload_large_folder = True)


if __name__ == "__main__":
    tyro.cli(json_to_lerobot)