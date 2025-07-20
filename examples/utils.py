import os
import cv2
import numpy as np
import tqdm
import json
import glob
import dataclasses
import shutil
from pathlib import Path
from collections import defaultdict
from typing import Literal, List, Dict, Optional


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
                    self.episode_paths.append(episode_paths[0]) # only take the data.json path, ignore colors/
        
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
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Image path does not exist: {image_path}")

                image = cv2.imread(image_path)
                if image is None:
                    raise RuntimeError(f"Failed to read image: {image_path}")

                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                images[image_key].append(image_rgb)

        return images
    
    def _process_retraction_episode(
            self,
            episode_data: Dict,
            part_key: str = "psm_retraction_js",
        ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (state, action) for the retraction arm only.

        • state  : shape (T, 7)  [qpos(6), gripper(1)]
        • action : shape (T, 7)  Δ-state; last frame is 0-vector
        """
        # ----– collect joint + gripper per frame –--------------------------------
        states = []
        ee_frames = []
        for sample in episode_data["data"]:
            js    = sample["states"][part_key]
            qpos  = js["qpos"]                 # 6-dim list
            grip  = js["gripper"]              # scalar
            states.append(qpos + [grip])       # 7-dim
            ee_frame_pos: list[float] = sample["states"]["psm_retraction_ee"]["psm_retraction_pos"]
            ee_frame_quat: list[float] = sample["states"]["psm_retraction_ee"]["psm_retraction_quat"]
            ee_frames.append(ee_frame_pos + ee_frame_quat)  # 3 + 4 = 7-dim

        states = np.asarray(states, dtype=np.float32)      # (T, 7)
        ee_frames = np.asarray(ee_frames, dtype=np.float32)  # (T, 7)

        # ----– first-order finite difference → action –---------------------------
        actions            = np.zeros_like(states)         # (T, 7)
        actions[:-1]       = states[1:] - states[:-1]      # Δq_t
        # actions[-1] is already zero

        return states, actions, ee_frames


    def get_item(self, index: Optional[int] = None,) -> Dict:
        """Get a training sample from the dataset.  """
            
        file_path = np.random.choice(self.episode_paths) if index is None else self.episode_paths[index]
        episode_data = self.episodes_data_cached[index]

        # Load state and action data (retraction only)
        state, action, ee_frames = self._process_retraction_episode(episode_data)
        # state = self._extract_data(episode_data, 'states', self.json_state_data_name)
        # action = self._extract_data(episode_data, 'actions', self.json_action_data_name)
        episode_length = len(state)
        state_dim = state.shape[1] if len(state.shape) == 2 else state.shape[0]
        action_dim = action.shape[1] if len(action.shape) == 2 else state.shape[0]

        # Load task description
        task = episode_data.get('text', {}).get('goal', "")
        
        # Load camera images
        cameras = self._parse_images(file_path[:-9], episode_data) #ignore the last 9 characters which is data.json

        # Extract camera configuration
        cam_height, cam_width = next(img for imgs in cameras.values() if imgs for img in imgs).shape[:2]
        data_cfg = {
            'camera_names': list(cameras.keys()),
            'cam_height': cam_height,
            'cam_width': cam_width,
            'state_dim': state_dim,
            'action_dim': action_dim,
        }
        
        # TODO: Add an if statement to handle different action types (future states or tool-centric actions)
        return {'episode_index': index,
                'episode_length': episode_length,
                'state': state, 
                'action': action,
                'ee_frames': ee_frames,
                'cameras': cameras,
                'task': task,
                'data_cfg':data_cfg}
    

def main():
    json_dataset = JsonDataset('dataset/', 'DVRK')

    for i in tqdm.tqdm(range(len(json_dataset))):
        episode = json_dataset.get_item(i)

        state = episode["state"]
        action = episode["action"]
        cameras = episode["cameras"]
        task = episode["task"]
        episode_length = episode["episode_length"]

        num_frames = episode_length
        for i in range(num_frames):
            frame = {
                "observation.state": state[i],
                "action": action[i],
                "task": task
            }
            import pdb; pdb.set_trace()  # Debugging breakpoint

            # for camera, img_array in cameras.items():
            #     frame[f"observation.images.{camera}"] = img_array[i]

if __name__ == "__main__":
    main()