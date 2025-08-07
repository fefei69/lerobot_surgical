# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This script demonstrates how to evaluate a pretrained policy from the HuggingFace Hub or from your local
training outputs directory. In the latter case, you might want to run examples/3_train_policy.py first.

It requires the installation of the 'gym_pusht' simulation environment. Install it by running:
```bash
pip install -e ".[pusht]"
```
"""

from pathlib import Path

import gym_pusht  # noqa: F401
import gymnasium as gym
import imageio
import numpy as np
import torch

# from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.common.policies.act.modeling_act import ACTPolicy

from glob import glob
from PIL import Image
import torch
from torchvision.transforms import ToTensor
import tqdm
from utils import JsonDataset

def load_episode_images(episode_dir: str | Path,
                        view_prefix: str = "left_image"):
    """
    Load all *{view_prefix}_######.jpg* files in episode_dir
    and return a stacked tensor (T, C, H, W) plus the filenames.

    Parameters
    ----------
    episode_dir : str | Path
        Folder containing the episode images.
    view_prefix : str
        Filename prefix, e.g. 'left_image' or 'right_image'.

    Returns
    -------
    imgs : torch.Tensor
        4-D tensor of shape (T, C, H, W).
    files : list[str]
        Sorted list of image paths that were loaded.
    """
    episode_dir = Path(episode_dir)
    pattern = episode_dir / f"{view_prefix}_*.jpg"
    files = sorted(glob(str(pattern)))
    if not files:
        raise FileNotFoundError(f"No images matched {pattern}")

    imgs = [ToTensor()(Image.open(fp).convert("RGB")) for fp in files]
    return torch.stack(imgs), files

# Create a directory to store the video of the evaluation
# output_directory = Path("outputs/eval/example_pusht_diffusion")
# output_directory.mkdir(parents=True, exist_ok=True)

# Select your device
device = "cuda"

# Provide the [hugging face repo id](https://huggingface.co/lerobot/diffusion_pusht):
pretrained_policy_path = "outputs/train/act_retraction_predict_js_30hz_eff_vel_depimg_pc/checkpoints/040000/pretrained_model"
# OR a path to a local outputs/train folder.
# pretrained_policy_path = Path("outputs/train/example_pusht_diffusion")
policy = ACTPolicy.from_pretrained(pretrained_policy_path, local_files_only=True)
import pdb; pdb.set_trace()  # Debugging breakpoint
json_dataset = JsonDataset('dataset/phantom_retraction', 'DVRK')


# We can verify that the shapes of the features expected by the policy match the ones from the observations
# produced by the environment
print(policy.config.input_features)

# Similarly, we can check that the actions produced by the policy will match the actions expected by the
# environment
print(policy.config.output_features)

# Reset the policy and environments to prepare for rollout
policy.reset()



step = 0
# Random episode index to evaluate
id = np.random.randint(0, len(json_dataset))
# for i in tqdm.tqdm(range(len(json_dataset))):
#     episode = json_dataset.get_item(i)

episode = json_dataset.get_item(id)
all_state = episode["state"]
all_action = episode["action"]
all_cameras = episode["cameras"]
all_task = episode["task"]
episode_length = episode["episode_length"]

num_frames = episode_length
accumulated_error = 0.0
for i in range(num_frames):
    frame = {
        "observation.state": all_state[i],
        "action": all_action[i],
        "task": all_task
    }
    for camera, img_array in all_cameras.items():
        frame[f"observation.images.{camera}"] = img_array[i]

    # Prepare observation for the policy running in Pytorch
    # Convert the state and images to tensors
    frame["observation.state"] = torch.tensor(frame["observation.state"], dtype=torch.float32)
    frame["observation.images.cam_left"] = torch.tensor(frame["observation.images.cam_left"], dtype=torch.float32)
    frame["observation.images.cam_right"] = torch.tensor(frame["observation.images.cam_right"], dtype=torch.float32)
    

    # Convert to float32 with image from channel first in [0,255]
    # to channel last in [0,1]
    state = frame["observation.state"]
    left_img = frame["observation.images.cam_left"] / 255
    right_img = frame["observation.images.cam_right"] / 255

    # make image channel first
    left_img = left_img.permute(2, 0, 1)
    right_img = right_img.permute(2, 0, 1)

    # Send data tensors from CPU to GPU
    state = state.to(device, non_blocking=True)
    left_img = left_img.to(device, non_blocking=True)
    right_img = right_img.to(device, non_blocking=True)

    # Add extra (empty) batch dimension, required to forward the policy
    state = state.unsqueeze(0)
    left_img = left_img.unsqueeze(0)
    right_img = right_img.unsqueeze(0)
    import pdb; pdb.set_trace()  # Debugging breakpoint

    # Create the policy input dictionary
    observation = {
        "observation.state": state, 
        "observation.images.cam_left": left_img, 
        "observation.images.cam_right": right_img, 
    }

    # Predict the next action with respect to the current observation
    with torch.inference_mode():
        action = policy.select_action(observation)
        # print(f"Step {step}: Predicted action: {action}")

    # Prepare the action for the environment
    numpy_action = action.squeeze(0).to("cpu").numpy()
    # L2 norm of the error between the predicted action and the ground truth action
    ground_truth_action = all_action[i]
    error = np.linalg.norm(numpy_action - ground_truth_action)
    print(f"Step {step}: Jaw action: {numpy_action[-1]} Error: {error}")

    accumulated_error += error

    # The rollout is considered done when the success state is reached (i.e. terminated is True),
    # or the maximum number of iterations is reached (i.e. truncated is True)
    step += 1
    if step >= episode_length:
        print(f"Episode {id} finished after {step} steps.")
        print(f"Accumulated error: {accumulated_error}")
        import pdb; pdb.set_trace()  # Debugging breakpoint

