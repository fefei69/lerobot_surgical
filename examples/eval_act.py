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
import numpy
import torch

# from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.common.policies.act.modeling_act import ACTPolicy

from glob import glob
from PIL import Image
import torch
from torchvision.transforms import ToTensor

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
pretrained_policy_path = "outputs/train/act_phantom_retraction/checkpoints/060000/pretrained_model"
# OR a path to a local outputs/train folder.
# pretrained_policy_path = Path("outputs/train/example_pusht_diffusion")

policy = ACTPolicy.from_pretrained(pretrained_policy_path, local_files_only=True)

left_imgs, _ = load_episode_images("dataset/phantom_retraction/episode_0000/colors", "left_image")
right_imgs, _ = load_episode_images("dataset/phantom_retraction/episode_0000/colors", "right_image")


# We can verify that the shapes of the features expected by the policy match the ones from the observations
# produced by the environment
print(policy.config.input_features)

# Similarly, we can check that the actions produced by the policy will match the actions expected by the
# environment
print(policy.config.output_features)

# Reset the policy and environments to prepare for rollout
policy.reset()



step = 0
done = False
while not done:
    # Prepare observation for the policy running in Pytorch
    state = torch.randn(7)  # Random state for the example
    left_img = left_imgs[step % len(left_imgs)]  # Get the left image
    right_img = right_imgs[step % len(right_imgs)]  # Get the right image

    # Convert to float32 with image from channel first in [0,255]
    # to channel last in [0,1]
    state = state.to(torch.float32)
    left_img = left_img.to(torch.float32) / 255
    right_img = right_img.to(torch.float32) / 255

    # Send data tensors from CPU to GPU
    state = state.to(device, non_blocking=True)
    left_img = left_img.to(device, non_blocking=True)
    right_img = right_img.to(device, non_blocking=True)

    # Add extra (empty) batch dimension, required to forward the policy
    state = state.unsqueeze(0)
    left_img = left_img.unsqueeze(0)
    right_img = right_img.unsqueeze(0)

    # Create the policy input dictionary
    observation = {
        "observation.state": state,
        "observation.images.cam_left": left_img,
        "observation.images.cam_right": right_img,
    }

    # Predict the next action with respect to the current observation
    with torch.inference_mode():
        action = policy.select_action(observation)
        print(f"Step {step}: Predicted action: {action}")

    # Prepare the action for the environment
    numpy_action = action.squeeze(0).to("cpu").numpy()




    # The rollout is considered done when the success state is reached (i.e. terminated is True),
    # or the maximum number of iterations is reached (i.e. truncated is True)
    step += 1
    done = step >= 10  # For this example, we stop after 10 steps

