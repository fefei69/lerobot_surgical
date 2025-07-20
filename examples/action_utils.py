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

from utils import JsonDataset
from scipy.spatial.transform import Rotation as R

import torch
import torch.nn.functional as F

# -----------------------------------------------
# helpers
# -----------------------------------------------
def quat_to_rotmat(q):
    """
    q: (B, 4) quaternion in (x, y, z, w) order, unit or non‑unit.
    returns: (B, 3, 3) rotation matrices
    """
    q = F.normalize(q, dim=-1)           # ensure unit length
    x, y, z, w = q.unbind(-1)
    B = q.shape[0]

    R = torch.empty(B, 3, 3, device=q.device, dtype=q.dtype)
    R[:, 0, 0] = 1 - 2*(y*y + z*z)
    R[:, 0, 1] = 2*(x*y - z*w)
    R[:, 0, 2] = 2*(x*z + y*w)

    R[:, 1, 0] = 2*(x*y + z*w)
    R[:, 1, 1] = 1 - 2*(x*x + z*z)
    R[:, 1, 2] = 2*(y*z - x*w)

    R[:, 2, 0] = 2*(x*z - y*w)
    R[:, 2, 1] = 2*(y*z + x*w)
    R[:, 2, 2] = 1 - 2*(x*x + y*y)
    return R                            # (B,3,3)

def left_inv_delta(R_cur, t_cur, R_next, t_next):
    """
    batched left‑invariant SE(3) difference: g_cur^{-1} g_next
    R_cur, R_next: (B,3,3)
    t_cur, t_next: (B,3)
    returns R_rel (B,3,3), t_rel (B,3)
    """
    Rt = R_cur.transpose(-1, -2)         # Rᵀ
    R_rel = Rt @ R_next                  # rotation part
    t_rel = (Rt @ (t_next - t_cur)[..., None]).squeeze(-1)
    return R_rel, t_rel

def rotmat_to_6d(R):
    """
    Zhou et al. 6‑D representation = first two columns.
    R: (B,3,3) → (B,6)
    """
    return R[..., :2].reshape(R.shape[0], 6)

# -----------------------------------------------
# main function
# -----------------------------------------------
def poses_to_act10_batch(poses_world, jaw=None):
    """
    poses_world: (B,7) = [x y z qx qy qz qw]
    jaw        : (B,) or None   (# absolute or delta jaw angles)
    returns    : actions (B-1, 10)
    """
    t = poses_world[:, :3]
    q = poses_world[:, 3:]

    R = quat_to_rotmat(q)                # (B,3,3)

    R_rel, t_rel = left_inv_delta(R[:-1], t[:-1],
                                  R[1:],  t[1:])  # each pair
    rot6 = rotmat_to_6d(R_rel)           # (B-1,6)

    if jaw is None:
        jaw_delta = torch.zeros_like(t_rel[:, 0:1])
    else:
        jaw_delta = jaw[1:, None] - jaw[:-1, None]

    act10 = torch.cat([t_rel, rot6, jaw_delta], dim=-1)  # (B-1,10)
    return act10


# def quat_to_matrix(q):
#     """
#     q = [w, x, y, z] 
#     """
#     assert len(q) == 4, "Quaternion must have 4 elements"
#     return R.from_quat(q).as_matrix()     # 3×3

# def make_SE3(R_mat, p):
#     """
#     Create a 4x4 SE(3) transformation matrix from a rotation matrix and a translation vector.
#     R_mat: 3x3 rotation matrix
#     p: 3x1 translation vector
#     Returns a 4x4 transformation matrix.
#     """
#     assert R_mat.shape == (3, 3), "R_mat must be a 3x3 matrix"
#     assert p.shape == (3,), "p must be a 3-element vector"
#     T = np.eye(4, dtype=np.float32)
#     T[:3, :3] = R_mat
#     T[:3, 3]  = p
#     return T

# def process_relative_action(ee_frames):
#     # Extract the end-effector pose from the frames
#     for ee_frame in ee_frames:
#         pos = ee_frame[:3]
#         quat = ee_frame[3:7]
#         R_mat = quat_to_matrix(quat)
#         T = make_SE3(R_mat, pos)

#     return None


def main():
    json_dataset = JsonDataset('dataset/phantom_retraction/', 'DVRK')

    for i in tqdm.tqdm(range(len(json_dataset))):
        episode = json_dataset.get_item(i)

        state = episode["state"]
        action = episode["action"]
        cameras = episode["cameras"]
        task = episode["task"]
        episode_length = episode["episode_length"]
        ee_frames = episode["ee_frames"]
        # ee_frames_relative = process_relative_action(ee_frames)
        import pdb; pdb.set_trace()  # Debugging breakpoint

        num_frames = episode_length
        for i in range(num_frames):
            # Process the end-effector frame
            frame = {
                "observation.state": state[i],
                "action": action[i],
                "task": task,
            }
            import pdb; pdb.set_trace()  # Debugging breakpoint



if __name__ == "__main__":
    print("Testing action utils...")
    main()