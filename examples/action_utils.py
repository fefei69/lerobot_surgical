
import tqdm

from utils import JsonDataset

import torch
import torch.nn.functional as F

# -----------------------------------------------
# copy from pytorch3d.transforms
# -----------------------------------------------
def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)

def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """
    Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
    by dropping the last row. Note that 6D representation is not unique.
    Args:
        matrix: batch of rotation matrices of size (*, 3, 3)

    Returns:
        6D rotation representation, of size (*, 6)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    batch_dim = matrix.size()[:-2]
    return matrix[..., :2, :].clone().reshape(batch_dim + (6,))

def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)

def poses_to_se3_and_rel(
    g_t: torch.Tensor,
    scalar_last: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args
    ----
    g_t : (B, 7) tensor
        Pose batch as concatenated position + quaternion.
        • If `scalar_last` is True (default) the quaternion order is (qx,qy,qz,qw),
          matching `pytorch3d.transforms.quaternion_to_matrix`.
        • If your data is (qw,qx,qy,qz) set `scalar_last=False`.
    scalar_last : bool
        Controls whether to roll the scalar‐component (w) to the end before conversion.

    Returns
    -------
    T    : (B, 4, 4)  batch of SE(3) matrices.
    A_t  : (B‑1, 4, 4) relative transforms s.t.  A_t = T_t⁻¹ @ T_{t+1}.
    """
    if g_t.ndim != 2 or g_t.shape[1] != 7:
        raise ValueError("g_t must have shape (B, 7)")

    pos, quat = g_t[:, :3], g_t[:, 3:]           # (B,3)  (B,4)

    if not scalar_last:                          # move the first value to end if needed
        quat = torch.cat([quat[:, 1:], quat[:, :1]], dim=-1)

    # (B,3,3) rotation matrices
    R = quaternion_to_matrix(quat)

    B = g_t.shape[0]
    T = torch.eye(4, device=g_t.device, dtype=g_t.dtype).expand(B, 4, 4).clone()
    T[:, :3, :3] = R
    T[:, :3,  3] = pos

    # --- relative transforms A_t = T_t^{-1} @ T_{t+1} ---
    # invert all but the last, then batch‑matmul
    T_inv = torch.linalg.inv(T[:-1])          # (B‑1,4,4)
    A_t   = T_inv @ T[1:]                     # (B‑1,4,4)

    return T, A_t


# ----------------------------------------------------------------------
# 1. Pose (x, y, z, r, i, j, k)  →  SE(3)  (H, 4, 4)
#    pytorch3d expects quaternion in (w, x, y, z) order.
# ----------------------------------------------------------------------
def pose7_to_se3(poses_7: torch.Tensor) -> torch.Tensor:
    """
    poses_7: (H, 7)  [x, y, z, r, i, j, k]  (scalar‑first quaternion)
    returns: (H, 4, 4) batch of SE(3) matrices
    """
    xyz   = poses_7[:, :3]                       # translations
    quat  = poses_7[:, 3:]                       # (r,i,j,k)
    quat = standardize_quaternion(quat)  # positive real part
    quat  = quat / quat.norm(dim=-1, keepdim=True)          # unit‑norm

    R     = quaternion_to_matrix(quat)           # (H, 3, 3)

    H     = poses_7.shape[0]
    g     = torch.eye(4, device=poses_7.device, dtype=poses_7.dtype).expand(H, 4, 4).clone()
    g[:, :3, :3] = R
    g[:, :3,  3] = xyz
    return g

# ----------------------------------------------------------------------
# 2. Tool‑centric actions  A_t = g_t^{-1} g_{t+1}
# ----------------------------------------------------------------------
def relative_actions(g: torch.Tensor) -> torch.Tensor:
    """g: (H,4,4) → actions A: (H‑1,4,4)"""
    return torch.linalg.inv(g[:-1]) @ g[1:]

# ----------------------------------------------------------------------
# 3. Sanity check  g_t @ A_t = g_{t+1}
# ----------------------------------------------------------------------
def tool_centric_relative_actions(ee_frames: torch.Tensor, atol: float = 1e-6):
    """
    ee_frames : (H,7) tensor of EE poses
    atol      : numerical tolerance for equality check
    """
    g = pose7_to_se3(ee_frames)   # (H,4,4)
    A = relative_actions(g)            # (H‑1,4,4)

    # verify   g_t A_t ≈ g_{t+1}
    recon = g[:-1] @ A
    assert torch.allclose(recon, g[1:], atol=atol), \
        f"❌  g_t @ A_t differs from g_(t+1) by > {atol}"

    print(f"✓ Sanity check passed (g_t @ A_t ≈ g_(t+1) within {atol})")
    return A

def se3_to_10d_actions(A: torch.Tensor, gripper: torch.Tensor) -> torch.Tensor:
    """
    Convert SE(3) actions A_t = g_t⁻¹ g_{t+1} to 10D representation.
    A: (H-1, 4, 4) tensor of relative transforms.
    gripper: (H-1,) tensor of gripper states.
    Returns: (H-1, 10) tensor of actions [dx, dy, dz, 6D rotation, gripper].
    """
    assert A.shape[-2:] == (4, 4), "A must be a batch of SE(3) matrices"
    
    # Extract translation and quaternion from the action matrix
    translations = A[:, :3, 3]  # (H-1, 3)
    rotation_matrices = A[:, :3, :3]  # (H-1, 3, 3)
    roations_6d = matrix_to_rotation_6d(rotation_matrices)  # (H-1, 6)

    return torch.cat((translations, roations_6d, gripper.unsqueeze(-1)), dim=-1)  # (H-1, 10)



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