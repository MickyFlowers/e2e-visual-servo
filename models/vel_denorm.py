import cv2
import torch
import numpy as np
from typing import Optional


K_training = np.array([[512, 0, 256], [0, 512, 256], [0, 0, 1]])


def sample_training_coords(
    dcR_norm: np.ndarray, dct_norm: np.ndarray, K_real: np.ndarray
):
    xx, yy = np.meshgrid(
        np.linspace(-0.5, 0.5, 10), np.linspace(-0.5, 0.5, 10), indexing="xy"
    )
    d_norm_coords = np.stack([xx, yy], axis=-1).reshape(-1, 2)  # (N, 2)

    train_fx = K_training[0, 0]
    train_fy = K_training[1, 1]
    train_cx = K_training[0, 2]
    train_cy = K_training[1, 2]
    dP_train = np.concatenate(
        [d_norm_coords, np.ones((len(d_norm_coords), 1))], axis=-1
    )  # (N, 3)

    real_fx = K_real[0, 0]
    real_fy = K_real[1, 1]
    real_cx = K_real[0, 2]
    real_cy = K_real[1, 2]
    xx_real = (xx * train_fx + train_cx - real_cx) / real_fx
    yy_real = (yy * train_fy + train_cy - real_cy) / real_fy
    d_norm_coords_real = np.stack([xx_real, yy_real], axis=-1).reshape(-1, 2)
    dP_real = np.concatenate(
        [d_norm_coords_real, np.ones((len(d_norm_coords_real), 1))], axis=-1
    )

    dcT = np.eye(4)
    dcT[:3, :3] = dcR_norm
    dcT[:3, 3] = dct_norm
    cdT = np.linalg.inv(dcT)

    cP = dP_train @ cdT[:3, :3].T + cdT[:3, 3]  # (N, 3)
    c_norm_coords = cP[:, :2] / cP[:, 2:3]  # (N, 2)
    current_pix_coords = c_norm_coords * np.array([train_fx, train_fy]) + np.array(
        [train_cx, train_cy]
    )

    valid_mask = np.abs(cP[:, -1]) > 1e-5
    return dP_real[valid_mask], current_pix_coords[valid_mask]


def find_pose(pts3d: np.ndarray, kpts_cur: np.ndarray, K_real: np.ndarray):
    success, rvecs, tvecs = cv2.solvePnP(pts3d, kpts_cur, K_real, None)
    assert success, "Failed to solve pose"
    cdT = np.eye(4)
    cdT[:3, :3] = cv2.Rodrigues(rvecs)[0]
    cdT[:3, 3] = np.asarray(tvecs).ravel()
    dcT = np.linalg.inv(cdT)
    return dcT


def denorm_numpy(
    dcT_norm: np.ndarray, d_star: int = 1, K_real: Optional[np.ndarray] = None
):
    dcR_norm = dcT_norm[:3, :3]
    dct_norm = dcT_norm[:3, 3]
    if K_real is None:
        K_real = K_training
    pts3d, kpts_cur = sample_training_coords(dcR_norm, dct_norm, K_real)
    dcT = find_pose(pts3d, kpts_cur, K_real)
    dcT[:3, 3] *= d_star
    # dcT = dcT_norm.copy()
    # dcT[:3, 3] *= d_star
    return dcT


def denorm_torch(
    dcT_norm: torch.Tensor, d_star: torch.Tensor, K_real: Optional[torch.Tensor] = None
):
    dcT = torch.zeros_like(dcT_norm, requires_grad=False)
    for b in range(dcT_norm.shape[0]):
        dcT_np = denorm_numpy(
            dcT_norm[b].detach().cpu().numpy(),
            d_star[b].item(),
            None if K_real is None else K_real[b].detach().cpu().numpy(),
        )
        dcT[b] = torch.from_numpy(dcT_np).to(dcT)
    return dcT


def infer_intrinsic_from_norm_xy_map(norm_xy: torch.Tensor):
    B, _, H, W = norm_xy.shape
    dxy = (
        norm_xy[:, :, H // 2, W // 2] - norm_xy[:, :, H // 2 - 1, W // 2 - 1]
    )  # (B, 2)
    fxy = 1.0 / dxy
    cxy = torch.tensor([H // 2, W // 2]).to(fxy) - fxy * norm_xy[:, :, H // 2, W // 2]
    K = norm_xy.new_zeros(B, 3, 3, requires_grad=False)
    K[:, [0, 1], [0, 1]] = fxy
    K[:, [0, 1], [2, 2]] = cxy
    K[:, -1, -1] = 1.0
    return K


if __name__ == "__main__":
    from scipy.spatial.transform import Rotation

    np.random.seed(0)
    # dcR_norm = Rotation.from_rotvec(np.random.rand(3) * 0.1).as_matrix()
    # a = np.random.rand(3) * 0.1
    # a[:2] = 0
    # dcR_norm = Rotation.from_rotvec(a).as_matrix()
    dct_norm = np.random.rand(3) * 0.1
    dcR_norm = np.eye(3)
    # dct_norm = np.zeros(3)
    d_star = 2

    testing_intrinsic = np.array([[400, 0, 256], [0, 400, 256], [0, 0, 1]])
    # testing_intrinsic = K_training

    pts3d, kpts_cur = sample_training_coords(dcR_norm, dct_norm, testing_intrinsic)
    dcT = find_pose(pts3d, kpts_cur, testing_intrinsic)
    dcT[:3, 3] *= d_star
    print(dcR_norm)
    print(dct_norm * d_star)
    print(dcT)
