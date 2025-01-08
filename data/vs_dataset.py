import os
import glob
import torch
import locket
import numpy as np
from typing import Dict
from torch.utils.data import IterableDataset, DataLoader


class PklReader(object):
    def __init__(self, data_root: str):
        self.data_root = data_root
        self.update_filelist()

    def update_filelist(self):
        self.data_flist = glob.glob(
            os.path.join(self.data_root, "**", "*.pt"), recursive=True
        )
        self.data_flist.sort()
        assert len(self.data_flist) > 0
        print(
            "[INFO] File list updated, available files: {}".format(len(self.data_flist))
        )

    @property
    def num_files(self):
        return len(self.data_flist)

    def get(self, i: int):
        data_path = self.data_flist[i]
        lock_path = data_path.replace("/data/", "/lock/", -1).replace(".pt", ".loc")
        with locket.lock_file(lock_path):
            data = torch.load(data_path, map_location="cpu")
        return data


class PklDataset(IterableDataset):
    def __init__(self, data_root: str):
        self.reader = PklReader(data_root)
        # self.aug = LightAug()
        self.aug = None

    def compute_flow(self, data: Dict[str, torch.Tensor]):
        current_pcd_cam = data["current_pcds"]  # (3, H, W)
        current_norm_xy = current_pcd_cam[:2] / current_pcd_cam[2:3]
        current_wcT = data["current_poses"]  # (4, 4)
        current_mask = data["current_masks"] > 0.5
        desired_wcT = data["desired_poses"]  # (4, 4)
        tcT = torch.inverse(desired_wcT) @ current_wcT

        # apply rotation
        desired_pcd_cam_proj = torch.einsum("rc,chw->rhw", tcT[:3, :3], current_pcd_cam)
        # apply translation
        desired_pcd_cam_proj = desired_pcd_cam_proj + tcT[:3, 3][:, None, None]

        desired_norm_xy_proj = desired_pcd_cam_proj[:2] / desired_pcd_cam_proj[2:3]
        norm_flow = desired_norm_xy_proj - current_norm_xy  # (2, H, W)
        norm_flow = torch.clip(norm_flow, -2, 2)
        proj_depth = desired_pcd_cam_proj[-1]
        proj_depth_valid_mask = (proj_depth > 0.01) & (proj_depth < 5)  # (H, W)
        norm_flow_no_nan_mask = ~norm_flow.isnan().any(dim=0)  # (H, W)

        intrinsic = data["intrinsics"].float()
        fxy = intrinsic[[0, 1], [0, 1]]
        cxy = intrinsic[[0, 1], [2, 2]]
        flow = norm_flow * fxy[:, None, None]  # (2, H, W)

        H, W = current_mask.shape
        current_xx, current_yy = torch.meshgrid(
            torch.arange(W), torch.arange(H), indexing="xy"
        )
        global_xy = torch.stack([current_xx, current_yy], dim=0)  # (2, H, W)
        norm_xy = (global_xy.float() - cxy[:, None, None]) / fxy[:, None, None]

        desired_xx = current_xx + flow[0]
        desired_yy = current_yy + flow[1]
        proj_in_img = (
            (desired_xx >= 0) & (desired_xx < W) & (desired_yy >= 0) & (desired_yy < H)
        )

        loss_mask = (
            current_mask & proj_in_img & norm_flow_no_nan_mask & proj_depth_valid_mask
        )
        flow.masked_fill_(~loss_mask[None], 0.0)
        return norm_xy, flow, loss_mask

    def get_desired_depth_hint(self, data: Dict[str, torch.Tensor]):
        pose = data["desired_poses"]
        dist = torch.norm(pose[:3, 3], dim=-1)  # always pointing to scene center
        assert dist < 1.5
        return dist

    def expand_to_bbox_mask(self, mask: torch.Tensor):
        H, W = mask.shape
        rr, cc = torch.nonzero(mask, as_tuple=True)
        bbox_mask = mask.new_zeros(H, W)
        bbox_mask[rr.min() : rr.max() + 1, cc.min() : cc.max() + 1] = 1
        return bbox_mask

    def __iter__(self):
        self.reader.update_filelist()
        f_idx = np.random.permutation(self.reader.num_files).tolist()
        for fi in f_idx:
            data = self.reader.get(fi)
            num_samples = data[next(iter(data.keys()))].shape[0]
            indices = np.random.permutation(num_samples).tolist()

            for i in indices:
                indexed_data = {k: v[i].contiguous() for k, v in data.items()}
                if self.aug:
                    for k in ["current_rgbs", "desired_rgbs"]:
                        indexed_data[k] = self.aug(
                            indexed_data[k].unsqueeze(0)
                        ).squeeze(0)

                norm_xy, flow, loss_mask = self.compute_flow(indexed_data)
                if loss_mask.float().mean() < 0.02:
                    continue

                if np.random.rand() < 0.5:
                    indexed_data["desired_masks"] = self.expand_to_bbox_mask(
                        indexed_data["desired_masks"]
                    )

                indexed_data["current_norm_xys"] = norm_xy
                indexed_data["desired_norm_xys"] = norm_xy
                indexed_data["flows"] = flow
                indexed_data["loss_masks"] = loss_mask
                indexed_data["depth_hint"] = self.get_desired_depth_hint(indexed_data)
                yield indexed_data


def get_dataloader(batch_size, num_workers):
    # dset = PklDataset("/home/chenanzhe/data_ssd/TFPose/v0903")
    # dset = PklDataset("/home/chenanzhe/data_ssd/TFPose/v0904")
    # dset = PklDataset("/home/chenanzhe/data_ssd/TFPose/v0905")
    dset = PklDataset("/home/chenanzhe/data_ssd/TFPose/v0915")
    dataloader = DataLoader(
        dataset=dset,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    return dataloader


def test_norm_flow():
    import cv2
    from einops import rearrange

    train_loader = get_dataloader(1, 0)
    for i, data in enumerate(train_loader):
        b_idx = 0
        current_rgb = rearrange(
            data["current_rgbs"][b_idx].cpu().numpy(), "c h w -> h w c"
        )
        desired_rgb = rearrange(
            data["desired_rgbs"][b_idx].cpu().numpy(), "c h w -> h w c"
        )
        rgb_pair = np.concatenate([current_rgb, desired_rgb], axis=1)
        bgr_pair = np.ascontiguousarray(rgb_pair[:, :, [2, 1, 0]])

        flow = rearrange(data["flows"][b_idx].cpu().numpy(), "c h w -> h w c")
        loss_mask = data["loss_masks"][b_idx].cpu().numpy()  # (H, W)
        yy, xx = np.nonzero(loss_mask)

        # rand_indices = np.random.permutation(len(yy))[:len(yy)//10]
        rand_indices = np.random.permutation(len(yy))[:256]
        yy = yy[rand_indices]
        xx = xx[rand_indices]

        flow = flow[yy, xx]  # (N, 2)
        desired_xx = flow[:, 0] + xx
        desired_yy = flow[:, 1] + yy

        H, W = desired_rgb.shape[:2]
        bgr_pair_wi_flow = bgr_pair.copy()
        for i in range(len(flow)):
            cv2.line(
                bgr_pair_wi_flow,
                pt1=(xx[i], yy[i]),
                pt2=(int(desired_xx[i] + W), int(desired_yy[i])),
                color=(0, 255, 0),
                thickness=1,
            )

        show = np.concatenate([bgr_pair, bgr_pair_wi_flow], axis=0)
        cv2.imshow("current | desired (debug)", show)
        key = cv2.waitKey(0)
        if key == ord("q"):
            break


if __name__ == "__main__":
    # test_dataloader()
    # iter_dataloader()
    # stat_depth()
    test_norm_flow()
