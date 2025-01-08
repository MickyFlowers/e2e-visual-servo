import torch
from torch import nn, optim, Tensor
from models.dino import Dinov2
from models.attention.models import Transformer
import pytorch_lightning as pl
from scipy.spatial.transform import Rotation as R
import torch.nn.functional as F
import torchvision.transforms as T
import math
from utils.utils import *
from torch.autograd import gradcheck
import cv2


class Policy(pl.LightningModule):
    def __init__(
        self,
        lr,
        transformer_config,
        dino_model="dinov2_vitg14",
        use_local=True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.learning_rate = lr
        self.encoder = Dinov2(dino_model, use_local=use_local).requires_grad_(False)
        dim_action_token = transformer_config["params"]["d_model_de"]
        dim_action = transformer_config["params"]["out_dim"]
        self.action_token = nn.Parameter(
            torch.zeros(1, 1, dim_action_token), requires_grad=True
        )
        self.transformer: Transformer = instantiate_from_config(transformer_config)
        self.ln_norm_head = nn.Sequential(
            nn.Linear(dim_action, dim_action, bias=False),
            # nn.LayerNorm(dim_action, eps=1e-6),
            nn.LeakyReLU(inplace=True),
            nn.Linear(dim_action, dim_action, bias=False),
            # nn.LayerNorm(dim_action, eps=1e-6),
            nn.LeakyReLU(inplace=True),
            nn.Linear(dim_action, 1, bias=False),
        )
        self.dir_head = nn.Sequential(
            nn.Linear(dim_action, dim_action, bias=False),
            # nn.LayerNorm(dim_action, eps=1e-6),
            nn.LeakyReLU(inplace=True),
            nn.Linear(dim_action, dim_action, bias=False),
            # nn.LayerNorm(dim_action, eps=1e-6),
            nn.LeakyReLU(inplace=True),
            nn.Linear(dim_action, 6, bias=False),
        )

    def forward(self, cur_img: Tensor, tar_img: Tensor):
        b, _, _, _ = cur_img.shape
        with torch.no_grad():
            cur_dino_feature = self.encoder(cur_img)
            tar_dino_feature = self.encoder(tar_img)
        feature = torch.cat([cur_dino_feature, tar_dino_feature], dim=1)
        action_token = self.action_token.expand(b, 1, -1)
        action = self.transformer(feature, action_token)
        action = action.squeeze(1)
        ln_norm = self.ln_norm_head(action)
        dir = self.dir_head(action)
        return ln_norm, dir

    def reshape(self, image):
        b, c, h, w = image.shape
        reshape = T.Resize((h // 14 * 14, w // 14 * 14))
        return reshape(image)

    def training_step(self, batch, batch_idx):
        b, c, h, w = batch["current_rgbs"].shape
        cur_img = self.reshape(batch["current_rgbs"]) * 2.0 - 1.0
        tar_img = self.reshape(batch["desired_rgbs"]) * 2.0 - 1.0
        depth_hint = batch["depth_hint"]
        gt_dT = torch.bmm(torch.inverse(batch["desired_poses"]), batch["current_poses"])
        loss = self.calc_loss(cur_img, tar_img, depth_hint, gt_dT)
        self.log_dict(loss, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        return loss["total_loss"]

    def calc_loss(self, cur_img: Tensor, tar_img: Tensor, depth_hint, gt_dT):
        ln_norm, dir = self(cur_img, tar_img)
        gt_v, gt_w = self.pbvs(gt_dT)
        gt_v_dir_noscale = torch.cat([gt_v / depth_hint[:, None], gt_w], dim=-1)
        gt_v_dir_noscale_norm = torch.norm(gt_v_dir_noscale, dim=-1, keepdim=True)
        norm_loss = F.l1_loss(ln_norm, rev_exp_lin(gt_v_dir_noscale_norm))
        dir_loss = 1 - F.cosine_similarity(gt_v_dir_noscale, dir).mean()
        total_loss = norm_loss * 0.2 + dir_loss
        loss = {
            "norm_loss": norm_loss,
            "dir_loss": dir_loss,
            "total_loss": total_loss,
        }
        # if total_loss < 1:
        #     print("check loss")
        # cur_img_show = (cur_img * 0.5 + 0.5)[0].permute(1, 2, 0).cpu().numpy()
        # tar_img_show = (tar_img * 0.5 + 0.5)[0].permute(1, 2, 0).cpu().numpy()
        # print(gt_v_dir_noscale[0], gt_v_dir_noscale_norm[0])
        # cv2.imshow("cur_img", cur_img_show)
        # cv2.imshow("tar_img", tar_img_show)
        # cv2.waitKey(0)

        return loss

    def pbvs(self, dT: Tensor):

        u = matrix_to_axis_angle(dT[:, :3, :3])
        v = -torch.bmm(dT[:, :3, :3].transpose(1, 2), dT[:, :3, 3:4]).squeeze(-1)
        w = -u
        return v, w

    def configure_optimizers(self):
        params = (
            [self.action_token]
            + list(self.transformer.parameters())
            + list(self.ln_norm_head.parameters())
            + list(self.dir_head.parameters())
        )
        optimizer = optim.AdamW(params=params, lr=self.learning_rate)
        return optimizer


def exp_lin(x: Tensor, x0: float = 1.0):
    y = x.clone()
    mask = x < x0
    if mask.any():
        y[mask] = x0 / torch.e * torch.exp(x[mask] / x0)
    return y


def rev_exp_lin(y: torch.Tensor, x0: float = 1.0, eps=1e-5):
    x = y.clone()
    mask = y < x0
    if mask.any():
        x[mask] = x0 * (math.log(torch.e / x0) + torch.log(y[mask] + eps))
    return x
