import torch
from torch import Tensor, nn

import pytorch_lightning as pl

from models.attention.models import TransformerBlock
from models.dino import Dinov2
from utils import instantiate_from_config


class ToolAdaptive(pl.LightningModule):
    def __init__(
        self,
        lr,
        att_config,
        max_len=10000,
        dino_model="dinov2_vitg14",
        use_local=True,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.learning_rate = lr
        self.encoder = Dinov2(dino_model, use_local)
        self.att_block: TransformerBlock = instantiate_from_config(att_config)

    def forward(self, tool_img: Tensor, ref_img: Tensor):
        b, _, _, _ = tool_img.shape

        tool_dino_feature = self.encoder(tool_img)
        ref_dino_feature = self.encoder(ref_img)
        feature = self.att_block(tool_dino_feature, ref_dino_feature)
        return feature
    
