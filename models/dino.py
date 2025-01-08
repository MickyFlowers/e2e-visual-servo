import torch
import torch.nn as nn


class Dinov2(nn.Module):
    def __init__(self, model: str, use_local=False):
        super().__init__()
        print(f"loading dinov2 model: {model}")
        if use_local:
            self.encoder = torch.hub.load(
                "/home/cyx/.cache/torch/hub/facebookresearch_dinov2_main",
                "dinov2_vits14",
                source="local",
                trust_repo=True,
            )
        else:
            self.encoder = torch.hub.load(
                "facebookresearch/dinov2", model, source="github", trust_repo=True
            )
        self.encoder.eval()

    def forward(self, x):
        feature_dict = self.encoder.forward_features(x)
        return feature_dict["x_norm_patchtokens"].detach()
