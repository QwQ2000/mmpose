from .temporal_regression_head import TemporalRegressionHead
from mmpose.models.builder import HEADS

import numpy as np
import torch.nn as nn

from mmpose.core import WeightNormClipHook
from mmpose.models.builder import HEADS, build_loss

@HEADS.register_module()
class PoseFormerHead(TemporalRegressionHead):
    def __init__(self, embed_dim_ratio=32, num_joints=17, max_norm=None, loss_keypoint=None, is_trajectory=False, train_cfg=None, test_cfg=None):
        nn.Module.__init__(self)

        self.embed_dim_ratio = embed_dim_ratio
        self.num_joints = num_joints
        self.max_norm = max_norm
        self.loss = build_loss(loss_keypoint)
        self.is_trajectory = is_trajectory
        if self.is_trajectory:
            assert self.num_joints == 1

        self.train_cfg = {} if train_cfg is None else train_cfg
        self.test_cfg = {} if test_cfg is None else test_cfg

        self.fc = nn.Sequential(
            nn.LayerNorm(embed_dim_ratio * num_joints),
            nn.Linear(embed_dim_ratio * num_joints, num_joints * 3),
        )

        if self.max_norm is not None:
            # Apply weight norm clip to conv layers
            weight_clip = WeightNormClipHook(self.max_norm)
            for module in self.modules():
                if isinstance(module, nn.modules.linear.Linear):
                    weight_clip.register(module)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.shape[0], 1, self.num_joints, -1)
        x = x.squeeze(1)
        return x