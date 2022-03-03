# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmpose.models import builder


def test_poseformer():
    """Test PoseFormer."""
    cfg = dict(type='PoseFormer')
    model = builder.build_backbone(cfg)
    input = torch.randn(16, 9, 17, 2)
    output = model(input)
    assert output.shape == (16, 1, 544)
