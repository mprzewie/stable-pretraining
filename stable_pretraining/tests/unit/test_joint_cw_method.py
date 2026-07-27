import pytest
import torch
from torch import nn

from stable_pretraining.methods.joint_cw import JointCW

pytestmark = pytest.mark.unit


def test_joint_cw_passes_w_plus_to_pair_losses(monkeypatch):
    class FakeBackbone(nn.Module):
        num_features = 16

        def forward(self, images):
            return torch.zeros(images.shape[0], self.num_features, device=images.device)

    monkeypatch.setattr(
        "stable_pretraining.methods.joint_cw.timm.create_model",
        lambda *args, **kwargs: FakeBackbone(),
    )

    model = JointCW(beta=0.8, w_plus=0.25)

    assert model.w_plus == 0.25
    assert model.jcw_gg.w_plus == 0.25
    assert model.jcw_gl.w_plus == 0.25
    assert model.jcw_ll.w_plus == 0.25
