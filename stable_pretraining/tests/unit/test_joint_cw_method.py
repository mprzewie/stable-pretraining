from types import SimpleNamespace

import pytest
import torch
from torch import nn

from stable_pretraining.forward import joint_cw
from stable_pretraining.methods.joint_cw import JointCW

pytestmark = pytest.mark.unit


class _FakeBackbone(nn.Module):
    num_features = 3

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return images


def _model(monkeypatch: pytest.MonkeyPatch, **kwargs) -> JointCW:
    monkeypatch.setattr(
        "stable_pretraining.methods.joint_cw.timm.create_model",
        lambda *args, **model_kwargs: _FakeBackbone(),
    )
    return JointCW(projector=nn.Identity(), **kwargs)


def test_joint_cw_preserves_pairwise_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model(monkeypatch, beta=0.8, w_plus=0.25)

    assert model.w_plus == 0.25
    assert model.jcw_gg.w_plus == 0.25
    assert model.jcw_gl.w_plus == 0.25
    assert model.jcw_ll.w_plus == 0.25
    assert not hasattr(model, "multiview_cw")


def test_joint_cw_returns_separate_pair_group_losses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model(monkeypatch)
    monkeypatch.setattr(
        "stable_pretraining.methods.joint_cw._grouped_pair_diagnostics",
        lambda *args, **kwargs: {},
    )
    global_views = [
        torch.randn(4, 3, requires_grad=True),
        torch.randn(4, 3, requires_grad=True),
    ]
    local_views = [torch.randn(4, 3, requires_grad=True) for _ in range(6)]

    output = model(global_views=global_views, local_views=local_views)

    assert output.gg_loss is not None
    assert output.gl_loss is not None
    assert output.ll_loss is not None
    assert torch.allclose(
        output.loss,
        (output.gg_loss + output.gl_loss + output.ll_loss) / 3,
    )
    output.loss.backward()
    for view in global_views + local_views:
        assert view.grad is not None
        assert torch.isfinite(view.grad).all()


def test_joint_cw_forward_adapter_uses_pairwise_model() -> None:
    class _Wrapper:
        def __init__(self) -> None:
            self.calls: list[dict[str, list[torch.Tensor]]] = []
            self.logged: list[str] = []

        def model(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(
                loss=torch.tensor(1.0),
                embedding=torch.randn(4, 3),
                diagnostics={"gg/cw_joint": torch.tensor(0.25)},
            )

        def log(self, name: str, *args, **kwargs) -> None:
            self.logged.append(name)

    wrapper = _Wrapper()
    batch = {
        "global_1": {"image": torch.full((2, 3), 1.0), "label": torch.arange(2)},
        "global_2": {"image": torch.full((2, 3), 2.0), "label": torch.arange(2)},
        **{
            f"local_{index}": {
                "image": torch.full((2, 3), float(index + 2)),
                "label": torch.arange(2),
            }
            for index in range(1, 7)
        },
    }

    result = joint_cw(wrapper, batch, "fit")

    call = wrapper.calls[0]
    assert [view[0, 0].item() for view in call["global_views"]] == [1.0, 2.0]
    assert [view[0, 0].item() for view in call["local_views"]] == [
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
    ]
    assert result["label"].shape == (4,)
    assert wrapper.logged == ["fit/loss", "fit/gg/cw_joint"]
