import pytest
import torch
from torch import nn

from stable_pretraining.methods.multi_cw_method import MultiCW

pytestmark = pytest.mark.unit


class _FakeBackbone(nn.Module):
    num_features = 3

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return images


def _model(monkeypatch: pytest.MonkeyPatch) -> MultiCW:
    monkeypatch.setattr(
        "stable_pretraining.methods.multi_cw_method.timm.create_model",
        lambda *args, **kwargs: _FakeBackbone(),
    )
    return MultiCW(projector=nn.Identity())


def test_multi_cw_configures_multiview_objective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model(monkeypatch)
    decomposition = model.multiview_cw.decomposition

    assert model.n_global == 2
    assert model.n_local == 6
    assert decomposition.rho_gg == 0.88
    assert decomposition.rho_gl == 0.72
    assert decomposition.rho_ll == 0.61
    assert not hasattr(model, "jcw_gg")


def test_multi_cw_preserves_image_grouping_and_view_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model(monkeypatch)
    captured: dict[str, torch.Tensor] = {}

    class _CaptureLoss(nn.Module):
        def forward(
            self,
            z: torch.Tensor,
        ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
            captured["z"] = z
            loss = z.square().mean()
            return loss, {"cw/joint": loss.detach()}

    model.multiview_cw = _CaptureLoss()
    monkeypatch.setattr(
        "stable_pretraining.methods.multi_cw_method._grouped_pair_diagnostics",
        lambda *args, **kwargs: {},
    )

    batch_size = 4
    global_views = [
        torch.full((batch_size, 3), float(index), requires_grad=True)
        for index in (1, 2)
    ]
    local_views = [
        torch.full((batch_size, 3), float(index), requires_grad=True)
        for index in range(3, 9)
    ]

    output = model(global_views=global_views, local_views=local_views)

    assert captured["z"].shape == (batch_size, 8, 3)
    assert torch.equal(
        captured["z"][:, :, 0],
        torch.arange(1, 9, dtype=torch.float32).expand(batch_size, -1),
    )
    assert output.loss is output.cw_loss
    assert output.diagnostics is not None
    assert not output.diagnostics["cw/joint"].requires_grad

    output.loss.backward()
    for view in global_views + local_views:
        assert view.grad is not None
        assert torch.isfinite(view.grad).all()


def test_multi_cw_rejects_misaligned_views(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model(monkeypatch)
    globals_ = [torch.randn(4, 3), torch.randn(4, 3)]
    locals_ = [torch.randn(4, 3) for _ in range(6)]
    locals_[2] = torch.randn(3, 3)

    with pytest.raises(ValueError, match="same batch size"):
        model(global_views=globals_, local_views=locals_)
