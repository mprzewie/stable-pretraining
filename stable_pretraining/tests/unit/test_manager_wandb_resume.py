import json
from types import SimpleNamespace

from stable_pretraining import manager as manager_module


def test_cache_dir_run_ignores_cwd_wandb_resume(monkeypatch, tmp_path):
    cwd = tmp_path / "shared-cwd"
    run_dir = tmp_path / "fresh-run"
    cwd.mkdir()
    run_dir.mkdir()
    (cwd / "wandb_resume.json").write_text(
        json.dumps({"id": "unrelated", "project": "project", "entity": "entity"})
    )
    monkeypatch.chdir(cwd)

    logger = SimpleNamespace(
        _wandb_init={"project": "project", "entity": "entity"},
        _id=None,
    )
    monkeypatch.setattr(manager_module, "find_wandb_logger", lambda trainer: logger)

    manager = manager_module.Manager.__new__(manager_module.Manager)
    manager._trainer = object()
    manager._run_dir = run_dir
    manager.ckpt_path = None

    manager._maybe_restore_wandb_run_id()

    assert "id" not in logger._wandb_init
    assert logger._id is None
