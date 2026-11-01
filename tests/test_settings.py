import json
from pathlib import Path

import pytest

from tsr.settings import apply_persisted, apply_settings, get_settings, save_settings


def test_get_settings_marks_editable_fields(cfg):
    fields = {field["key"]: field for field in get_settings(cfg)}
    assert fields["strategies.clip_nlp.weights.clip"]["value"] == 0.5
    assert fields["strategies.clip_nlp.weights.clip"]["editable"] is True
    assert fields["models.clip.name"]["editable"] is False
    assert fields["dataset.name"]["value"] == "memecap"


def test_apply_settings_coerces_and_sets_nested_keys(cfg):
    apply_settings(
        cfg,
        {
            "app.default_top_k": "20",
            "strategies.clip.prompt_ensemble": "true",
            "strategies.vlm.clip_weight": "0.35",
        },
    )
    assert cfg.app.default_top_k == 20
    assert cfg.strategies.clip.prompt_ensemble is True
    assert abs(cfg.strategies.vlm.clip_weight - 0.35) < 1e-9


def test_apply_settings_rejects_readonly_and_bad_values(cfg):
    with pytest.raises(ValueError):
        apply_settings(cfg, {"models.clip.name": "whatever"})
    with pytest.raises(ValueError):
        apply_settings(cfg, {"not.a.setting": 1})
    with pytest.raises(ValueError):
        apply_settings(cfg, {"strategies.vlm.clip_weight": 5.0})
    with pytest.raises(ValueError):
        apply_settings(cfg, {"app.default_top_k": "many"})


def test_apply_settings_is_atomic_on_error(cfg):
    before = cfg.app.default_top_k
    with pytest.raises(ValueError):
        apply_settings(
            cfg,
            {"app.default_top_k": 7, "strategies.vlm.clip_weight": 99.0},
        )
    assert cfg.app.default_top_k == before


def test_save_and_reapply_settings(cfg, tmp_path):
    apply_settings(cfg, {"strategies.clip_nlp.weights.bm25": 0.45})
    save_settings(cfg)
    path = tmp_path / "settings.json"
    assert path.exists()
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["strategies.clip_nlp.weights.bm25"] == 0.45

    cfg.strategies.clip_nlp.weights["bm25"] = 0.2
    apply_persisted(cfg)
    assert cfg.strategies.clip_nlp.weights.bm25 == 0.45


def test_apply_persisted_ignores_invalid_entries(cfg):
    path = Path(cfg.paths.data_dir) / "settings.json"
    path.write_text(
        json.dumps(
            {
                "app.default_top_k": 15,
                "models.clip.name": "hacked",
                "strategies.vlm.clip_weight": "not-a-number",
            }
        ),
        encoding="utf-8",
    )
    apply_persisted(cfg)
    assert cfg.app.default_top_k == 15
    assert cfg.models.clip.name != "hacked"
