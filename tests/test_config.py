import yaml

from tsr.config import (
    AttrDict,
    canonical_vlm_backend,
    load_config,
    manifest_path,
    strategy_index_dir,
    vlm_doc_filters,
    vlm_provenance,
    web_dir,
)


def test_default_config_has_strategy_sections():
    cfg = load_config()
    assert cfg.dataset.name == "memecap"
    assert cfg.models.clip.name
    assert cfg.models.text.name
    assert cfg.strategies.enabled == ["clip", "clip_nlp", "vlm"]
    assert cfg.strategies.clip_nlp.weights.clip > 0
    assert cfg.index.type in {"flat", "hnsw", "ivf", "pq"}
    assert cfg.app.port > 0


def test_override_file_is_merged_over_default(tmp_path):
    override = tmp_path / "override.yaml"
    override.write_text(
        yaml.safe_dump({"app": {"port": 9123}, "strategies": {"vlm": {"clip_weight": 0.0}}}),
        encoding="utf-8",
    )
    cfg = load_config(override)
    assert cfg.app.port == 9123
    assert cfg.strategies.vlm.clip_weight == 0.0
    # keys not present in the override still come from default.yaml
    assert cfg.app.host == "127.0.0.1"
    assert cfg.models.clip.pretrained


def test_env_override_uses_double_underscore(monkeypatch):
    monkeypatch.setenv("TSR_APP__PORT", "9234")
    monkeypatch.setenv("TSR_STRATEGIES__CLIP__PROMPT_ENSEMBLE", "true")
    cfg = load_config()
    assert cfg.app.port == 9234
    assert cfg.strategies.clip.prompt_ensemble is True


def test_relative_paths_resolve_under_project_or_data_dir():
    cfg = load_config()
    assert cfg.paths.data_dir.endswith("data")
    assert cfg.paths.cache_dir.startswith(cfg.paths.data_dir)
    assert manifest_path(cfg).name == "manifest.jsonl"
    assert strategy_index_dir(cfg, "clip").name == "clip"
    assert web_dir(cfg).name == "web"


def test_attrdict_get_and_attribute_access():
    node = AttrDict({"a": AttrDict({"b": 1})})
    assert node.a.b == 1
    assert node.a.get("missing", 5) == 5
    node.a.b = 2
    assert node["a"]["b"] == 2


def test_vlm_backend_aliases_share_one_canonical_name():
    # The captioner persists QwenVLBackend.name, so every alias must normalise
    # to that exact string or builds would filter out their own rows.
    assert canonical_vlm_backend("qwen") == "qwen2_5_vl"
    assert canonical_vlm_backend("Qwen2.5-VL") == "qwen2_5_vl"
    assert canonical_vlm_backend("qwen2_5_vl") == "qwen2_5_vl"
    assert canonical_vlm_backend("none") is None
    assert canonical_vlm_backend("disabled") is None


def test_vlm_provenance_reports_disabled_separately_from_no_filter():
    cfg = load_config()
    cfg.models.vlm.backend = "qwen2.5-vl"
    provenance = vlm_provenance(cfg)
    assert provenance["backend"] == "qwen2_5_vl"
    assert provenance["enabled"] is True
    assert "enabled" not in vlm_doc_filters(cfg)

    cfg.models.vlm.backend = "none"
    disabled = vlm_provenance(cfg)
    assert disabled["enabled"] is False
    assert disabled["backend"] is None
