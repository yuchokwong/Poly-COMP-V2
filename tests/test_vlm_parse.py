import pytest

from tsr.encoders.vlm import (
    NoneBackend,
    VLMResult,
    caption_gallery,
    get_backend,
    parse_vlm_output,
)


def test_parse_fenced_json_with_alias_keys():
    text = """```json
    {
      "scene": "A dog sits in a burning room.",
      "text_in_image": "THIS IS FINE",
      "meaning": "Denial in a disaster.",
      "emotions": "calm, ironic",
      "metaphors": [{"metaphor": "fire", "meaning": "chaos"}, "coffee"],
      "keywords": ["dog", "fire", "coffee", "denial"]
    }
    ```
    Anything after the fence is ignored."""

    result = parse_vlm_output(text)
    assert result.scene.startswith("A dog")
    assert result.ocr_text == "THIS IS FINE"
    assert result.meaning == "Denial in a disaster."
    assert result.emotions == ["calm", "ironic"]
    assert result.metaphors[0] == {"metaphor": "fire", "meaning": "chaos"}
    assert result.metaphors[1] == {"metaphor": "coffee", "meaning": ""}
    assert result.keywords[:2] == ["dog", "fire"]
    assert result.raw == text


def test_parse_json_embedded_in_prose_and_ocr_alias():
    result = parse_vlm_output(
        'Sure, here it is: {"scene": "cat", "ocr": "hello", "metaphors": []} done'
    )
    assert result.scene == "cat"
    assert result.ocr_text == "hello"
    assert result.metaphors == []


def test_parse_limits_metaphors_and_keywords():
    result = parse_vlm_output(
        '{"scene":"x","metaphors":['
        '{"metaphor":"a","meaning":"b"},{"metaphor":"c","meaning":"d"},'
        '{"metaphor":"e","meaning":"f"},{"metaphor":"g","meaning":"h"}],'
        '"keywords":["1","2","3","4","5","6","7","8","9","10","11"]}'
    )
    assert len(result.metaphors) == 3
    assert len(result.keywords) == 10


@pytest.mark.parametrize(
    "text",
    [
        "no json here",
        '{"scene": "unterminated"',
        '{"scene": "bad escape \\q"}',
        '["not", "an", "object"]',
    ],
)
def test_invalid_vlm_output_raises(text):
    with pytest.raises(ValueError):
        parse_vlm_output(text)


def test_parse_repairs_unescaped_quotes_inside_a_value():
    """Qwen sometimes quotes a word inside the meaning without escaping it."""
    result = parse_vlm_output(
        '{"scene": "x", "meaning": "criticizes "scary" Jokers.", "metaphors": []}'
    )
    assert result.meaning == 'criticizes "scary" Jokers.'
    assert result.metaphors == []


def test_parse_repairs_a_missing_comma_between_members():
    result = parse_vlm_output(
        '{"metaphors": [{"metaphor": "Scorpion as a \'death lobster,\'"\n'
        '      "meaning": "a scorpion"}]}'
    )
    assert result.metaphors == [{"metaphor": "Scorpion as a 'death lobster,'", "meaning": "a scorpion"}]


def test_parse_merges_a_value_split_across_adjacent_strings():
    """The model can emit two strings for one key; keep the text, not a crash."""
    result = parse_vlm_output('{"ocr_text": "LEFT", "RIGHT", "meaning": "m"}')
    assert result.ocr_text == "LEFT, RIGHT"
    assert result.meaning == "m"


def test_parse_salvages_complete_fields_from_a_truncated_object():
    """A generate call cut off by max_new_tokens keeps its finished members."""
    result = parse_vlm_output('{"scene": "a box of crackers", "ocr_text": "honey 130 130 6 6 12 12')
    assert result.scene == "a box of crackers"
    assert result.ocr_text == ""


def test_doc_text_is_ordered_and_skips_empty_fields():
    result = VLMResult(
        scene="a cat",
        ocr_text="",
        meaning="politeness",
        emotions=["awkward"],
        metaphors=[{"metaphor": "table", "meaning": "formality"}],
        keywords=["cat"],
    )
    doc = result.to_doc_text()
    assert doc.splitlines() == [
        "scene: a cat",
        "meaning: politeness",
        "metaphors: table means formality",
        "emotions: awkward",
        "keywords: cat",
    ]


def test_get_backend_disabled_and_unknown(cfg):
    cfg.models.vlm["backend"] = "none"
    backend = get_backend(cfg)
    assert isinstance(backend, NoneBackend)
    assert backend.available is False
    assert "disabled" in backend.error

    cfg.models.vlm["backend"] = "mystery"
    assert isinstance(get_backend(cfg), NoneBackend)
    assert "Unknown" in get_backend(cfg).error


def test_caption_gallery_rejects_an_unavailable_backend(cfg):
    with pytest.raises(RuntimeError, match="disabled"):
        caption_gallery(cfg, store=object(), backend=NoneBackend("VLM disabled"))


class _FakeBackend:
    """Backend that never calls a model; returns a fixed description."""

    name = "fake"
    model = "fake-vlm"
    available = True
    error = None
    calls = 0

    def describe(self, image_path):
        _FakeBackend.calls += 1
        return VLMResult(scene="a cat", ocr_text="hello", meaning="greeting", raw="{}")


def _make_images(cfg, store):
    from tsr.config import images_dir

    for record in store.iter_all():
        image = images_dir(cfg) / record.path
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(b"stub")


class _BatchBackend(_FakeBackend):
    """Records the batch sizes it was asked to caption."""

    batch_sizes: list[int] = []
    fail_batches = False

    def describe_many(self, image_paths):
        _BatchBackend.batch_sizes.append(len(image_paths))
        if _BatchBackend.fail_batches and len(image_paths) > 1:
            raise RuntimeError("CUDA out of memory")
        _FakeBackend.calls += len(image_paths)
        return [
            VLMResult(scene=f"batch image {i}", ocr_text="", meaning="joke", raw="{}")
            for i in range(len(image_paths))
        ]


def test_caption_gallery_captions_in_batches(cfg, store):
    _make_images(cfg, store)
    cfg.models.vlm["batch_size"] = 4
    _BatchBackend.batch_sizes = []
    _BatchBackend.fail_batches = False
    _FakeBackend.calls = 0

    summary = caption_gallery(cfg, store=store, backend=_BatchBackend())

    assert summary["ok"] == 3
    assert _BatchBackend.batch_sizes == [3]
    assert store.vlm_count() == 3
    assert _BatchBackend.calls == 3


def test_caption_gallery_falls_back_to_single_images_when_a_batch_fails(cfg, store):
    _make_images(cfg, store)
    cfg.models.vlm["batch_size"] = 4
    _BatchBackend.batch_sizes = []
    _BatchBackend.fail_batches = True
    _FakeBackend.calls = 0

    summary = caption_gallery(cfg, store=store, backend=_BatchBackend())

    assert summary["ok"] == 3
    assert summary["errors"] == 0
    assert _BatchBackend.batch_sizes == [3]  # one failed batch, then per-image
    assert _FakeBackend.calls == 3
    assert store.vlm_count() == 3


def test_caption_gallery_reupserts_cached_docs_when_database_lost_them(cfg, store, records):
    """A gallery rebuild drops vlm_docs; resuming must not leave strategy C empty."""
    from tsr.config import images_dir

    for record in store.iter_all():
        image = images_dir(cfg) / record.path
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(b"stub")

    backend = _FakeBackend()
    _FakeBackend.calls = 0
    first = caption_gallery(cfg, store=store, backend=backend)
    assert first["ok"] == 3

    # simulate `build --strategy clip --force`: rows cascade away, cache remains
    store.clear("images")
    store.insert_images(records)
    assert store.iter_vlm_docs() == []

    second = caption_gallery(cfg, store=store, backend=backend)
    assert second["ok"] == 0
    assert second["skipped"] == 3
    assert _FakeBackend.calls == 3  # only the first run called the model
    assert len(store.iter_vlm_docs()) == 3
