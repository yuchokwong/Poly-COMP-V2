import json
import shutil
import zipfile
from pathlib import Path

import pytest

pytest.importorskip("PIL")

from PIL import Image  # noqa: E402

from tsr.config import (  # noqa: E402
    archive_path,
    manifest_path,
    raw_captions_dir,
    raw_images_dir,
)
from tsr.manifest import read_manifest  # noqa: E402
from tsr.memecap import (  # noqa: E402
    _to_rgb,
    extract_archive,
    normalize_record,
    parse_annotations,
    prepare_gallery,
)


def _raw(fname, **overrides):
    data = {
        "img_fname": fname,
        "post_id": fname.split(".")[0],
        "title": "A title",
        "url": "https://example.com/meme",
        "img_captions": ["a cat at a table"],
        "meme_captions": ["trying to be polite"],
        "metaphors": [{"metaphor": "cat", "meaning": "meme poster"}],
        "category": "memes",
    }
    data.update(overrides)
    return data


def test_parse_annotations_infers_split_and_cleans_fields(tmp_path):
    captions = tmp_path / "captions"
    captions.mkdir()
    (captions / "memes-test.json").write_text(
        json.dumps([_raw("test_one.png"), _raw("shared.png")]),
        encoding="utf-8",
    )
    (captions / "memes-trainval.json").write_text(
        json.dumps(
            {
                "data": [
                    _raw("train_one.png", title=float("nan")),
                    _raw("shared.png", title="ignored duplicate"),
                ]
            }
        ),
        encoding="utf-8",
    )

    records = parse_annotations(captions)
    assert set(records) == {"test_one.png", "train_one.png", "shared.png"}
    assert records["test_one.png"]["split"] == "test"
    assert records["train_one.png"]["split"] == "trainval"
    assert records["train_one.png"]["title"] is None
    # The test file sorts first, so setdefault keeps its copy.
    assert records["shared.png"]["title"] == "A title"


def test_normalize_record_drops_invalid_metaphors():
    record = normalize_record(
        _raw(
            "nested/path/memes_1.png",
            title="  ",
            metaphors=[
                {"metaphor": "fire", "meaning": "chaos"},
                {"metaphor": "", "meaning": "no metaphor"},
                {"metaphor": "rain", "meaning": ""},
                "not-a-dict",
            ],
            category=None,
        ),
        "test",
    )
    assert record is not None
    assert record["img_fname"] == "memes_1.png"
    assert record["title"] is None
    assert record["category"] == "memes"
    assert record["metaphors"] == [{"metaphor": "fire", "meaning": "chaos"}]
    assert normalize_record({"img_fname": None}, "test") is None


def test_to_rgb_composites_transparency_on_white():
    rgba = Image.new("RGBA", (4, 4), (0, 0, 0, 0))
    rgb = _to_rgb(rgba)
    assert rgb.mode == "RGB"
    assert rgb.getpixel((0, 0)) == (255, 255, 255)


def _prepare_raw(cfg):
    raw_dir = raw_images_dir(cfg)
    raw_dir.mkdir(parents=True)
    raw_captions_dir(cfg).mkdir(parents=True)

    Image.new("RGB", (300, 200), (200, 30, 30)).save(raw_dir / "a.png")
    shutil.copyfile(raw_dir / "a.png", raw_dir / "b.png")  # identical bytes
    Image.new("RGB", (300, 200), (30, 30, 200)).save(raw_dir / "c.png")  # no annotation
    Image.new("RGB", (50, 40), (30, 200, 30)).save(raw_dir / "d.png")

    (raw_captions_dir(cfg) / "memes-test.json").write_text(
        json.dumps([_raw("a.png"), _raw("b.png"), _raw("missing.png")]),
        encoding="utf-8",
    )
    (raw_captions_dir(cfg) / "memes-trainval.json").write_text(
        json.dumps([_raw("d.png")]),
        encoding="utf-8",
    )


def test_prepare_gallery_resizes_dedupes_and_writes_manifest(cfg):
    _prepare_raw(cfg)
    cfg.dataset["max_side"] = 100
    cfg.dataset["quality"] = 80

    assert prepare_gallery(cfg) == 2
    manifest = read_manifest(manifest_path(cfg))
    assert [record.image_id for record in manifest] == ["memecap_a", "memecap_d"]
    by_id = {record.image_id: record for record in manifest}
    assert by_id["memecap_a"].split == "test"
    assert by_id["memecap_a"].path == "memecap/a.jpg"
    assert by_id["memecap_d"].split == "trainval"

    dst = Path(cfg.paths.data_dir) / "images" / "memecap" / "a.jpg"
    with Image.open(dst) as image:
        assert max(image.size) <= 100
        assert image.mode == "RGB"

    meta = json.loads(
        (Path(cfg.paths.data_dir) / "manifest.meta.json").read_text(encoding="utf-8")
    )
    assert meta["num_records"] == 2
    assert meta["num_duplicates"] == 1
    assert meta["num_annotations"] == 4
    assert meta["num_without_annotations"] == 1

    # Existing manifest short-circuits unless forced.
    assert prepare_gallery(cfg) == 2
    assert prepare_gallery(cfg, limit=1, force=True) == 1
    assert [record.image_id for record in read_manifest(manifest_path(cfg))] == ["memecap_a"]


def test_extract_archive_only_keeps_images_and_captions(cfg):
    zip_path = archive_path(cfg)
    zip_path.parent.mkdir(parents=True)
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("memecap/images/x.png", b"fake png")
        zf.writestr("memecap/caption/memes-test.json", b"{}")
        zf.writestr("memecap/other.txt", b"ignored")
        zf.writestr("__MACOSX/memecap/images/y.png", b"ignored")

    out = extract_archive(cfg)
    assert out.exists()
    assert (raw_images_dir(cfg) / "x.png").read_bytes() == b"fake png"
    assert (raw_captions_dir(cfg) / "memes-test.json").exists()
    assert not (raw_images_dir(cfg) / "y.png").exists()

    marker = out / ".extract_complete.json"
    assert json.loads(marker.read_text(encoding="utf-8"))["zip_size"] == zip_path.stat().st_size
    # A completed marker makes the call a no-op.
    (raw_images_dir(cfg) / "x.png").write_bytes(b"changed")
    extract_archive(cfg)
    assert (raw_images_dir(cfg) / "x.png").read_bytes() == b"changed"
