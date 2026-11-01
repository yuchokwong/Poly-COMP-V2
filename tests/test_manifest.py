import json

from tsr.manifest import (
    ManifestRecord,
    content_hash,
    manifest_sha256,
    read_manifest,
    write_manifest,
)


def test_from_dict_cleans_nan_title_and_bad_numbers():
    record = ManifestRecord.from_dict(
        {
            "image_id": "memes_1",
            "img_fname": "memes_1.png",
            "split": "test",
            "path": "memecap/memes_1.jpg",
            "title": float("nan"),
            "width": "not-a-number",
            "height": 480,
        }
    )
    assert record.title is None
    assert record.width is None
    assert record.height == 480
    assert record.category == "memes"


def test_manifest_roundtrip(tmp_path, records):
    path = tmp_path / "manifest.jsonl"
    write_manifest(records, path)
    loaded = read_manifest(path)
    assert loaded == records
    assert path.read_text(encoding="utf-8").count("\n") == len(records)

    first = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert first["image_id"] == "memes_00000"
    assert first["metaphors"][0]["meaning"] == "meme poster"

    original = manifest_sha256(path)
    write_manifest(records[:-1], path)
    assert manifest_sha256(path) != original
    assert read_manifest(tmp_path / "missing.jsonl") == []


def test_content_hash_tracks_annotation_text_not_only_ids(tmp_path, records):
    original = content_hash(records)
    assert content_hash(records) == original

    records[1].img_captions = ["a corrected caption"]
    assert content_hash(records) != original
    # The file hash changes too, but content_hash works on records read back
    # from SQLite, which is what the build guard compares.
    path = tmp_path / "manifest.jsonl"
    write_manifest(records, path)
    assert content_hash(read_manifest(path)) == content_hash(records)


def test_meaningful_metaphors_filters_literal_entries():
    record = ManifestRecord(
        image_id="memes_1",
        img_fname="memes_1.png",
        split="test",
        path="memecap/memes_1.jpg",
        metaphors=[
            {"metaphor": "woman", "meaning": "meme poster"},
            {"metaphor": "background", "meaning": "not related to the meme context"},
            {"metaphor": "", "meaning": "empty"},
            {"metaphor": "  fire  ", "meaning": "  chaos  "},
            "not-a-dict",
        ],
    )
    assert record.meaningful_metaphors() == [
        {"metaphor": "woman", "meaning": "meme poster"},
        {"metaphor": "fire", "meaning": "chaos"},
    ]


def test_content_hash_covers_the_split(records):
    original = content_hash(records)
    records[0].split = "trainval"
    assert content_hash(records) != original


def test_content_hash_covers_the_image_bytes(records):
    """A re-issued file with the same name must invalidate the gallery guards."""
    original = content_hash(records)
    records[0].sha1 = "0" * 40
    assert content_hash(records) != original
