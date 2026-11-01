import sqlite3

import pytest

from tsr.store import SCHEMA_VERSION, GalleryStore, SchemaVersionError


def test_schema_version_and_insert_order(store, records):
    version = store.conn.execute("PRAGMA user_version").fetchone()[0]
    assert version == SCHEMA_VERSION
    assert store.count() == len(records)
    assert [record.id for record in store.iter_all()] == [1, 2, 3]
    assert store.get_meta("missing", "fallback") == "fallback"


def test_annotations_and_metaphors_are_enriched(store):
    loaded = store.get_many([1, 2])
    store.enrich(loaded)
    assert loaded[2].img_captions == ["a cat sitting at a dinner table"]
    assert loaded[2].meme_captions == ["pretending to be polite at dinner"]
    assert loaded[2].metaphors == [{"metaphor": "object 1", "meaning": "meme poster"}]

    loaded_all = store.get_many([1, 2, 3])
    store.enrich(loaded_all)
    assert loaded_all[3].metaphors == [{"metaphor": "fire", "meaning": "chaos"}]

    annotations = store.annotations_for([1, 3])
    assert annotations[1]["img_caption"]
    assert annotations[3]["meme_caption"]


def test_clear_images_restarts_ids(store, records):
    store.clear("images")
    assert store.count() == 0
    ids = store.insert_images(records)
    assert ids == [1, 2, 3]


def test_bm25_field_weights_change_ranking(store):
    store.write_nlp_docs(
        [
            (1, "cat", "", "", ""),
            (2, "", "cat", "", ""),
        ]
    )
    # meaning is column 2; weight it heavily and pk 2 should win.
    meaning_heavy = store.bm25(["cat"], 10, (1.0, 10.0, 1.0, 1.0))
    assert meaning_heavy[0][0] == 2
    assert meaning_heavy[0][1] > 0

    scene_heavy = store.bm25(["cat"], 10, (10.0, 1.0, 1.0, 1.0))
    assert scene_heavy[0][0] == 1


def test_bm25_fts_triggers_follow_rewrites(store):
    store.write_nlp_docs([(1, "cat", "", "", "")])
    assert store.bm25(["cat"], 10, (1, 1, 1, 1))

    store.write_nlp_docs([(1, "dog", "", "", "")])
    assert store.bm25(["cat"], 10, (1, 1, 1, 1)) == []
    assert store.bm25(["dog"], 10, (1, 1, 1, 1))[0][0] == 1


def test_bm25_rejects_unsafe_terms(store):
    store.write_nlp_docs([(1, "cat", "", "", "")])
    assert store.bm25(["cat", "bad-token!"], 10, (1, 1, 1, 1))[0][0] == 1
    assert store.bm25(["!!!"], 10, (1, 1, 1, 1)) == []
    assert store.bm25([], 10, (1, 1, 1, 1)) == []
    assert store.bm25(["cat"], 0, (1, 1, 1, 1)) == []


def test_strategy_docs_are_replaced_per_strategy(store):
    assert store.write_strategy_docs("clip_nlp", [(1, "one"), (2, "two")]) == 2
    store.write_strategy_docs("vlm", [(3, "three")])
    assert store.iter_strategy_docs("clip_nlp") == [(1, "one"), (2, "two")]
    assert store.strategy_docs_for("clip_nlp", [1, 3]) == {1: "one"}

    store.write_strategy_docs("clip_nlp", [(2, "two-b")])
    assert store.iter_strategy_docs("clip_nlp") == [(2, "two-b")]


def test_vlm_docs_upsert_and_coverage(store):
    base = {
        "image_pk": 1,
        "backend": "fake",
        "model": "fake-vlm",
        "prompt_version": "v1",
        "scene": "a cat",
        "doc_text": "scene: a cat",
        "status": "ok",
    }
    store.upsert_vlm_doc(base)
    store.upsert_vlm_doc({**base, "scene": "a dog", "doc_text": "scene: a dog"})
    assert store.vlm_count() == 1
    doc = store.get_vlm_doc(1)
    assert doc["scene"] == "a dog"

    store.upsert_vlm_doc(
        {
            **base,
            "image_pk": 2,
            "status": "error",
            "doc_text": "",
            "error": "boom",
        }
    )
    assert store.vlm_count() == 1
    assert store.get_vlm_doc(2) is None
    assert [doc["image_pk"] for doc in store.iter_vlm_docs()] == [1]


def test_meta_roundtrip(store):
    store.set_meta("built", {"count": 3})
    assert store.get_meta("built") == {"count": 3}


def test_old_schema_is_rejected(tmp_path):
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE images (id INTEGER PRIMARY KEY)")
    conn.execute("PRAGMA user_version=0")
    conn.commit()
    conn.close()

    with pytest.raises(SchemaVersionError):
        GalleryStore(path)


def test_future_schema_is_rejected(tmp_path):
    path = tmp_path / "future.db"
    conn = sqlite3.connect(path)
    conn.execute(f"PRAGMA user_version={SCHEMA_VERSION + 1}")
    conn.commit()
    conn.close()

    with pytest.raises(SchemaVersionError):
        GalleryStore(path)
