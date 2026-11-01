"""SQLite schema v2: the single source of image identity.

``images.id`` is the key shared by every FAISS index. Rows are inserted in a
deterministic order by the build pipeline, so row position and primary key stay
aligned. Strategy-specific documents live in separate tables and can be rebuilt
without touching image ids.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from tsr.logging_utils import get_logger

logger = get_logger(__name__)

SCHEMA_VERSION = 2
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS images (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id    TEXT UNIQUE NOT NULL,
    img_fname   TEXT UNIQUE,
    split       TEXT NOT NULL,
    path        TEXT NOT NULL,
    width       INTEGER,
    height      INTEGER,
    title       TEXT,
    source_url  TEXT,
    post_id     TEXT,
    sha1        TEXT,
    phash       TEXT,
    category    TEXT
);
CREATE INDEX IF NOT EXISTS idx_images_split ON images(split);
CREATE INDEX IF NOT EXISTS idx_images_sha1 ON images(sha1);

CREATE TABLE IF NOT EXISTS annotations (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    image_pk INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    kind     TEXT NOT NULL CHECK (kind IN ('img_caption', 'meme_caption')),
    ordinal  INTEGER NOT NULL DEFAULT 0,
    text     TEXT NOT NULL,
    UNIQUE(image_pk, kind, ordinal)
);
CREATE INDEX IF NOT EXISTS idx_annotations_image ON annotations(image_pk);

CREATE TABLE IF NOT EXISTS metaphors (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    image_pk INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    ordinal  INTEGER NOT NULL DEFAULT 0,
    metaphor TEXT NOT NULL,
    meaning  TEXT NOT NULL,
    UNIQUE(image_pk, ordinal)
);

CREATE TABLE IF NOT EXISTS vlm_docs (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    image_pk       INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    backend        TEXT NOT NULL,
    model          TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    scene          TEXT,
    ocr_text       TEXT,
    meaning        TEXT,
    emotions       TEXT,
    metaphors      TEXT,
    keywords       TEXT,
    doc_text       TEXT NOT NULL,
    raw            TEXT,
    status         TEXT NOT NULL DEFAULT 'ok',
    error          TEXT,
    latency_ms     REAL,
    created_at     TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(image_pk, backend, model, prompt_version)
);
CREATE INDEX IF NOT EXISTS idx_vlm_docs_pk ON vlm_docs(image_pk);

CREATE TABLE IF NOT EXISTS strategy_docs (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy   TEXT NOT NULL,
    image_pk   INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    text       TEXT NOT NULL,
    UNIQUE(strategy, image_pk)
);
CREATE INDEX IF NOT EXISTS idx_strategy_docs ON strategy_docs(strategy, image_pk);

CREATE TABLE IF NOT EXISTS nlp_docs (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    image_pk INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    scene    TEXT,
    meaning  TEXT,
    metaphor TEXT,
    title    TEXT,
    UNIQUE(image_pk)
);
CREATE VIRTUAL TABLE IF NOT EXISTS nlp_docs_fts USING fts5(
    scene, meaning, metaphor, title,
    content='nlp_docs', content_rowid='id',
    tokenize='unicode61 remove_diacritics 2'
);
CREATE TRIGGER IF NOT EXISTS nlp_docs_ai AFTER INSERT ON nlp_docs BEGIN
    INSERT INTO nlp_docs_fts(rowid, scene, meaning, metaphor, title)
    VALUES (new.id, new.scene, new.meaning, new.metaphor, new.title);
END;
CREATE TRIGGER IF NOT EXISTS nlp_docs_ad AFTER DELETE ON nlp_docs BEGIN
    INSERT INTO nlp_docs_fts(nlp_docs_fts, rowid, scene, meaning, metaphor, title)
    VALUES ('delete', old.id, old.scene, old.meaning, old.metaphor, old.title);
END;
CREATE TRIGGER IF NOT EXISTS nlp_docs_au AFTER UPDATE ON nlp_docs BEGIN
    INSERT INTO nlp_docs_fts(nlp_docs_fts, rowid, scene, meaning, metaphor, title)
    VALUES ('delete', old.id, old.scene, old.meaning, old.metaphor, old.title);
    INSERT INTO nlp_docs_fts(rowid, scene, meaning, metaphor, title)
    VALUES (new.id, new.scene, new.meaning, new.metaphor, new.title);
END;
"""


@dataclass
class ImageRecord:
    id: int
    image_id: str
    img_fname: str | None
    split: str
    path: str
    width: int | None = None
    height: int | None = None
    title: str | None = None
    source_url: str | None = None
    post_id: str | None = None
    sha1: str | None = None
    phash: str | None = None
    category: str | None = "memes"
    img_captions: list[str] = field(default_factory=list)
    meme_captions: list[str] = field(default_factory=list)
    metaphors: list[dict] = field(default_factory=list)

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> ImageRecord:
        return cls(
            id=int(row["id"]),
            image_id=row["image_id"],
            img_fname=row["img_fname"],
            split=row["split"],
            path=row["path"],
            width=row["width"],
            height=row["height"],
            title=row["title"],
            source_url=row["source_url"],
            post_id=row["post_id"],
            sha1=row["sha1"],
            phash=row["phash"],
            category=row["category"],
        )


class SchemaVersionError(RuntimeError):
    pass


class GalleryStore:
    def __init__(self, path: str | Path, reset: bool = False) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        version = int(self.conn.execute("PRAGMA user_version").fetchone()[0])
        if reset and version != SCHEMA_VERSION:
            self._drop_all()
            version = 0
        if version == 0:
            has_tables = self.conn.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='images'"
            ).fetchone()[0]
            if has_tables:
                raise SchemaVersionError(
                    "Existing database uses an old schema. Move it away or run "
                    "`python -m tsr clean --all --yes` before rebuilding."
                )
            self.conn.executescript(_SCHEMA)
            self.conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
            self.conn.commit()
        elif version != SCHEMA_VERSION:
            raise SchemaVersionError(
                f"Database schema v{version} is incompatible with v{SCHEMA_VERSION}. "
                "Run `python -m tsr clean --all --yes` and rebuild."
            )

    def _drop_all(self) -> None:
        names = [
            r[0]
            for r in self.conn.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table','trigger','view')"
            )
        ]
        self.conn.execute("PRAGMA foreign_keys=OFF")
        for name in names:
            if name.startswith("sqlite_"):
                continue
            self.conn.execute(f'DROP TABLE IF EXISTS "{name}"')
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.execute("PRAGMA user_version=0")
        self.conn.commit()

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        with self._lock:
            self._drop_all()
            self.conn.executescript(_SCHEMA)
            self.conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
            self.conn.commit()

    def clear(self, scope: str = "images") -> None:
        """Delete images (cascades) or only strategy-derived documents."""
        with self._lock:
            if scope == "images":
                self.conn.execute("DELETE FROM images")
                self.conn.execute("DELETE FROM sqlite_sequence WHERE name='images'")
            self.conn.execute("DELETE FROM strategy_docs")
            self.conn.execute("DELETE FROM nlp_docs")
            self.conn.execute("DELETE FROM vlm_docs")
            self.conn.commit()

    # ------------------------------------------------------------------ #
    def insert_images(self, records: Iterable) -> list[int]:
        """Insert manifest records in order; return primary keys in the same order."""
        ids: list[int] = []
        with self._lock:
            cur = self.conn.cursor()
            for rec in records:
                data = rec.to_dict() if hasattr(rec, "to_dict") else dict(rec)
                cur.execute(
                    """
                    INSERT INTO images
                        (image_id, img_fname, split, path, width, height, title,
                         source_url, post_id, sha1, phash, category)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        data["image_id"],
                        data.get("img_fname"),
                        data.get("split", "trainval"),
                        data["path"],
                        data.get("width"),
                        data.get("height"),
                        data.get("title"),
                        data.get("source_url"),
                        data.get("post_id"),
                        data.get("sha1"),
                        data.get("phash"),
                        data.get("category"),
                    ),
                )
                pk = int(cur.lastrowid)
                ids.append(pk)
                for ordinal, text in enumerate(data.get("img_captions") or []):
                    cur.execute(
                        "INSERT INTO annotations (image_pk, kind, ordinal, text) "
                        "VALUES (?, 'img_caption', ?, ?)",
                        (pk, ordinal, text),
                    )
                for ordinal, text in enumerate(data.get("meme_captions") or []):
                    cur.execute(
                        "INSERT INTO annotations (image_pk, kind, ordinal, text) "
                        "VALUES (?, 'meme_caption', ?, ?)",
                        (pk, ordinal, text),
                    )
                for ordinal, item in enumerate(data.get("metaphors") or []):
                    if not isinstance(item, dict):
                        continue
                    cur.execute(
                        "INSERT INTO metaphors (image_pk, ordinal, metaphor, meaning) "
                        "VALUES (?, ?, ?, ?)",
                        (pk, ordinal, item.get("metaphor", ""), item.get("meaning", "")),
                    )
            self.conn.commit()
        return ids

    def get(self, pk: int) -> ImageRecord | None:
        with self._lock:
            row = self.conn.execute(
                "SELECT * FROM images WHERE id = ?", (pk,)
            ).fetchone()
        return ImageRecord.from_row(row) if row else None

    def get_many(self, pks: Sequence[int]) -> dict[int, ImageRecord]:
        if not pks:
            return {}
        placeholders = ",".join("?" for _ in pks)
        with self._lock:
            rows = self.conn.execute(
                f"SELECT * FROM images WHERE id IN ({placeholders})", tuple(pks)
            ).fetchall()
        return {int(r["id"]): ImageRecord.from_row(r) for r in rows}

    def get_by_image_id(self, image_id: str) -> ImageRecord | None:
        with self._lock:
            row = self.conn.execute(
                "SELECT * FROM images WHERE image_id = ?", (image_id,)
            ).fetchone()
        return ImageRecord.from_row(row) if row else None

    def iter_all(self) -> list[ImageRecord]:
        with self._lock:
            rows = self.conn.execute("SELECT * FROM images ORDER BY id").fetchall()
        return [ImageRecord.from_row(r) for r in rows]

    def count(self) -> int:
        with self._lock:
            return int(self.conn.execute("SELECT COUNT(*) FROM images").fetchone()[0])

    def annotations_for(self, pks: Sequence[int]) -> dict[int, dict[str, list[str]]]:
        out: dict[int, dict[str, list[str]]] = {int(pk): {"img_caption": [], "meme_caption": []}
                                                for pk in pks}
        if not pks:
            return out
        placeholders = ",".join("?" for _ in pks)
        with self._lock:
            rows = self.conn.execute(
                f"SELECT image_pk, kind, text FROM annotations "
                f"WHERE image_pk IN ({placeholders}) ORDER BY image_pk, kind, ordinal",
                tuple(pks),
            ).fetchall()
        for row in rows:
            out.setdefault(int(row["image_pk"]), {}).setdefault(row["kind"], []).append(
                row["text"]
            )
        return out

    def metaphors_for(self, pks: Sequence[int]) -> dict[int, list[dict]]:
        out: dict[int, list[dict]] = {int(pk): [] for pk in pks}
        if not pks:
            return out
        placeholders = ",".join("?" for _ in pks)
        with self._lock:
            rows = self.conn.execute(
                f"SELECT image_pk, metaphor, meaning FROM metaphors "
                f"WHERE image_pk IN ({placeholders}) ORDER BY image_pk, ordinal",
                tuple(pks),
            ).fetchall()
        for row in rows:
            out.setdefault(int(row["image_pk"]), []).append(
                {"metaphor": row["metaphor"], "meaning": row["meaning"]}
            )
        return out

    def enrich(self, records: dict[int, ImageRecord]) -> dict[int, ImageRecord]:
        pks = list(records)
        annotations = self.annotations_for(pks)
        metaphors = self.metaphors_for(pks)
        for pk, rec in records.items():
            rec.img_captions = annotations.get(pk, {}).get("img_caption", [])
            rec.meme_captions = annotations.get(pk, {}).get("meme_caption", [])
            rec.metaphors = metaphors.get(pk, [])
        return records

    # ------------------------------------------------------------------ #
    def write_strategy_docs(self, strategy: str, rows: Iterable[tuple[int, str]]) -> int:
        count = 0
        with self._lock:
            self.conn.execute("DELETE FROM strategy_docs WHERE strategy = ?", (strategy,))
            for pk, text in rows:
                self.conn.execute(
                    "INSERT INTO strategy_docs (strategy, image_pk, text) VALUES (?, ?, ?)",
                    (strategy, pk, text),
                )
                count += 1
            self.conn.commit()
        return count

    def count_strategy_docs(self, strategy: str) -> int:
        with self._lock:
            return int(
                self.conn.execute(
                    "SELECT COUNT(*) FROM strategy_docs WHERE strategy = ?", (strategy,)
                ).fetchone()[0]
            )

    def count_nlp_docs(self) -> int:
        with self._lock:
            return int(self.conn.execute("SELECT COUNT(*) FROM nlp_docs").fetchone()[0])

    def iter_strategy_docs(self, strategy: str) -> list[tuple[int, str]]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT image_pk, text FROM strategy_docs WHERE strategy = ? ORDER BY image_pk",
                (strategy,),
            ).fetchall()
        return [(int(r["image_pk"]), r["text"]) for r in rows]

    def strategy_docs_for(self, strategy: str, pks: Sequence[int]) -> dict[int, str]:
        if not pks:
            return {}
        placeholders = ",".join("?" for _ in pks)
        with self._lock:
            rows = self.conn.execute(
                f"SELECT image_pk, text FROM strategy_docs "
                f"WHERE strategy = ? AND image_pk IN ({placeholders})",
                (strategy, *pks),
            ).fetchall()
        return {int(r["image_pk"]): r["text"] for r in rows}

    def nlp_docs_for(self, pks: Sequence[int]) -> dict[int, dict]:
        if not pks:
            return {}
        placeholders = ",".join("?" for _ in pks)
        with self._lock:
            rows = self.conn.execute(
                f"SELECT * FROM nlp_docs WHERE image_pk IN ({placeholders})", tuple(pks)
            ).fetchall()
        return {int(r["image_pk"]): dict(r) for r in rows}

    def write_nlp_docs(self, rows: Iterable[tuple[int, str, str, str, str]]) -> int:
        count = 0
        with self._lock:
            self.conn.execute("DELETE FROM nlp_docs")
            for pk, scene, meaning, metaphor, title in rows:
                self.conn.execute(
                    "INSERT INTO nlp_docs (image_pk, scene, meaning, metaphor, title) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (pk, scene, meaning, metaphor, title),
                )
                count += 1
            self.conn.commit()
        return count

    def bm25(self, terms: Sequence[str], k: int, weights: Sequence[float]) -> list[tuple[int, float]]:
        clean = [t for t in terms if _TOKEN_RE.fullmatch(t)]
        if not clean or k <= 0:
            return []
        match = " OR ".join(f'"{t}"' for t in clean)
        weight_sql = ", ".join(f"{float(w):.4f}" for w in weights)
        sql = (
            f"SELECT d.image_pk AS image_pk, "
            f"bm25(nlp_docs_fts, {weight_sql}) AS score "
            "FROM nlp_docs_fts JOIN nlp_docs d ON d.id = nlp_docs_fts.rowid "
            "WHERE nlp_docs_fts MATCH ? ORDER BY score LIMIT ?"
        )
        try:
            with self._lock:
                rows = self.conn.execute(sql, (match, int(k))).fetchall()
        except sqlite3.OperationalError as exc:  # malformed MATCH must not crash search
            logger.warning("BM25 query failed (%s); returning no keyword hits", exc)
            return []
        return [(int(r["image_pk"]), -float(r["score"])) for r in rows]

    # ------------------------------------------------------------------ #
    def upsert_vlm_doc(self, doc: dict) -> None:
        fields = (
            "image_pk", "backend", "model", "prompt_version", "scene", "ocr_text",
            "meaning", "emotions", "metaphors", "keywords", "doc_text", "raw",
            "status", "error", "latency_ms",
        )
        values = [doc.get(name) for name in fields]
        with self._lock:
            self.conn.execute(
                f"INSERT INTO vlm_docs ({', '.join(fields)}) "
                f"VALUES ({', '.join('?' for _ in fields)}) "
                "ON CONFLICT(image_pk, backend, model, prompt_version) DO UPDATE SET "
                + ", ".join(f"{name}=excluded.{name}" for name in fields[4:]),
                values,
            )
            self.conn.commit()

    def get_vlm_doc(
        self,
        pk: int,
        *,
        backend: str | None = None,
        model: str | None = None,
        prompt_version: str | None = None,
    ) -> dict | None:
        """Newest successful description for one image, from the given generator.

        Without the filters an image captioned by an older model would show that
        stale description even though the current index no longer contains it.
        """
        sql = "SELECT * FROM vlm_docs WHERE image_pk = ? AND status = 'ok'"
        params: list = [int(pk)]
        for column, value in (
            ("backend", backend), ("model", model),
            ("prompt_version", prompt_version),
        ):
            if value:
                sql += f" AND {column} = ?"
                params.append(str(value))
        sql += " ORDER BY id DESC LIMIT 1"
        with self._lock:
            row = self.conn.execute(sql, params).fetchone()
        return dict(row) if row else None

    def iter_vlm_docs(
        self,
        *,
        backend: str | None = None,
        model: str | None = None,
        prompt_version: str | None = None,
    ) -> list[dict]:
        """Successful descriptions, optionally restricted to one generator.

        The build path filters by the configured backend/model/prompt version so
        stale descriptions from an earlier model can never be indexed by mistake.
        """
        sql = "SELECT * FROM vlm_docs WHERE status = 'ok'"
        params: list = []
        for column, value in (
            ("backend", backend), ("model", model), ("prompt_version", prompt_version)
        ):
            if value:
                sql += f" AND {column} = ?"
                params.append(str(value))
        sql += " ORDER BY image_pk"
        with self._lock:
            rows = self.conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def vlm_count(
        self,
        *,
        backend: str | None = None,
        model: str | None = None,
        prompt_version: str | None = None,
    ) -> int:
        sql = "SELECT COUNT(*) FROM vlm_docs WHERE status = 'ok'"
        params: list = []
        for column, value in (
            ("backend", backend), ("model", model), ("prompt_version", prompt_version)
        ):
            if value:
                sql += f" AND {column} = ?"
                params.append(str(value))
        with self._lock:
            return int(self.conn.execute(sql, params).fetchone()[0])

    # ------------------------------------------------------------------ #
    def set_meta(self, key: str, value) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT INTO meta (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, json.dumps(value)),
            )
            self.conn.commit()

    def get_meta(self, key: str, default=None):
        with self._lock:
            row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        if row is None:
            return default
        try:
            return json.loads(row["value"])
        except json.JSONDecodeError:
            return row["value"]

    def close(self) -> None:
        with self._lock:
            self.conn.close()
