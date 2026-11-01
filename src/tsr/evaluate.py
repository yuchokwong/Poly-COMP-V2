"""Evaluation metrics, leakage checks, bootstrap significance and eval-set I/O."""

from __future__ import annotations

import json
import math
import random
import re
import statistics
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from tsr.logging_utils import get_logger

logger = get_logger(__name__)

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


# --------------------------------------------------------------------------- #
# Queries and qrels
# --------------------------------------------------------------------------- #
@dataclass
class EvalQuery:
    qid: str
    query: str
    relevant: list[str]
    category: str = "visual"
    grades: dict[str, int] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> EvalQuery:
        relevant = data.get("relevant") or data.get("gt") or []
        if isinstance(relevant, str):
            relevant = [relevant]
        grades = data.get("grades") or {}
        return cls(
            qid=str(data.get("qid") or data.get("query", "")[:16]),
            query=str(data["query"]),
            relevant=[str(item) for item in relevant],
            category=str(data.get("category") or "visual"),
            grades={str(k): int(v) for k, v in grades.items()},
            metadata=dict(data.get("metadata") or {}),
        )

    def to_dict(self) -> dict:
        return {
            "qid": self.qid,
            "query": self.query,
            "relevant": self.relevant,
            "category": self.category,
            "grades": self.grades,
            "metadata": self.metadata,
        }


def save_queries(queries: Sequence[EvalQuery], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([q.to_dict() for q in queries], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def load_queries(path: str | Path) -> list[EvalQuery]:
    path = Path(path)
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("queries") or []
    return [EvalQuery.from_dict(item) for item in data]


def save_qrels(queries: Sequence[EvalQuery], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for query in queries:
            for image_id, grade in query.grades.items():
                fh.write(
                    json.dumps(
                        {"qid": query.qid, "image_id": image_id, "grade": grade},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    return path


# --------------------------------------------------------------------------- #
# Metric primitives
# --------------------------------------------------------------------------- #
def precision_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    return sum(1 for doc in retrieved[:k] if doc in relevant) / k


def recall_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return sum(1 for doc in retrieved[:k] if doc in relevant) / len(relevant)


def average_precision(retrieved: Sequence[str], relevant: set[str]) -> float:
    if not relevant or not retrieved:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for i, doc in enumerate(retrieved, start=1):
        if doc in relevant:
            hits += 1
            precision_sum += hits / i
    return precision_sum / min(len(relevant), len(retrieved))


def reciprocal_rank(retrieved: Sequence[str], relevant: set[str]) -> float:
    for i, doc in enumerate(retrieved, start=1):
        if doc in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(
    retrieved: Sequence[str],
    relevant: set[str],
    k: int,
    gains: dict[str, int] | None = None,
) -> float:
    """nDCG@k with graded gains (default binary gain 1)."""
    if k <= 0 or not relevant:
        return 0.0
    gains = gains or {}
    dcg = 0.0
    for i, doc in enumerate(retrieved[:k], start=1):
        gain = float(gains.get(doc, 1 if doc in relevant else 0))
        if gain > 0:
            dcg += (2**gain - 1) / math.log2(i + 1)
    ideal_gains = sorted((float(gains.get(doc, 1)) for doc in relevant), reverse=True)[:k]
    ideal = sum((2**gain - 1) / math.log2(i + 1) for i, gain in enumerate(ideal_gains, start=1))
    return dcg / ideal if ideal > 0 else 0.0


def recall_at_r(retrieved: Sequence[str], relevant: set[str]) -> float:
    if not relevant:
        return 0.0
    r = len(relevant)
    return sum(1 for doc in retrieved[:r] if doc in relevant) / r


def first_relevant_rank(retrieved: Sequence[str], relevant: set[str]) -> int | None:
    for i, doc in enumerate(retrieved, start=1):
        if doc in relevant:
            return i
    return None


# --------------------------------------------------------------------------- #
# Full evaluation
# --------------------------------------------------------------------------- #
def retrieval_depth(queries: Sequence[EvalQuery], top_k: Sequence[int]) -> int:
    """How deep to retrieve so recall@R and mAP are not truncated by @k.

    Retrieving only ``max(top_k)`` silently caps recall@R at ``max_k / R`` when
    a query has more relevant images than that — which graded metaphor queries do.
    """
    largest_to_k = max(top_k, default=10)
    largest_relevant = max((len(q.relevant) for q in queries), default=0)
    return max(largest_to_k, largest_relevant, 100)


def evaluate_queries(
    retriever: Callable[[str, int], list[str]],
    queries: Sequence[EvalQuery],
    top_k: Iterable[int] = (1, 5, 10),
    max_rank: int | None = None,
) -> dict:
    top_k = sorted({int(k) for k in top_k})
    limit = max_rank or retrieval_depth(queries, top_k)
    sums: dict[str, float] = {}
    for k in top_k:
        sums[f"precision@{k}"] = 0.0
        sums[f"recall@{k}"] = 0.0
        sums[f"ndcg@{k}"] = 0.0
    ap_sum = rr_sum = recall_r_sum = 0.0
    ranks: list[int] = []
    misses = 0
    per_query: list[dict] = []
    groups: dict[str, list[dict]] = {}

    for query in queries:
        relevant = set(query.relevant)
        retrieved = retriever(query.query, limit)
        grades = query.grades or {image_id: 1 for image_id in relevant}
        for k in top_k:
            sums[f"precision@{k}"] += precision_at_k(retrieved, relevant, k)
            sums[f"recall@{k}"] += recall_at_k(retrieved, relevant, k)
            sums[f"ndcg@{k}"] += ndcg_at_k(retrieved, relevant, k, grades)
        ap = average_precision(retrieved, relevant)
        rr = reciprocal_rank(retrieved, relevant)
        r_at_r = recall_at_r(retrieved, relevant)
        ap_sum += ap
        rr_sum += rr
        recall_r_sum += r_at_r
        rank = first_relevant_rank(retrieved, relevant)
        if rank is None:
            misses += 1
        else:
            ranks.append(rank)
        row = {
            "qid": query.qid,
            "query": query.query,
            "category": query.category,
            "rank": rank,
            "ap": round(ap, 4),
            "rr": round(rr, 4),
            "ndcg@10": round(ndcg_at_k(retrieved, relevant, 10, grades), 4),
            "retrieved": list(retrieved),
        }
        per_query.append(row)
        for bucket in (query.category, str(query.metadata.get("split", "all"))):
            groups.setdefault(bucket, []).append(row)

    n = max(1, len(queries))
    metrics = {name: value / n for name, value in sums.items()}
    metrics["mAP"] = ap_sum / n
    metrics["MRR"] = rr_sum / n
    metrics["recall@R"] = recall_r_sum / n
    metrics["median_rank"] = statistics.median(ranks) if ranks else None
    metrics["misses"] = misses
    metrics["num_queries"] = len(queries)

    per_group: dict[str, dict] = {}
    for name, rows in groups.items():
        count = max(1, len(rows))
        per_group[name] = {
            "num_queries": len(rows),
            "mAP": round(sum(row["ap"] for row in rows) / count, 4),
            "MRR": round(sum(row["rr"] for row in rows) / count, 4),
        }
    return {
        "metrics": metrics,
        "per_query": per_query,
        "per_category": {k: v for k, v in per_group.items() if k in CATEGORIES},
        "per_split": {k: v for k, v in per_group.items() if k not in CATEGORIES},
    }


CATEGORIES = {"visual", "intent", "metaphor", "independent", "mixed"}


def per_query_values(
    retriever: Callable[[str, int], list[str]],
    queries: Sequence[EvalQuery],
    top_k: Iterable[int] = (1, 5, 10),
) -> dict[str, list[float]]:
    top_k = sorted({int(k) for k in top_k})
    limit = retrieval_depth(queries, top_k)
    values: dict[str, list[float]] = {"mAP": [], "MRR": [], "recall@R": []}
    for k in top_k:
        values[f"precision@{k}"] = []
        values[f"recall@{k}"] = []
        values[f"ndcg@{k}"] = []
    for query in queries:
        relevant = set(query.relevant)
        grades = query.grades or {image_id: 1 for image_id in relevant}
        retrieved = retriever(query.query, limit)
        values["mAP"].append(average_precision(retrieved, relevant))
        values["MRR"].append(reciprocal_rank(retrieved, relevant))
        values["recall@R"].append(recall_at_r(retrieved, relevant))
        for k in top_k:
            values[f"precision@{k}"].append(precision_at_k(retrieved, relevant, k))
            values[f"recall@{k}"].append(recall_at_k(retrieved, relevant, k))
            values[f"ndcg@{k}"].append(ndcg_at_k(retrieved, relevant, k, grades))
    return values


# --------------------------------------------------------------------------- #
# Bootstrap and significance
# --------------------------------------------------------------------------- #
def bootstrap_ci(
    values: dict[str, list[float]],
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, tuple[float, float, float]]:
    rng = np.random.default_rng(seed)
    out: dict[str, tuple[float, float, float]] = {}
    for name, values_list in values.items():
        data = np.asarray(values_list, dtype=np.float64)
        if data.size == 0:
            out[name] = (0.0, 0.0, 0.0)
            continue
        n = data.size
        boot = data[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
        lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        out[name] = (float(data.mean()), float(lo), float(hi))
    return out


def paired_bootstrap_test(
    values_a: Sequence[float],
    values_b: Sequence[float],
    n_boot: int = 1000,
    seed: int = 42,
) -> tuple[float, float, float, float]:
    """Paired bootstrap for B-A: (delta, lo, hi, two-sided p)."""
    a = np.asarray(values_a, dtype=np.float64)
    b = np.asarray(values_b, dtype=np.float64)
    if a.size == 0 or a.size != b.size:
        return (0.0, 0.0, 0.0, 1.0)
    diff = b - a
    rng = np.random.default_rng(seed)
    boot = diff[rng.integers(0, diff.size, size=(n_boot, diff.size))].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p = 2.0 * min(float((boot <= 0).mean()), float((boot >= 0).mean()))
    # A finite bootstrap cannot resolve below 1/n_boot; reporting p=0 would
    # overclaim certainty, so floor it at the resampling resolution.
    p = min(1.0, max(p, 1.0 / max(1, n_boot)))
    return (float(diff.mean()), float(lo), float(hi), p)


def win_tie_loss(values_a: Sequence[float], values_b: Sequence[float], tol: float = 1e-9) -> dict:
    wins = ties = losses = 0
    for a, b in zip(values_a, values_b, strict=False):
        if abs(a - b) <= tol:
            ties += 1
        elif b > a:
            wins += 1
        else:
            losses += 1
    return {"win": wins, "tie": ties, "loss": losses}


def holm_bonferroni(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjustment; returns adjusted p-values by key."""
    items = sorted(pvalues.items(), key=lambda item: item[1])
    m = len(items)
    adjusted: dict[str, float] = {}
    running = 0.0
    for index, (key, value) in enumerate(items):
        running = max(running, min(1.0, (m - index) * value))
        adjusted[key] = running
    return adjusted


# --------------------------------------------------------------------------- #
# Leakage checks
# --------------------------------------------------------------------------- #
def _tokens(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(text or "")]


def _ngrams(tokens: Sequence[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(tokens[i : i + n]) for i in range(0, len(tokens) - n + 1)}


def _jaccard(left: str, right: str) -> float:
    a, b = set(_tokens(left)), set(_tokens(right))
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _short_doc_tokens(docs: Sequence[str], n: int) -> set[tuple[str, ...]]:
    """Token tuples for annotations too short to produce an ``n``-gram."""
    return {tuple(tokens) for tokens in (_tokens(doc) for doc in docs) if 0 < len(tokens) < n}


def _has_contained_span(
    q_tokens: Sequence[str], docs: set[tuple[str, ...]], min_tokens: int = 4
) -> bool:
    """True when a whole short annotation appears verbatim inside the query.

    The n-gram check cannot fire on annotations shorter than ``n`` tokens, so a
    pasted short caption needs this length-independent check. Runs shorter than
    ``min_tokens`` are ignored: ordinary descriptions legitimately share content
    words with the source caption, and only a copied phrase is a leak. This only
    sees complete annotations; copied spans embedded inside a longer annotation
    are caught by ``_longest_shared_span``.
    """
    if not docs:
        return False
    by_length: dict[int, set[tuple[str, ...]]] = {}
    for doc in docs:
        if min_tokens <= len(doc) < len(q_tokens) + 1:
            by_length.setdefault(len(doc), set()).add(doc)
    for length, group in by_length.items():
        for start in range(0, len(q_tokens) - length + 1):
            if tuple(q_tokens[start : start + length]) in group:
                return True
    return False


def _longest_shared_span(
    q_tokens: Sequence[str], docs_tokens: Sequence[Sequence[str]], max_len: int
) -> int:
    """Longest contiguous token run the query shares with any given document.

    ``max_len`` caps the search at ``n - 1`` because a run of ``n`` or more is
    already counted by the n-gram check.
    """
    limit = min(max_len, len(q_tokens))
    for length in range(limit, 1, -1):
        q_grams = _ngrams(q_tokens, length)
        if any(q_grams & _ngrams(list(doc), length) for doc in docs_tokens):
            return length
    return 0


def check_leakage(
    query: str,
    source_docs: Sequence[str],
    all_docs: Sequence[str] | None = None,
    n: int = 8,
    jaccard_threshold: float = 0.6,
    *,
    all_grams: set[tuple[str, ...]] | None = None,
    short_docs: set[tuple[str, ...]] | None = None,
    min_contained_tokens: int = 4,
    min_copy_tokens: int = 5,
) -> dict:
    """Flag a query that copies its source annotation.

    Returns n-gram overlap and maximum Jaccard similarity against the source
    image's annotation fields, plus the same overlap over the whole gallery
    (strict check, guards against qrels text leaking into other images).

    A query is rejected when it contains an ``n``-gram of any annotation, a
    whole short annotation (4-7 tokens), a contiguous run of at least
    ``min_copy_tokens`` tokens from the source annotation, or when its token
    Jaccard with a source caption is at least ``jaccard_threshold``.

    ``all_grams``/``short_docs`` let callers precompute the gallery view once
    instead of re-tokenising every annotation per query.
    """
    q_tokens = _tokens(query)
    q_grams = _ngrams(q_tokens, n)
    source_grams: set[tuple[str, ...]] = set()
    source_short = set()
    source_tokens: list[list[str]] = []
    max_jaccard = 0.0
    for doc in source_docs:
        d_tokens = _tokens(doc)
        source_tokens.append(d_tokens)
        source_grams |= _ngrams(d_tokens, n)
        if 0 < len(d_tokens) < n:
            source_short.add(tuple(d_tokens))
        if q_tokens and d_tokens:
            union = len(set(q_tokens) | set(d_tokens))
            max_jaccard = max(max_jaccard, len(set(q_tokens) & set(d_tokens)) / union)
    overlap = len(q_grams & source_grams)
    strict_overlap = overlap
    gallery_short = short_docs
    if all_docs is not None:
        if all_grams is None:
            all_grams = set()
            for doc in all_docs:
                all_grams |= _ngrams(_tokens(doc), n)
        if gallery_short is None:
            gallery_short = _short_doc_tokens(all_docs, n)
        strict_overlap = len(q_grams & all_grams)
    elif all_grams is not None:
        strict_overlap = len(q_grams & all_grams)
    contained = _has_contained_span(q_tokens, source_short, min_contained_tokens)
    gallery_contained = _has_contained_span(
        q_tokens, gallery_short or set(), min_contained_tokens
    )
    # The n-gram rule misses a copied phrase shorter than n tokens, and the
    # containment rule only sees whole short annotations. This compares the
    # longest contiguous run against the source annotations directly, which is
    # what makes "the query was written, not pasted" checkable at 5+ tokens.
    source_span = _longest_shared_span(q_tokens, source_tokens, n - 1)
    passed = (
        overlap == 0
        and strict_overlap == 0
        and not contained
        and not gallery_contained
        and source_span < min_copy_tokens
        and max_jaccard < jaccard_threshold
    )
    return {
        "max_ngram_overlap": overlap,
        "strict_overlap": strict_overlap,
        "contained": contained,
        "gallery_contained": gallery_contained,
        "max_source_span": source_span,
        "min_copy_tokens": min_copy_tokens,
        "jaccard": round(max_jaccard, 4),
        "num_tokens": len(q_tokens),
        "passed": passed,
        "checked_n": n,
    }


# --------------------------------------------------------------------------- #
# Synthetic eval-set drafting
# --------------------------------------------------------------------------- #
_SYNONYMS = {
    "person": "figure", "people": "folks", "man": "guy", "woman": "lady",
    "boy": "kid", "girl": "youngster", "holding": "carrying", "wearing": "dressed in",
    "standing": "posed", "sitting": "seated", "looking": "gazing", "smiling": "grinning",
    "talking": "speaking", "shows": "depicts", "showing": "depicting",
    "picture": "image", "photo": "shot", "meme poster": "caption writer",
    "feels": "is feeling", "because": "since", "about": "regarding",
    "trying": "attempting", "wants": "wishes", "going": "heading", "says": "states",
    "said": "stated", "thinks": "believes", "knows": "realises", "makes": "creates",
    "uses": "employs", "getting": "receiving", "gives": "hands", "takes": "grabs",
}

# Function words that carry no visual meaning are swapped so an annotation cannot
# be copied verbatim into a query. Pronouns and possessives are left alone because
# changing them would alter who the meme is about.
_STOPWORD_SWAPS = {
    "the": "this", "a": "one", "an": "one", "of": "from", "is": "seems",
    "are": "seem", "to": "toward", "in": "inside", "and": "plus",
    "with": "alongside", "for": "regarding", "on": "upon", "at": "near",
}


def _paraphrase(text: str) -> str:
    lowered = text.strip().lower()
    for source, target in _SYNONYMS.items():
        lowered = re.sub(rf"\b{re.escape(source)}\b", target, lowered)
    lowered = re.sub(r"[^a-z0-9\s']", " ", lowered)
    words = re.findall(r"[a-z0-9']+", lowered)
    lowered = " ".join(_STOPWORD_SWAPS.get(word, word) for word in words)
    return lowered.strip() or text.strip().lower()


def _source_docs(rec) -> list[str]:
    """Annotation text that a strategy could have indexed for this image."""
    docs = list(rec.img_captions) + list(rec.meme_captions)
    docs += [
        f"{item.get('metaphor', '')} means {item.get('meaning', '')}"
        for item in _meaningful(rec)
    ]
    title = getattr(rec, "title", None)
    if title:
        docs.append(title)
    return [doc for doc in docs if str(doc).strip()]


NOT_RELATED_MEANING = "not related to the meme context"

# Meme-Cap marks ordinary subjects with these meanings; they are literal
# depictions, not symbolism, so they must not create "shared metaphor" qrels.
# The annotators were inconsistent about the article and about capitalisation
# ("Meme poster", "poster", "the poster", "the meme poster"), so match the
# whole normalized phrase instead of an exact string.
_LITERAL_MEANING_RE = re.compile(
    rf"^(?:{re.escape(NOT_RELATED_MEANING)}|(?:the )?(?:meme )?poster)$"
)


def _is_literal_meaning(meaning: str) -> bool:
    normalized = " ".join(str(meaning or "").strip().lower().split())
    return bool(_LITERAL_MEANING_RE.match(normalized))


def _meaningful(rec) -> list[dict]:
    """Metaphor rows that a strategy indexes (drops the literal marker only).

    Kept broader than ``_nonliteral_metaphors`` because the leakage check must
    cover the "Meme poster" rows that Strategy B indexes as text.
    """
    return [
        item for item in rec.metaphors
        if str(item.get("meaning", "")).strip().lower() != NOT_RELATED_MEANING
    ]


def _nonliteral_metaphors(rec) -> list[dict]:
    """Metaphor rows that name a symbol and an intended meaning.

    A row whose meaning is "Meme poster" describes the annotation, not a
    metaphor, so it can neither seed a metaphor query nor earn graded
    relevance.
    """
    out = []
    for item in rec.metaphors:
        label = str(item.get("metaphor", "")).strip()
        meaning = str(item.get("meaning", "")).strip()
        if not label or not meaning or _is_literal_meaning(meaning):
            continue
        out.append({"metaphor": label, "meaning": meaning})
    return out


def _metaphor_pairs(rec) -> dict[str, set[frozenset[str]]]:
    """label -> the set of token sets for its non-literal meanings."""
    pairs: dict[str, set[frozenset[str]]] = {}
    for item in _nonliteral_metaphors(rec):
        label = item["metaphor"].lower()
        pairs.setdefault(label, set()).add(frozenset(_tokens(item["meaning"])))
    return pairs


def _meaning_similar(left: set[frozenset[str]], right: set[frozenset[str]]) -> bool:
    """True when the same label carries the same intended meaning in both images."""
    for a in left:
        for b in right:
            if a == b:
                return True
            union = len(a | b)
            if union and len(a & b) / union >= 0.6:
                return True
    return False


def gallery_leakage_index(
    records: Sequence, ngram: int = 8
) -> tuple[set[tuple[str, ...]], set[tuple[str, ...]]]:
    """Precompute gallery n-grams and short documents once for all queries."""
    all_grams: set[tuple[str, ...]] = set()
    short: set[tuple[str, ...]] = set()
    for record in records:
        for doc in _source_docs(record):
            tokens = _tokens(doc)
            all_grams |= _ngrams(tokens, ngram)
            if 0 < len(tokens) < ngram:
                short.add(tuple(tokens))
    return all_grams, short


def draft_eval_set(
    records: Sequence,
    num_queries: int = 240,
    seed: int = 42,
    per_category: dict[str, int] | None = None,
    ngram: int = 8,
    metaphor_max_df: int = 10,
    min_copy_tokens: int = 5,
) -> list[EvalQuery]:
    """Draft a stratified evaluation set from gallery records.

    This is a starting point for human review, not a finished benchmark. Every
    query is checked for annotation leakage and flagged in ``metadata``.
    """
    rng = random.Random(seed)
    per_category = per_category or {
        "visual": num_queries // 3,
        "intent": num_queries // 3,
        "metaphor": num_queries - 2 * (num_queries // 3),
    }
    test = [rec for rec in records if rec.split == "test"]
    pool = test if len(test) >= num_queries else list(records)
    metaphor_df = metaphor_document_frequency(records)
    all_grams, short_docs = gallery_leakage_index(records, ngram)
    queries: list[EvalQuery] = []
    used: set[int] = set()
    rejected = 0
    for category, wanted in per_category.items():
        # Each category draws from its own shuffled pool: a category with many
        # rejected drafts must not consume the candidates needed by the next one.
        candidates = [_eligible(rec, category) for rec in pool]
        candidates = [rec for rec in candidates if rec is not None]
        rng.shuffle(candidates)
        made = 0
        for rec in candidates:
            if made >= wanted:
                break
            if rec.id in used:
                continue
            query_text = _draft_query(rec, category, rng)
            query = build_eval_query(
                rec,
                query_text,
                category,
                records,
                ngram=ngram,
                origin="synthetic_paraphrase",
                metaphor_df=metaphor_df,
                metaphor_max_df=metaphor_max_df,
                all_grams=all_grams,
                short_docs=short_docs,
                min_copy_tokens=min_copy_tokens,
            )
            if query is None:
                rejected += 1
                logger.debug("Rejected leaked draft for %s", rec.image_id)
                continue
            query.qid = f"q{len(queries) + 1:04d}"
            used.add(rec.id)
            queries.append(query)
            made += 1
        if made < wanted:
            logger.warning(
                "Category %s: only %d/%d drafts survived leakage checks",
                category,
                made,
                wanted,
            )
    logger.info(
        "Drafted %d queries (%s); rejected %d leaked drafts",
        len(queries),
        per_category,
        rejected,
    )
    return queries


def metaphor_document_frequency(records: Sequence) -> dict[str, int]:
    """How many images carry each meaningful metaphor label.

    Corpus-common labels ("a man", "looking") are literal depictions rather than
    shared symbolism; an uncapped rule would mark hundreds of images relevant and
    make metaphor mAP meaningless.
    """
    df: dict[str, int] = {}
    for record in records:
        for label in _metaphor_pairs(record):
            if label:
                df[label] = df.get(label, 0) + 1
    return df


def build_eval_query(
    rec,
    query_text: str,
    category: str,
    records: Sequence,
    *,
    ngram: int = 8,
    origin: str = "llm_rewrite",
    metaphor_df: dict[str, int] | None = None,
    metaphor_max_df: int = 10,
    all_grams: set[tuple[str, ...]] | None = None,
    short_docs: set[tuple[str, ...]] | None = None,
    min_copy_tokens: int = 5,
) -> EvalQuery | None:
    """Attach rule-derived qrels to a reviewed query, or reject it as leaked.

    Grade 2 is the source image only. Grade 1 comes from the three deterministic
    rules (near-identical scene, identical intent wording, shared specific
    metaphor) and is never assigned by a model judgement.
    """
    if all_grams is None or short_docs is None:
        all_grams, short_docs = gallery_leakage_index(records, ngram)
    leakage = check_leakage(
        query_text,
        _source_docs(rec),
        n=ngram,
        all_grams=all_grams,
        short_docs=short_docs,
        min_copy_tokens=min_copy_tokens,
    )
    if not leakage["passed"]:
        return None
    if category == "metaphor" and metaphor_df is None:
        metaphor_df = metaphor_document_frequency(records)
    source_pairs = _metaphor_pairs(rec) if category == "metaphor" else {}
    grades = {rec.image_id: 2}
    for other in records:
        if other.image_id == rec.image_id:
            continue
        if category == "visual":
            if (
                other.img_captions
                and rec.img_captions
                and _jaccard(rec.img_captions[0], other.img_captions[0]) >= 0.8
            ):
                grades[other.image_id] = 1
        elif category == "metaphor":
            assert metaphor_df is not None
            other_pairs = _metaphor_pairs(other)
            shared = set(source_pairs) & set(other_pairs)
            # A shared label is only a shared metaphor when it also carries the
            # same intended meaning; otherwise "having" or "a masked man" would
            # mark unrelated images relevant.
            if any(
                metaphor_df.get(label, 0) <= metaphor_max_df
                and _meaning_similar(source_pairs[label], other_pairs[label])
                for label in shared
            ):
                grades[other.image_id] = 1
        elif category == "intent":
            if (
                other.meme_captions
                and rec.meme_captions
                and other.meme_captions[0].strip().lower()
                == rec.meme_captions[0].strip().lower()
            ):
                grades[other.image_id] = 1
    return EvalQuery(
        qid="",
        query=query_text,
        relevant=list(grades),
        category=category,
        grades=grades,
        metadata={
            "split": rec.split,
            "image_id": rec.image_id,
            "query_origin": origin,
            "leakage": leakage,
        },
    )


def _snapshot_metaphors(items) -> list[tuple[str, str]]:
    """The reviewed snapshot drops only the literal 'not related' marker rows.

    The gallery keeps those rows (Strategy B indexes their labels), but they
    never seed a query, so the draft written for review omitted them.
    """
    pairs = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        metaphor = str(item.get("metaphor", "")).strip()
        meaning = str(item.get("meaning", "")).strip()
        if not metaphor or not meaning:
            continue
        if meaning.lower() == NOT_RELATED_MEANING:
            continue
        pairs.append((metaphor, meaning))
    return pairs


def _snapshot_matches(source: dict, rec) -> bool:
    """True when the reviewed source still describes the served annotations.

    Drafts store a prefix of each annotation list (captions and metaphor rows
    were trimmed for review), so the draft must remain a prefix of the current
    values: appending a caption keeps the reviewed text intact, while editing or
    removing one makes the query describe something the gallery no longer says.
    Only fields present in the draft are compared, so older review files keep
    working.
    """

    def prefix_ok(draft, current) -> bool:
        draft = [str(x).strip() for x in draft or []]
        current = [str(x).strip() for x in current or []]
        return current[: len(draft)] == draft

    if "img_captions" in source and not prefix_ok(
        source.get("img_captions"), rec.img_captions
    ):
        return False
    if "meme_captions" in source and not prefix_ok(
        source.get("meme_captions"), rec.meme_captions
    ):
        return False
    if "metaphors" in source:
        draft_rows = _snapshot_metaphors(source.get("metaphors"))
        current_rows = _snapshot_metaphors(rec.metaphors)
        if current_rows[: len(draft_rows)] != draft_rows:
            return False
    if "title" in source:
        draft_title = str(source.get("title") or "").strip() or None
        if draft_title != (str(rec.title or "").strip() or None):
            return False
    return True


def reviewed_eval_set(
    sources: Sequence[dict],
    rewrites: dict[str, str],
    records: Sequence,
    *,
    ngram: int = 8,
    origin: str = "llm_rewrite",
    metaphor_max_df: int = 10,
    min_copy_tokens: int = 5,
) -> tuple[list[EvalQuery], list[dict]]:
    """Assemble the reviewed benchmark from draft sources plus written queries.

    ``sources`` is the deterministic draft produced for review (``qid``,
    ``image_id``, ``category``); ``rewrites`` maps those qids to the reviewed
    query text. Every query still goes through the leakage gate, so a reviewer
    who pastes an annotation back in cannot silently leak it into the benchmark.

    The draft's own annotation snapshot is compared with the served record, so a
    source whose annotations changed after review is rejected instead of being
    silently re-pointed at different text. Only test-split sources are accepted:
    the benchmark is documented as test-only, and a trainval source would leak
    the gallery's own annotations into the query.
    """
    by_image = {str(rec.image_id): rec for rec in records}
    metaphor_df = metaphor_document_frequency(records)
    all_grams, short_docs = gallery_leakage_index(records, ngram)
    queries: list[EvalQuery] = []
    rejected: list[dict] = []
    for source in sources:
        qid = str(source.get("qid") or "")
        text = str(rewrites.get(qid, "") or "").strip()
        if not text:
            rejected.append({"qid": qid, "reason": "missing_rewrite"})
            continue
        rec = by_image.get(str(source.get("image_id") or ""))
        if rec is None:
            rejected.append({"qid": qid, "reason": "unknown_image"})
            continue
        if str(rec.split) != "test":
            rejected.append(
                {"qid": qid, "reason": "not_test_split", "split": str(rec.split)}
            )
            continue
        if not _snapshot_matches(source, rec):
            rejected.append({"qid": qid, "reason": "stale_source"})
            continue
        query = build_eval_query(
            rec,
            text,
            str(source.get("category") or "visual"),
            records,
            ngram=ngram,
            origin=origin,
            metaphor_df=metaphor_df,
            metaphor_max_df=metaphor_max_df,
            all_grams=all_grams,
            short_docs=short_docs,
            min_copy_tokens=min_copy_tokens,
        )
        if query is None:
            rejected.append({"qid": qid, "reason": "leakage", "query": text})
            continue
        query.qid = qid
        queries.append(query)
    return queries, rejected


def scale_per_category(per_category: dict[str, int], num_queries: int) -> dict[str, int]:
    """Scale category counts to a requested total, keeping their proportions."""
    total = sum(int(v) for v in per_category.values())
    if total <= 0 or num_queries <= 0:
        return dict(per_category)
    scaled = {key: num_queries * int(value) / total for key, value in per_category.items()}
    counts = {key: int(value) for key, value in scaled.items()}
    remainder = num_queries - sum(counts.values())
    for key in sorted(scaled, key=lambda k: scaled[k] - counts[k], reverse=True)[:remainder]:
        counts[key] += 1
    return counts


def _eligible(rec, category: str):
    if category == "visual":
        return rec if rec.img_captions else None
    if category == "intent":
        return rec if rec.meme_captions else None
    if category in {"metaphor", "mixed"}:
        return rec if _nonliteral_metaphors(rec) else None
    return rec


def _draft_query(rec, category: str, rng: random.Random) -> str:
    if category == "visual":
        return "i am looking for an image that depicts " + _paraphrase(rec.img_captions[0])
    if category == "intent":
        return "find a meme whose message is that " + _paraphrase(rec.meme_captions[0])
    item = rng.choice(_nonliteral_metaphors(rec))
    return (
        f"a meme using {_paraphrase(item['metaphor'])} as a symbol for "
        f"{_paraphrase(item['meaning'])}"
    )
