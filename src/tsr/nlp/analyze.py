"""Tokenisation and lexicon-based query expansion for strategy B.

Expansion is deliberately small and hand-authored. It only affects BM25, never
the CLIP or dense text encoders, so it cannot silently change the semantic
query that the other signals see.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "for",
    "from", "had", "has", "have", "he", "her", "his", "i", "in", "is", "it",
    "its", "me", "my", "of", "on", "or", "our", "she", "that", "the", "their",
    "them", "there", "they", "this", "to", "was", "were", "what", "when",
    "where", "which", "who", "will", "with", "you", "your", "image", "meme",
    "picture", "photo", "find", "show", "looking", "want",
}


@dataclass
class QueryTerms:
    seed: list[str] = field(default_factory=list)
    expansion: list[str] = field(default_factory=list)
    phrases: list[str] = field(default_factory=list)

    def all_terms(self) -> list[str]:
        return self.seed + [term for term in self.expansion if term not in self.seed]


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def terms_for_fts(text: str) -> list[str]:
    """Content tokens only (stopwords removed), safe for FTS quoting."""
    return [token for token in tokenize(text) if token not in STOPWORDS and len(token) > 1]


def expand_query(
    query: str,
    lexicon: Mapping[str, list[str]] | None,
    expansion_weight: float = 0.5,
) -> QueryTerms:
    """Return seed terms plus lexicon-derived expansion terms.

    ``expansion_weight`` is carried for callers that rank expansion matches
    lower; the terms themselves are returned unweighted.
    """
    seed = terms_for_fts(query)
    lowered = (query or "").lower()
    expansion: list[str] = []
    phrases: list[str] = []
    for phrase, related in (lexicon or {}).items():
        if phrase.lower() in lowered:
            phrases.append(phrase)
            for term in related:
                token = str(term).lower().strip()
                if token and token not in seed and token not in expansion:
                    expansion.append(token)
    return QueryTerms(seed=seed, expansion=expansion, phrases=phrases)


def build_fts_query(terms: list[str]) -> str:
    """Quote every term and OR them; unknown characters are already stripped."""
    clean = [term for term in terms if _TOKEN_RE.fullmatch(term)]
    return " OR ".join(f'"{term}"' for term in clean)
