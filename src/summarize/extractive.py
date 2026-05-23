"""Extractive reduction for long Multi-LexSum source texts.

The public entry point is `extractive_summarize`, which exposes LexRank and
TextRank through one shared interface. The implementation uses sentence-level
TF-IDF cosine graphs plus PageRank. For very long cases, it first keeps a
position-balanced candidate pool so PageRank stays tractable on CPU.

CLI examples
------------
    python -m src.summarize.extractive \
        --input data/multilexsum_clean.parquet \
        --output results/extractive_lengths.csv \
        --method lexrank

    python -m src.summarize.extractive --text "Long case text..."
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from tqdm import tqdm

from src.utils import RESULTS_DIR, get_logger

logger = get_logger(__name__)

ExtractiveMethod = Literal["lexrank", "textrank"]

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]*|\d+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'])")
_ABBREVIATIONS = {
    "mr.",
    "mrs.",
    "ms.",
    "dr.",
    "prof.",
    "inc.",
    "corp.",
    "co.",
    "ltd.",
    "u.s.",
    "u.s.c.",
    "e.g.",
    "i.e.",
    "no.",
    "nos.",
    "v.",
}


@dataclass(frozen=True)
class ExtractiveSummary:
    """Structured output from the extractive reducer."""

    method: str
    summary_text: str
    original_tokens: int
    reduced_tokens: int
    original_sentences: int
    selected_sentences: int
    candidate_sentences: int
    reduction_pct: float

    def to_length_row(self, case_id: str | None = None, split: str | None = None) -> dict:
        row = asdict(self)
        row.pop("summary_text", None)
        if case_id is not None:
            row["case_id"] = case_id
        if split is not None:
            row["split"] = split
        return row


@dataclass(frozen=True)
class _Sentence:
    text: str
    original_index: int
    token_count: int


def whitespace_token_count(text: str | None) -> int:
    """Cheap token counter used consistently for length reporting."""

    if not text:
        return 0
    return len(str(text).split())


def split_sentences(text: str | None) -> list[str]:
    """Sentence split with NLTK when available and a regex fallback otherwise."""

    if not text:
        return []
    normalized = re.sub(r"\s+", " ", str(text)).strip()
    if not normalized:
        return []

    try:
        import nltk

        return [s.strip() for s in nltk.sent_tokenize(normalized) if s.strip()]
    except Exception:
        pieces = _SENTENCE_SPLIT_RE.split(normalized)
        return [s.strip() for s in pieces if s.strip()]


def _word_tokens(text: str) -> list[str]:
    return [m.group(0).lower() for m in _WORD_RE.finditer(text)]


def _valid_sentence(text: str, min_tokens: int, max_tokens: int) -> bool:
    tokens = _word_tokens(text)
    if len(tokens) < min_tokens or len(tokens) > max_tokens:
        return False
    lowered = text.strip().lower()
    if lowered in _ABBREVIATIONS:
        return False
    if sum(ch.isalpha() for ch in text) < 12:
        return False
    return True


def _candidate_score(sentence: _Sentence, total_sentences: int) -> float:
    """Light heuristic only used to cap extreme documents before graph ranking."""

    words = _word_tokens(sentence.text)
    if not words:
        return 0.0
    unique = len(set(words))
    alpha_ratio = sum(w.isalpha() for w in words) / len(words)
    position = sentence.original_index / max(total_sentences - 1, 1)
    early_bonus = 0.15 if position < 0.12 else 0.0
    legal_bonus = 0.0
    for marker in ("court", "plaintiff", "defendant", "claim", "rights", "injunction"):
        if marker in words:
            legal_bonus += 0.04
    return math.log1p(sentence.token_count) + 0.015 * unique + alpha_ratio + early_bonus + legal_bonus


def _build_candidates(
    sentences: list[str],
    *,
    min_sentence_tokens: int,
    max_sentence_tokens: int,
    max_candidate_sentences: int,
) -> list[_Sentence]:
    candidates = [
        _Sentence(text=s, original_index=i, token_count=whitespace_token_count(s))
        for i, s in enumerate(sentences)
        if _valid_sentence(s, min_sentence_tokens, max_sentence_tokens)
    ]
    if len(candidates) <= max_candidate_sentences:
        return candidates

    # Keep candidates across the full document instead of taking only the lead.
    n_bins = min(50, max_candidate_sentences)
    per_bin = max(1, math.ceil(max_candidate_sentences / n_bins))
    selected: list[_Sentence] = []
    for start in range(0, len(candidates), math.ceil(len(candidates) / n_bins)):
        bin_sentences = candidates[start : start + math.ceil(len(candidates) / n_bins)]
        bin_sentences = sorted(
            bin_sentences,
            key=lambda s: _candidate_score(s, len(sentences)),
            reverse=True,
        )
        selected.extend(bin_sentences[:per_bin])
        if len(selected) >= max_candidate_sentences:
            break
    return sorted(selected[:max_candidate_sentences], key=lambda s: s.original_index)


def _tfidf(sentences: Iterable[str], max_features: int):
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words=list(ENGLISH_STOP_WORDS),
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z'-]{2,}\b",
        max_df=0.92,
        min_df=1,
        max_features=max_features,
        norm="l2",
    )
    return vectorizer.fit_transform(sentences)


def _pagerank(graph: csr_matrix, damping: float, max_iter: int, tol: float) -> np.ndarray:
    n = graph.shape[0]
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([1.0])

    graph = graph.astype(float).tocsr()
    row_sums = np.asarray(graph.sum(axis=1)).ravel()
    nonzero = row_sums > 0
    if nonzero.any():
        inv = np.zeros_like(row_sums, dtype=float)
        inv[nonzero] = 1.0 / row_sums[nonzero]
        transition = csr_matrix(np.diag(inv)) @ graph
    else:
        transition = graph

    rank = np.full(n, 1.0 / n)
    teleport = (1.0 - damping) / n
    for _ in range(max_iter):
        dangling_mass = rank[~nonzero].sum() / n
        next_rank = teleport + damping * (transition.T @ rank + dangling_mass)
        if np.abs(next_rank - rank).sum() < tol:
            return np.asarray(next_rank).ravel()
        rank = np.asarray(next_rank).ravel()
    return rank


def _sentence_graph(
    texts: list[str],
    *,
    method: ExtractiveMethod,
    lexrank_threshold: float,
    textrank_min_similarity: float,
    max_features: int,
) -> csr_matrix:
    tfidf = _tfidf(texts, max_features=max_features)
    similarity = (tfidf @ tfidf.T).tocoo()
    mask = similarity.row != similarity.col
    rows = similarity.row[mask]
    cols = similarity.col[mask]
    data = similarity.data[mask]

    if method == "lexrank":
        keep = data >= lexrank_threshold
        data = np.ones(int(keep.sum()), dtype=float)
        rows = rows[keep]
        cols = cols[keep]
    elif method == "textrank":
        keep = data >= textrank_min_similarity
        data = data[keep]
        rows = rows[keep]
        cols = cols[keep]
    else:
        raise ValueError(f"Unknown extractive method: {method}")

    return csr_matrix((data, (rows, cols)), shape=(len(texts), len(texts)))


def _select_sentences(
    candidates: list[_Sentence],
    scores: np.ndarray,
    *,
    target_tokens: int,
    max_sentences: int | None,
) -> list[_Sentence]:
    order = np.argsort(-scores)
    selected: list[_Sentence] = []
    used_tokens = 0

    for idx in order:
        sentence = candidates[int(idx)]
        if max_sentences is not None and len(selected) >= max_sentences:
            break
        if selected and used_tokens + sentence.token_count > target_tokens:
            continue
        selected.append(sentence)
        used_tokens += sentence.token_count
        if used_tokens >= target_tokens:
            break

    if not selected and candidates:
        selected = [candidates[int(order[0])]]
    return sorted(selected, key=lambda s: s.original_index)


def extractive_summarize(
    text: str,
    *,
    method: ExtractiveMethod = "lexrank",
    target_tokens: int = 3500,
    max_sentences: int | None = None,
    min_sentence_tokens: int = 8,
    max_sentence_tokens: int = 120,
    max_candidate_sentences: int = 650,
    lexrank_threshold: float = 0.10,
    textrank_min_similarity: float = 0.01,
    damping: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-6,
    max_features: int = 5000,
) -> ExtractiveSummary:
    """Reduce a long source document with LexRank or TextRank.

    Parameters mirror the README section: LexRank uses a 0.10 cosine threshold;
    TextRank keeps weighted edges above 0.01; both rank at most 650 candidate
    sentences and then keep sentences up to the requested output token budget.
    """

    original_tokens = whitespace_token_count(text)
    sentences = split_sentences(text)
    candidates = _build_candidates(
        sentences,
        min_sentence_tokens=min_sentence_tokens,
        max_sentence_tokens=max_sentence_tokens,
        max_candidate_sentences=max_candidate_sentences,
    )

    if not candidates:
        summary = " ".join(sentences[: max_sentences or 5])
        reduced_tokens = whitespace_token_count(summary)
        return ExtractiveSummary(
            method=method,
            summary_text=summary,
            original_tokens=original_tokens,
            reduced_tokens=reduced_tokens,
            original_sentences=len(sentences),
            selected_sentences=len(sentences[: max_sentences or 5]),
            candidate_sentences=0,
            reduction_pct=1.0 - (reduced_tokens / max(original_tokens, 1)),
        )

    try:
        graph = _sentence_graph(
            [s.text for s in candidates],
            method=method,
            lexrank_threshold=lexrank_threshold,
            textrank_min_similarity=textrank_min_similarity,
            max_features=max_features,
        )
        scores = _pagerank(graph, damping=damping, max_iter=max_iter, tol=tol)
    except ValueError:
        scores = np.array([_candidate_score(s, len(sentences)) for s in candidates], dtype=float)
    selected = _select_sentences(
        candidates,
        scores,
        target_tokens=target_tokens,
        max_sentences=max_sentences,
    )
    summary = " ".join(s.text for s in selected)
    reduced_tokens = whitespace_token_count(summary)
    return ExtractiveSummary(
        method=method,
        summary_text=summary,
        original_tokens=original_tokens,
        reduced_tokens=reduced_tokens,
        original_sentences=len(sentences),
        selected_sentences=len(selected),
        candidate_sentences=len(candidates),
        reduction_pct=1.0 - (reduced_tokens / max(original_tokens, 1)),
    )


def _run_parquet(args: argparse.Namespace) -> pd.DataFrame:
    df = pd.read_parquet(args.input)
    if args.split:
        df = df[df["split"] == args.split]
    if args.limit is not None:
        df = df.head(args.limit)

    rows = []
    for row in tqdm(df.itertuples(index=False), total=len(df), desc=f"{args.method} reduction"):
        result = extractive_summarize(
            getattr(row, args.text_column),
            method=args.method,
            target_tokens=args.target_tokens,
            max_sentences=args.max_sentences,
            max_candidate_sentences=args.max_candidate_sentences,
            lexrank_threshold=args.lexrank_threshold,
            textrank_min_similarity=args.textrank_min_similarity,
        )
        rows.append(result.to_length_row(case_id=getattr(row, "case_id", None), split=getattr(row, "split", None)))

    out = pd.DataFrame(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)
    logger.info("Wrote extractive length report -> %s", output)
    if not out.empty:
        logger.info("Median reduction: %.1f%%", out["reduction_pct"].median() * 100)
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LexRank/TextRank extractive reduction.")
    parser.add_argument("--text", help="Summarize raw text directly.")
    parser.add_argument("--input", default="data/multilexsum_clean.parquet", help="Input parquet.")
    parser.add_argument("--output", default=str(RESULTS_DIR / "extractive_lengths.csv"), help="Output CSV.")
    parser.add_argument("--text-column", default="source_text", help="Parquet text column.")
    parser.add_argument("--split", choices=["train", "val", "test"], help="Optional split filter.")
    parser.add_argument("--limit", type=int, help="Optional row limit for quick experiments.")
    parser.add_argument("--method", choices=["lexrank", "textrank"], default="lexrank")
    parser.add_argument("--target-tokens", type=int, default=3500)
    parser.add_argument("--max-sentences", type=int)
    parser.add_argument("--max-candidate-sentences", type=int, default=650)
    parser.add_argument("--lexrank-threshold", type=float, default=0.10)
    parser.add_argument("--textrank-min-similarity", type=float, default=0.01)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.text:
        result = extractive_summarize(
            args.text,
            method=args.method,
            target_tokens=args.target_tokens,
            max_sentences=args.max_sentences,
            max_candidate_sentences=args.max_candidate_sentences,
            lexrank_threshold=args.lexrank_threshold,
            textrank_min_similarity=args.textrank_min_similarity,
        )
        print(result.summary_text)
        print(f"\nReduced {result.original_tokens} -> {result.reduced_tokens} tokens")
        return
    _run_parquet(args)


if __name__ == "__main__":
    main()
