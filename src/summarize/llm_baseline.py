"""Zero-shot LLM baseline for multi-granularity summaries.

The default provider is OpenAI `gpt-4o-mini` because it has a 128k context
window and low text-token pricing. Very long cases are still passed through the
same extractive reducer first so the baseline is affordable and reproducible.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from src.summarize.extractive import extractive_summarize, whitespace_token_count
from src.utils import RESULTS_DIR, get_logger

logger = get_logger(__name__)

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_INPUT_PRICE_PER_1M = 0.15
DEFAULT_OUTPUT_PRICE_PER_1M = 0.60

SYSTEM_PROMPT = """You summarize U.S. federal civil-rights litigation for legal NLP evaluation.
Be faithful to the supplied case text. Do not add facts, outcomes, parties, statutes,
or remedies that are not supported by the input."""

USER_PROMPT_TEMPLATE = """Write three summaries of the case text below.

Return strict JSON with exactly these keys:
- "long": 500-800 words, covering parties, claims, procedural posture, facts, and outcome when present.
- "short": 90-140 words, focused on claims and outcome.
- "tiny": one sentence under 35 words.

Case text:
{case_text}
"""


@dataclass(frozen=True)
class CostEstimate:
    input_tokens: int
    output_tokens: int
    input_cost_usd: float
    output_cost_usd: float
    total_cost_usd: float


def estimate_tokens(text: str) -> int:
    """Approximate tokens without requiring tiktoken."""

    if not text:
        return 0
    return max(1, int(whitespace_token_count(text) * 1.33))


def estimate_cost(
    prompt_text: str,
    *,
    expected_output_tokens: int = 900,
    input_price_per_1m: float = DEFAULT_INPUT_PRICE_PER_1M,
    output_price_per_1m: float = DEFAULT_OUTPUT_PRICE_PER_1M,
) -> CostEstimate:
    input_tokens = estimate_tokens(prompt_text)
    input_cost = input_tokens / 1_000_000 * input_price_per_1m
    output_cost = expected_output_tokens / 1_000_000 * output_price_per_1m
    return CostEstimate(
        input_tokens=input_tokens,
        output_tokens=expected_output_tokens,
        input_cost_usd=input_cost,
        output_cost_usd=output_cost,
        total_cost_usd=input_cost + output_cost,
    )


class OpenAIZeroShotBaseline:
    """Minimal OpenAI Chat Completions client wrapper."""

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("Install the OpenAI client first: pip install openai") from exc
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is required to run the zero-shot baseline.")
        self.client = OpenAI()
        self.model = model

    def summarize(self, case_text: str) -> dict[str, Any]:
        prompt = USER_PROMPT_TEMPLATE.format(case_text=case_text)
        response = self.client.chat.completions.create(
            model=self.model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        content = response.choices[0].message.content or "{}"
        data = json.loads(content)
        usage = getattr(response, "usage", None)
        data["input_tokens"] = getattr(usage, "prompt_tokens", None)
        data["output_tokens"] = getattr(usage, "completion_tokens", None)
        return data


def prepare_llm_input(
    text: str,
    *,
    target_tokens: int = 10_000,
    method: str = "lexrank",
    max_candidate_sentences: int = 850,
) -> str:
    """Reduce source text before the LLM if needed."""

    if whitespace_token_count(text) <= target_tokens:
        return text
    return extractive_summarize(
        text,
        method=method,  # type: ignore[arg-type]
        target_tokens=target_tokens,
        max_candidate_sentences=max_candidate_sentences,
    ).summary_text


def _run_parquet(args: argparse.Namespace) -> pd.DataFrame:
    df = pd.read_parquet(args.input)
    if args.split:
        df = df[df["split"] == args.split]
    if args.limit is not None:
        df = df.head(args.limit)

    baseline = None if args.dry_run_cost else OpenAIZeroShotBaseline(model=args.model)
    rows = []
    for row in tqdm(df.itertuples(index=False), total=len(df), desc="llm baseline"):
        reduced = prepare_llm_input(
            getattr(row, args.text_column),
            target_tokens=args.target_tokens,
            method=args.extractive_method,
        )
        prompt = USER_PROMPT_TEMPLATE.format(case_text=reduced)
        cost = estimate_cost(prompt)
        record = {
            "case_id": getattr(row, "case_id", None),
            "split": getattr(row, "split", None),
            "model": args.model,
            "llm_input_tokens_est": cost.input_tokens,
            "llm_cost_usd_est": cost.total_cost_usd,
        }
        if baseline is not None:
            output = baseline.summarize(reduced)
            record.update(
                {
                    "long_pred": output.get("long", ""),
                    "short_pred": output.get("short", ""),
                    "tiny_pred": output.get("tiny", ""),
                    "input_tokens": output.get("input_tokens"),
                    "output_tokens": output.get("output_tokens"),
                }
            )
        rows.append(record)

    out = pd.DataFrame(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)
    logger.info("Wrote LLM baseline output -> %s", output)
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run or cost the zero-shot LLM summarization baseline.")
    parser.add_argument("--input", default="data/multilexsum_clean.parquet")
    parser.add_argument("--output", default=str(RESULTS_DIR / "llm_baseline_summaries.csv"))
    parser.add_argument("--text-column", default="source_text")
    parser.add_argument("--split", choices=["train", "val", "test"])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--target-tokens", type=int, default=10_000)
    parser.add_argument("--extractive-method", choices=["lexrank", "textrank"], default="lexrank")
    parser.add_argument("--dry-run-cost", action="store_true", help="Only estimate prompt tokens/cost.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    _run_parquet(args)


if __name__ == "__main__":
    main()
