"""Abstractive generation for long and short summaries.

Default model choice is `facebook/bart-large-cnn`: Stage A already reduces the
case to a few thousand tokens, so a reliable 1k-token CNN/DailyMail model is a
better course-project tradeoff than serving LED-large for every example.
Pegasus-X and LED entries are kept in the registry for ablations.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd
from tqdm import tqdm

from src.summarize.extractive import extractive_summarize
from src.utils import RESULTS_DIR, get_logger

logger = get_logger(__name__)

Granularity = Literal["long", "short"]

MODEL_REGISTRY = {
    "bart-large-cnn": {
        "model_name": "facebook/bart-large-cnn",
        "context_tokens": 1024,
        "notes": "Reliable, easy to run after extractive reduction; chosen default.",
    },
    "pegasus-x": {
        "model_name": "google/pegasus-x-large",
        "context_tokens": 4096,
        "notes": "Longer context, larger memory footprint; useful as an ablation.",
    },
    "led-large": {
        "model_name": "allenai/led-large-16384-arxiv",
        "context_tokens": 16384,
        "notes": "Best context length, but slow and memory-heavy for the team pipeline.",
    },
}

GENERATION_PRESETS = {
    "long": {
        "max_length": 420,
        "min_length": 160,
        "num_beams": 4,
        "length_penalty": 1.0,
        "no_repeat_ngram_size": 3,
    },
    "short": {
        "max_length": 120,
        "min_length": 45,
        "num_beams": 4,
        "length_penalty": 0.9,
        "no_repeat_ngram_size": 3,
    },
}


@dataclass(frozen=True)
class AbstractiveOutput:
    long: str
    short: str
    reduced_text: str
    model_name: str


class AbstractiveSummarizer:
    """BART/Pegasus/LED wrapper with chunk-then-combine generation."""

    def __init__(
        self,
        model_key: str = "bart-large-cnn",
        model_name: str | None = None,
        device: str | None = None,
        max_input_tokens: int | None = None,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "Install summarization dependencies first: "
                "pip install transformers torch sentencepiece accelerate"
            ) from exc

        if model_name is None:
            if model_key not in MODEL_REGISTRY:
                raise ValueError(f"Unknown model_key {model_key!r}; choices: {sorted(MODEL_REGISTRY)}")
            model_name = MODEL_REGISTRY[model_key]["model_name"]
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.torch = torch

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        self.model.to(device)
        self.model.eval()

        registry_context = MODEL_REGISTRY.get(model_key, {}).get("context_tokens", 1024)
        self.max_input_tokens = max_input_tokens or min(registry_context, 1024)
        logger.info("Loaded %s on %s", model_name, device)

    def _token_chunks(self, text: str, *, chunk_tokens: int, stride: int = 80) -> list[str]:
        token_ids = self.tokenizer.encode(text, add_special_tokens=False, truncation=False)
        if not token_ids:
            return [""]
        chunks = []
        start = 0
        while start < len(token_ids):
            end = min(start + chunk_tokens, len(token_ids))
            chunks.append(self.tokenizer.decode(token_ids[start:end], skip_special_tokens=True))
            if end == len(token_ids):
                break
            start = max(end - stride, start + 1)
        return chunks

    def _generate_once(self, text: str, granularity: Granularity) -> str:
        preset = GENERATION_PRESETS[granularity]
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with self.torch.no_grad():
            output_ids = self.model.generate(**inputs, **preset)
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

    def generate(self, text: str, granularity: Granularity = "long") -> str:
        """Generate one summary, chunking inputs longer than the model context."""

        chunk_budget = max(256, self.max_input_tokens - 64)
        chunks = self._token_chunks(text, chunk_tokens=chunk_budget)
        if len(chunks) == 1:
            return self._generate_once(chunks[0], granularity)

        chunk_summaries = [self._generate_once(chunk, "short") for chunk in chunks]
        combined = " ".join(s for s in chunk_summaries if s)
        return self._generate_once(combined, granularity)

    def summarize(self, text: str, reduced_text: str | None = None) -> AbstractiveOutput:
        """Generate long and short summaries from reduced case text."""

        reduced = reduced_text if reduced_text is not None else text
        long_summary = self.generate(reduced, "long")
        # Generate short from long to improve consistency across granularities.
        short_summary = self.generate(long_summary, "short")
        return AbstractiveOutput(
            long=long_summary,
            short=short_summary,
            reduced_text=reduced,
            model_name=self.model_name,
        )


def _run_parquet(args: argparse.Namespace) -> pd.DataFrame:
    df = pd.read_parquet(args.input)
    if args.split:
        df = df[df["split"] == args.split]
    if args.limit is not None:
        df = df.head(args.limit)

    summarizer = AbstractiveSummarizer(
        model_key=args.model_key,
        model_name=args.model_name,
        device=args.device,
        max_input_tokens=args.max_input_tokens,
    )

    rows = []
    for row in tqdm(df.itertuples(index=False), total=len(df), desc="abstractive summaries"):
        text = getattr(row, args.text_column)
        if args.skip_extractive:
            reduced = text
            reduction = None
        else:
            reduction = extractive_summarize(
                text,
                method=args.extractive_method,
                target_tokens=args.extractive_tokens,
                max_candidate_sentences=args.max_candidate_sentences,
            )
            reduced = reduction.summary_text
        output = summarizer.summarize(text, reduced_text=reduced)
        rows.append(
            {
                "case_id": getattr(row, "case_id", None),
                "split": getattr(row, "split", None),
                "model_name": output.model_name,
                "long_pred": output.long,
                "short_pred": output.short,
                "reduced_tokens": reduction.reduced_tokens if reduction else None,
                "original_tokens": reduction.original_tokens if reduction else None,
            }
        )

    out = pd.DataFrame(rows)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    logger.info("Wrote abstractive summaries -> %s", output_path)
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate long/short abstractive summaries.")
    parser.add_argument("--input", default="data/multilexsum_clean.parquet")
    parser.add_argument("--output", default=str(RESULTS_DIR / "abstractive_summaries.csv"))
    parser.add_argument("--text-column", default="source_text")
    parser.add_argument("--split", choices=["train", "val", "test"])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--model-key", choices=sorted(MODEL_REGISTRY), default="bart-large-cnn")
    parser.add_argument("--model-name", help="Override Hugging Face model id.")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--max-input-tokens", type=int)
    parser.add_argument("--skip-extractive", action="store_true")
    parser.add_argument("--extractive-method", choices=["lexrank", "textrank"], default="lexrank")
    parser.add_argument("--extractive-tokens", type=int, default=3500)
    parser.add_argument("--max-candidate-sentences", type=int, default=650)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    _run_parquet(args)


if __name__ == "__main__":
    main()
