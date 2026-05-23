"""Unified multi-granularity summarization pipeline."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from src.summarize.abstractive import AbstractiveSummarizer
from src.summarize.extractive import extractive_summarize
from src.summarize.tiny import DEFAULT_MODEL_DIR, TinyT5Summarizer


@dataclass(frozen=True)
class PipelineConfig:
    extractive_method: str = "lexrank"
    extractive_tokens: int = 3500
    extractive_candidates: int = 650
    abstractive_model_key: str = "bart-large-cnn"
    abstractive_model_name: str | None = None
    tiny_model_dir: Path = DEFAULT_MODEL_DIR
    device: str | None = None


class MultiGranularitySummarizer:
    """Single object that returns `{long, short, tiny}` summaries."""

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()
        self.abstractive = AbstractiveSummarizer(
            model_key=self.config.abstractive_model_key,
            model_name=self.config.abstractive_model_name,
            device=self.config.device,
        )
        self.tiny = TinyT5Summarizer(
            model_dir=self.config.tiny_model_dir,
            device=self.config.device,
        )

    def summarize(self, text: str) -> dict[str, str]:
        reduction = extractive_summarize(
            text,
            method=self.config.extractive_method,  # type: ignore[arg-type]
            target_tokens=self.config.extractive_tokens,
            max_candidate_sentences=self.config.extractive_candidates,
        )
        abstractive = self.abstractive.summarize(text, reduced_text=reduction.summary_text)
        tiny = self.tiny.summarize(abstractive.short)
        return {
            "long": abstractive.long,
            "short": abstractive.short,
            "tiny": tiny,
        }


def summarize(text: str) -> dict[str, str]:
    """Convenience entry point requested by the README."""

    return MultiGranularitySummarizer().summarize(text)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the full multi-granularity summarization pipeline.")
    parser.add_argument("--text", help="Raw text to summarize.")
    parser.add_argument("--input-file", help="Path to a text file to summarize.")
    parser.add_argument("--output", help="Optional JSON output path.")
    parser.add_argument("--extractive-method", choices=["lexrank", "textrank"], default="lexrank")
    parser.add_argument("--extractive-tokens", type=int, default=3500)
    parser.add_argument("--model-key", default="bart-large-cnn")
    parser.add_argument("--model-name")
    parser.add_argument("--tiny-model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"])
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.input_file:
        text = Path(args.input_file).read_text(encoding="utf-8")
    elif args.text:
        text = args.text
    else:
        raise SystemExit("Provide --text or --input-file.")

    config = PipelineConfig(
        extractive_method=args.extractive_method,
        extractive_tokens=args.extractive_tokens,
        abstractive_model_key=args.model_key,
        abstractive_model_name=args.model_name,
        tiny_model_dir=Path(args.tiny_model_dir),
        device=args.device,
    )
    result = MultiGranularitySummarizer(config).summarize(text)
    payload = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
