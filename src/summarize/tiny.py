"""T5-small fine-tuning and inference for one-sentence tiny summaries.

Training uses `short_ref -> tiny_ref` pairs. At inference, the pipeline feeds
the generated short summary to the fine-tuned T5 checkpoint, which keeps tiny
generation cheap and less exposed to the full legal source.
"""

from __future__ import annotations

import argparse
import inspect
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.utils import MODELS_DIR, RESULTS_DIR, get_logger, set_seed

logger = get_logger(__name__)

DEFAULT_MODEL_DIR = MODELS_DIR / "t5_tiny_summarizer"
DEFAULT_BASE_MODEL = "t5-small"


@dataclass(frozen=True)
class TinyTrainConfig:
    input_path: Path = Path("data/multilexsum_clean.parquet")
    output_dir: Path = DEFAULT_MODEL_DIR
    source_column: str = "short_ref"
    target_column: str = "tiny_ref"
    base_model: str = DEFAULT_BASE_MODEL
    epochs: int = 3
    learning_rate: float = 5e-5
    batch_size: int = 4
    max_source_length: int = 256
    max_target_length: int = 48
    max_train: int | None = None
    max_val: int | None = None
    seed: int = 42


class TinyT5Summarizer:
    """Inference wrapper for the fine-tuned tiny-summary model."""

    def __init__(
        self,
        model_dir: str | Path = DEFAULT_MODEL_DIR,
        fallback_model: str = DEFAULT_BASE_MODEL,
        device: str | None = None,
        max_source_length: int = 256,
        max_new_tokens: int = 40,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError("Install transformers, torch, and sentencepiece before tiny inference.") from exc

        model_path = Path(model_dir)
        model_name = str(model_path) if (model_path / "config.json").exists() else fallback_model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.torch = torch
        self.max_source_length = max_source_length
        self.max_new_tokens = max_new_tokens

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
        logger.info("Loaded tiny summarizer %s on %s", model_name, device)

    def summarize(self, short_summary: str) -> str:
        prompt = f"summarize: {short_summary.strip()}"
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_source_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with self.torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                num_beams=4,
                no_repeat_ngram_size=2,
                length_penalty=0.8,
            )
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


def _prepare_dataset(df: pd.DataFrame, source_column: str, target_column: str, max_rows: int | None):
    from datasets import Dataset

    work = df[[source_column, target_column]].dropna().rename(
        columns={source_column: "source", target_column: "target"}
    )
    if max_rows is not None:
        work = work.head(max_rows)
    return Dataset.from_pandas(work.reset_index(drop=True))


def _training_args_kwargs(config: TinyTrainConfig) -> dict:
    from transformers import Seq2SeqTrainingArguments

    params = inspect.signature(Seq2SeqTrainingArguments.__init__).parameters
    kwargs = {
        "output_dir": str(config.output_dir),
        "learning_rate": config.learning_rate,
        "per_device_train_batch_size": config.batch_size,
        "per_device_eval_batch_size": config.batch_size,
        "num_train_epochs": config.epochs,
        "weight_decay": 0.01,
        "predict_with_generate": True,
        "logging_steps": 25,
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "report_to": [],
        "seed": config.seed,
    }
    if "eval_strategy" in params:
        kwargs["eval_strategy"] = "epoch"
    else:
        kwargs["evaluation_strategy"] = "epoch"
    return kwargs


def train_tiny_model(config: TinyTrainConfig) -> pd.DataFrame:
    """Fine-tune T5-small and write the checkpoint plus validation-loss logs."""

    try:
        from transformers import (
            AutoModelForSeq2SeqLM,
            AutoTokenizer,
            DataCollatorForSeq2Seq,
            Seq2SeqTrainer,
            Seq2SeqTrainingArguments,
        )
    except ImportError as exc:
        raise ImportError("Install transformers, torch, datasets, and sentencepiece before training.") from exc

    set_seed(config.seed)
    df = pd.read_parquet(config.input_path)
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]

    train_ds = _prepare_dataset(train_df, config.source_column, config.target_column, config.max_train)
    val_ds = _prepare_dataset(val_df, config.source_column, config.target_column, config.max_val)

    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model)

    def preprocess(batch):
        inputs = [f"summarize: {x}" for x in batch["source"]]
        model_inputs = tokenizer(
            inputs,
            max_length=config.max_source_length,
            truncation=True,
        )
        labels = tokenizer(
            text_target=batch["target"],
            max_length=config.max_target_length,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_train = train_ds.map(preprocess, batched=True, remove_columns=train_ds.column_names)
    tokenized_val = val_ds.map(preprocess, batched=True, remove_columns=val_ds.column_names)

    args = Seq2SeqTrainingArguments(**_training_args_kwargs(config))
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    trainer_kwargs = {
        "model": model,
        "args": args,
        "train_dataset": tokenized_train,
        "eval_dataset": tokenized_val,
        "data_collator": collator,
    }
    trainer_params = inspect.signature(Seq2SeqTrainer.__init__).parameters
    if "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    trainer = Seq2SeqTrainer(**trainer_kwargs)

    trainer.train()
    trainer.save_model(str(config.output_dir))
    tokenizer.save_pretrained(str(config.output_dir))
    logger.info("Saved fine-tuned T5 checkpoint -> %s", config.output_dir)

    history = pd.DataFrame(trainer.state.log_history)
    results_path = RESULTS_DIR / "t5_tiny_val_loss.csv"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    history.to_csv(results_path, index=False)
    logger.info("Wrote training log -> %s", results_path)

    if "eval_loss" in history.columns:
        import matplotlib.pyplot as plt

        curve = history.dropna(subset=["eval_loss"])
        if not curve.empty:
            plt.figure(figsize=(6, 4))
            plt.plot(curve["epoch"], curve["eval_loss"], marker="o")
            plt.xlabel("Epoch")
            plt.ylabel("Validation loss")
            plt.title("T5-small tiny-summary fine-tune")
            plt.tight_layout()
            plot_path = RESULTS_DIR / "t5_tiny_val_loss.png"
            plt.savefig(plot_path, dpi=160)
            plt.close()
            logger.info("Wrote validation-loss curve -> %s", plot_path)

    return history


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune or run T5-small tiny summarization.")
    parser.add_argument("--train", action="store_true", help="Fine-tune the model.")
    parser.add_argument("--input", default="data/multilexsum_clean.parquet")
    parser.add_argument("--output-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--source-column", default="short_ref")
    parser.add_argument("--target-column", default="tiny_ref")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-train", type=int)
    parser.add_argument("--max-val", type=int)
    parser.add_argument("--text", help="Short summary to compress at inference time.")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"])
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.train:
        config = TinyTrainConfig(
            input_path=Path(args.input),
            output_dir=Path(args.output_dir),
            source_column=args.source_column,
            target_column=args.target_column,
            base_model=args.base_model,
            epochs=args.epochs,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            max_train=args.max_train,
            max_val=args.max_val,
        )
        train_tiny_model(config)
        return

    if not args.text:
        raise SystemExit("Provide --train or --text.")
    summarizer = TinyT5Summarizer(model_dir=args.output_dir, device=args.device)
    print(summarizer.summarize(args.text))


if __name__ == "__main__":
    main()
