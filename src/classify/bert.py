"""Legal-BERT fine-tuning (§3.6).

Model
-----
``nlpaueb/legal-bert-base-uncased`` — a BERT-base (110M params) pretrained
on US/EU legal corpora. Ideal fit for Multi-LexSum; the alternative
``distilbert-base-uncased`` (66M) is supported via ``--model-name`` for
free-Colab fallback.

Inputs
------
The long reference summary or model-generated long summary is ~250–400
tokens — comfortably under BERT's 512 cap, so we use truncation with no
chunking. The full ``source_text`` (median 44k tokens) would need
chunk-then-pool which is out of scope for the course project.

Training defaults
-----------------
- batch_size = 8 (with grad-accum 2 → effective 16) — fits T4 16GB
- lr = 2e-5 (paper default for BERT classification)
- 3 epochs (more overfits on 1129 samples)
- AdamW, linear warmup over 10% of steps
- Class-weighted cross-entropy (matches LR/LSTM treatment)

Output artifacts
----------------
- ``models/bert_{task_tag}/``  — HuggingFace save_pretrained directory
- ``results/training_curves/bert_{task_tag}.png``  — loss + acc per epoch
- ``results/training_curves/bert_{task_tag}.csv``  — same data as CSV
- a row appended to ``results/classification_metrics.csv`` (via train.py)
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.utils import MODELS_DIR, RESULTS_DIR, get_logger, set_seed

logger = get_logger(__name__)

TRAINING_CURVES_DIR = RESULTS_DIR / "training_curves"
DEFAULT_MODEL_NAME = "nlpaueb/legal-bert-base-uncased"


@dataclass(frozen=True)
class BertConfig:
    """Knobs surfaced to the CLI; everything else is internal."""

    model_name: str = DEFAULT_MODEL_NAME
    max_length: int = 512
    batch_size: int = 8
    grad_accum: int = 2
    epochs: int = 3
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    seed: int = 42


def _resolve_device(device: str | None):
    import torch

    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _class_weights(y: np.ndarray, n_classes: int):
    counts = np.bincount(y, minlength=n_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    return counts.sum() / (n_classes * counts)


def _encode(tokenizer, texts: list[str], *, max_length: int):
    return tokenizer(
        list(texts),
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )


def _evaluate_loop(model, loader, device, criterion) -> tuple[float, float]:
    import torch

    model.eval()
    total_loss, total, correct = 0.0, 0, 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            total_loss += criterion(logits, labels).item() * labels.size(0)
            preds = logits.argmax(dim=-1)
            correct += int((preds == labels).sum().item())
            total += labels.size(0)
    return total_loss / max(1, total), correct / max(1, total)


def _save_training_curves(history, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [r["epoch"] for r in history]
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4))
    ax_loss.plot(epochs, [r["train_loss"] for r in history], label="train")
    ax_loss.plot(epochs, [r["val_loss"] for r in history], label="val")
    ax_loss.set_xlabel("epoch"); ax_loss.set_ylabel("loss"); ax_loss.set_title("Loss")
    ax_loss.legend()
    ax_acc.plot(epochs, [r["train_acc"] for r in history], label="train")
    ax_acc.plot(epochs, [r["val_acc"] for r in history], label="val")
    ax_acc.set_xlabel("epoch"); ax_acc.set_ylabel("accuracy"); ax_acc.set_title("Accuracy")
    ax_acc.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved BERT training curves → %s", output_path)


def _save_log_csv(history, output_path: Path) -> None:
    if not history:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def train_bert(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    *,
    label_names: list[str],
    task_tag: str,
    config: BertConfig | None = None,
    device: str | None = None,
):
    """Fine-tune Legal-BERT for one task. Returns ``(model, tokenizer, device)``.

    Persists model + tokenizer under ``models/bert_{task_tag}/`` so the
    Gradio app can ``AutoModelForSequenceClassification.from_pretrained``
    that path directly.
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    cfg = config or BertConfig()
    set_seed(cfg.seed)
    dev = _resolve_device(device)
    logger.info("Legal-BERT on device=%s | task_tag=%s | model=%s",
                dev, task_tag, cfg.model_name)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    n_classes = len(label_names)
    id2label = {i: name for i, name in enumerate(label_names)}
    label2id = {name: i for i, name in enumerate(label_names)}
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=n_classes,
        id2label=id2label,
        label2id=label2id,
    ).to(dev)

    train_enc = _encode(tokenizer, list(X_train), max_length=cfg.max_length)
    val_enc = _encode(tokenizer, list(X_val), max_length=cfg.max_length)

    class TextDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = torch.tensor(labels, dtype=torch.long)

        def __len__(self) -> int:
            return len(self.labels)

        def __getitem__(self, idx):
            return {
                "input_ids": self.encodings["input_ids"][idx],
                "attention_mask": self.encodings["attention_mask"][idx],
                "labels": self.labels[idx],
            }

    train_loader = DataLoader(TextDataset(train_enc, y_train), batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(TextDataset(val_enc, y_val), batch_size=cfg.batch_size, shuffle=False)

    weights = torch.tensor(_class_weights(y_train, n_classes), device=dev)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
    )

    steps_per_epoch = max(1, len(train_loader) // cfg.grad_accum)
    total_steps = steps_per_epoch * cfg.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * cfg.warmup_ratio),
        num_training_steps=total_steps,
    )

    history: list[dict[str, float]] = []
    best_val_acc = -1.0
    best_state: dict | None = None

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss, total, correct = 0.0, 0, 0
        optimizer.zero_grad()
        for step, batch in enumerate(train_loader, start=1):
            input_ids = batch["input_ids"].to(dev)
            attention_mask = batch["attention_mask"].to(dev)
            labels = batch["labels"].to(dev)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels) / cfg.grad_accum
            loss.backward()
            if step % cfg.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item() * cfg.grad_accum * labels.size(0)
            preds = outputs.logits.argmax(dim=-1)
            correct += int((preds == labels).sum().item())
            total += labels.size(0)

        train_loss = total_loss / max(1, total)
        train_acc = correct / max(1, total)
        val_loss, val_acc = _evaluate_loop(model, val_loader, dev, criterion)

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
        })
        logger.info(
            "epoch %d/%d | train loss=%.4f acc=%.4f | val loss=%.4f acc=%.4f",
            epoch, cfg.epochs, train_loss, train_acc, val_loss, val_acc,
        )

        if val_acc > best_val_acc + 1e-4:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    _save_training_curves(history, TRAINING_CURVES_DIR / f"bert_{task_tag}.png")
    _save_log_csv(history, TRAINING_CURVES_DIR / f"bert_{task_tag}.csv")

    out_dir = MODELS_DIR / f"bert_{task_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    (out_dir / "label_names.json").write_text(json.dumps(label_names))
    logger.info("Saved BERT model + tokenizer → %s", out_dir)

    return model, tokenizer, dev


def predict_bert(model, tokenizer, X_texts, *, max_length: int = 512, batch_size: int = 16, device=None):
    """Inference helper used by ``train.py`` for eval + the Gradio app."""
    import torch
    from torch.utils.data import DataLoader

    model.eval()
    dev = device or next(model.parameters()).device
    enc = tokenizer(
        list(X_texts), truncation=True, padding="max_length",
        max_length=max_length, return_tensors="pt",
    )

    class _IDS(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return len(enc["input_ids"])

        def __getitem__(self, idx):
            return {
                "input_ids": enc["input_ids"][idx],
                "attention_mask": enc["attention_mask"][idx],
            }

    loader = DataLoader(_IDS(), batch_size=batch_size, shuffle=False)
    all_proba: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            logits = model(
                input_ids=batch["input_ids"].to(dev),
                attention_mask=batch["attention_mask"].to(dev),
            ).logits
            all_proba.append(torch.softmax(logits, dim=-1).cpu().numpy())
    proba = np.vstack(all_proba) if all_proba else np.zeros((0, 0))
    preds = proba.argmax(axis=-1)
    return preds, proba


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune Legal-BERT for §3.")
    parser.add_argument("--task", required=True, choices=["class_action", "case_type"])
    parser.add_argument("--text-source", default="long_ref",
                        choices=["long_ref", "long_pred", "source_text"])
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"])
    return parser


def main() -> None:
    """Smoke-test entry point. The real CLI path is `src.classify.train --model bert`."""
    from src.classify.data import load_classification_data

    args = build_arg_parser().parse_args()
    data = load_classification_data(task=args.task, text_source=args.text_source)
    cfg = BertConfig(
        model_name=args.model_name, epochs=args.epochs,
        batch_size=args.batch_size, max_length=args.max_length,
    )
    task_tag = "classaction" if args.task == "class_action" else "casetype"
    train_bert(
        data.X_train, data.y_train, data.X_val, data.y_val,
        label_names=data.label_names, task_tag=task_tag,
        config=cfg, device=args.device,
    )


if __name__ == "__main__":
    main()
