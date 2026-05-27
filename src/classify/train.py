"""Unified training CLI for §3 classification.

This is the *one* command anyone (Feng, teammates, the Gradio app) runs
to (re)produce a single (task × model × text_source) result:

    python -m src.classify.train --task class_action --model nb
    python -m src.classify.train --task case_type    --model lr --text-source long_pred

Each invocation:
  1. loads the configured slice via ``src.classify.data``
  2. trains the requested classifier
  3. evaluates on val + test
  4. saves a self-contained pickled Pipeline (TF-IDF + classifier)
  5. appends a row to ``results/classification_metrics.csv``
  6. writes a confusion-matrix PNG for the test split

W8 deep models (lstm, bert) plug into this same CLI by adding new
``--model`` choices once their modules exist.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")  # headless: required when invoked from a server / CI
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from src.classify.classical import (
    ClassifierMetrics,
    evaluate_classifier,
    train_logistic_regression,
    train_naive_bayes,
)
from src.classify.data import Task, TextSource, load_classification_data
from src.utils import MODELS_DIR, RESULTS_DIR, get_logger, set_seed

logger = get_logger(__name__)

METRICS_CSV = RESULTS_DIR / "classification_metrics.csv"
CONFUSION_DIR = RESULTS_DIR / "confusion_matrices"

# Classical models go through ``Pipeline``. Deep models (lstm/bert) are
# handled via their own training routines (see ``_run_deep`` below) and
# evaluated through a small shared helper.
CLASSICAL_TRAINERS: dict[str, Callable[..., Pipeline]] = {
    "nb": train_naive_bayes,
    "lr": train_logistic_regression,
}
DEEP_MODELS = {"lstm", "bert"}
ALL_MODELS = sorted(set(CLASSICAL_TRAINERS) | DEEP_MODELS)

# Map ``--task`` flag → short tag used in artifact filenames.
TASK_TAG: dict[str, str] = {
    "class_action": "classaction",
    "case_type": "casetype",
}

METRICS_HEADER = [
    "task",
    "model",
    "input_source",
    "split",
    "accuracy",
    "f1_macro",
    "f1_weighted",
    "auc_roc",
    "train_seconds",
    "notes",
]


def _save_pipeline(pipe: Pipeline, task: Task, model: str) -> Path:
    path = MODELS_DIR / f"{model}_{TASK_TAG[task]}.pkl"
    with path.open("wb") as fh:
        pickle.dump(pipe, fh)
    logger.info("Saved pipeline → %s", path)
    return path


def _append_metrics_row(
    *,
    task: str,
    model: str,
    input_source: str,
    split: str,
    metrics: ClassifierMetrics,
    train_seconds: float,
    notes: str,
) -> None:
    """Idempotent header creation, append-only writes — safe under reruns."""
    new_file = not METRICS_CSV.exists()
    METRICS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with METRICS_CSV.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=METRICS_HEADER)
        if new_file:
            writer.writeheader()
        writer.writerow(
            {
                "task": task,
                "model": model,
                "input_source": input_source,
                "split": split,
                "accuracy": f"{metrics.accuracy:.4f}",
                "f1_macro": f"{metrics.f1_macro:.4f}",
                "f1_weighted": f"{metrics.f1_weighted:.4f}",
                "auc_roc": "" if metrics.auc_roc is None else f"{metrics.auc_roc:.4f}",
                "train_seconds": f"{train_seconds:.1f}",
                "notes": notes,
            }
        )


def _plot_confusion(
    cm: np.ndarray,
    label_names: list[str],
    *,
    task: str,
    model: str,
    split: str,
) -> Path:
    """Heat-mapped confusion matrix; the same plotting code is reused by LSTM/BERT."""
    CONFUSION_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4) if len(label_names) == 2 else (7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(label_names)))
    ax.set_yticks(range(len(label_names)))
    ax.set_xticklabels(label_names, rotation=30, ha="right")
    ax.set_yticklabels(label_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{model.upper()} · {task} · {split}")

    # Annotate cells. Use white text on dark cells so values stay legible.
    vmax = cm.max() if cm.size else 1
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > vmax / 2 else "black"
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color=color, fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    path = CONFUSION_DIR / f"{model}_{TASK_TAG[task]}_{split}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved confusion matrix → %s", path)
    return path


def _save_classification_report(
    metrics: ClassifierMetrics,
    *,
    task: str,
    model: str,
    split: str,
) -> None:
    """Per-class precision/recall — used by slide 9 case_type table."""
    report_dir = RESULTS_DIR / "classification_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / f"{model}_{TASK_TAG[task]}_{split}.json"
    with path.open("w") as fh:
        json.dump(metrics.per_class_report, fh, indent=2)
    logger.info("Saved per-class report → %s", path)


def _metrics_from_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    proba: np.ndarray | None,
    label_names: list[str],
) -> ClassifierMetrics:
    """Same metric bundle as ``evaluate_classifier`` but driven by raw arrays.

    Used by the deep-model path where we already have predictions in hand
    and don't want to wrap the model in an sklearn ``Pipeline``.
    """
    n_classes = len(label_names)
    labels = list(range(n_classes))
    report = classification_report(
        y_true, y_pred, labels=labels, target_names=label_names,
        output_dict=True, zero_division=0,
    )
    auc: float | None = None
    if proba is not None:
        unique = set(np.unique(y_true).tolist())
        try:
            if n_classes == 2 and unique == {0, 1}:
                auc = float(roc_auc_score(y_true, proba[:, 1]))
            elif len(unique) == n_classes:
                auc = float(roc_auc_score(y_true, proba, multi_class="ovr", average="macro"))
        except ValueError as exc:
            logger.warning("AUC unavailable: %s", exc)

    return ClassifierMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        f1_macro=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        f1_weighted=float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        auc_roc=auc,
        per_class_report=report,
        confusion=confusion_matrix(y_true, y_pred, labels=labels),
    )


def _train_classical(task: Task, model: str, data, *, text_source: TextSource) -> Pipeline:
    started = time.perf_counter()
    pipe = CLASSICAL_TRAINERS[model](data.X_train, data.y_train)
    train_seconds = time.perf_counter() - started
    _save_pipeline(pipe, task, model)

    for split, X, y in (
        ("val", data.X_val, data.y_val),
        ("test", data.X_test, data.y_test),
    ):
        metrics = evaluate_classifier(pipe, X, y, label_names=data.label_names)
        notes = _build_notes(model, text_source, n_classes=data.n_classes)
        _append_metrics_row(
            task=task, model=model, input_source=text_source,
            split=split, metrics=metrics,
            train_seconds=train_seconds, notes=notes,
        )
        _save_classification_report(metrics, task=task, model=model, split=split)
        if split == "test":
            _plot_confusion(metrics.confusion, data.label_names,
                            task=task, model=model, split=split)
        logger.info(
            "  [%s] acc=%.4f f1_macro=%.4f auc=%s",
            split, metrics.accuracy, metrics.f1_macro,
            "n/a" if metrics.auc_roc is None else f"{metrics.auc_roc:.4f}",
        )
    return pipe


def _train_lstm(task: Task, data, *, text_source: TextSource, device: str | None,
                epochs: int, batch_size: int, max_len: int):
    from src.classify.lstm import LSTMConfig, predict_lstm, train_lstm

    cfg = LSTMConfig(epochs=epochs, batch_size=batch_size, max_len=max_len)
    task_tag = TASK_TAG[task]
    started = time.perf_counter()
    model_obj, vocab, dev = train_lstm(
        data.X_train, data.y_train, data.X_val, data.y_val,
        label_names=data.label_names, task_tag=task_tag,
        config=cfg, device=device,
    )
    train_seconds = time.perf_counter() - started

    for split, X, y in (
        ("val", data.X_val, data.y_val),
        ("test", data.X_test, data.y_test),
    ):
        preds, proba = predict_lstm(model_obj, vocab, X, max_len=cfg.max_len, device=dev)
        metrics = _metrics_from_predictions(y, preds, proba, data.label_names)
        notes = _build_notes("lstm", text_source, n_classes=data.n_classes)
        _append_metrics_row(
            task=task, model="lstm", input_source=text_source,
            split=split, metrics=metrics,
            train_seconds=train_seconds, notes=notes,
        )
        _save_classification_report(metrics, task=task, model="lstm", split=split)
        if split == "test":
            _plot_confusion(metrics.confusion, data.label_names,
                            task=task, model="lstm", split=split)
        logger.info(
            "  [%s] acc=%.4f f1_macro=%.4f auc=%s",
            split, metrics.accuracy, metrics.f1_macro,
            "n/a" if metrics.auc_roc is None else f"{metrics.auc_roc:.4f}",
        )


def _train_bert(task: Task, data, *, text_source: TextSource, device: str | None,
                epochs: int, batch_size: int, max_length: int, model_name: str | None):
    from src.classify.bert import BertConfig, DEFAULT_MODEL_NAME, predict_bert, train_bert

    cfg = BertConfig(
        model_name=model_name or DEFAULT_MODEL_NAME,
        epochs=epochs, batch_size=batch_size, max_length=max_length,
    )
    task_tag = TASK_TAG[task]
    started = time.perf_counter()
    model_obj, tokenizer, dev = train_bert(
        data.X_train, data.y_train, data.X_val, data.y_val,
        label_names=data.label_names, task_tag=task_tag,
        config=cfg, device=device,
    )
    train_seconds = time.perf_counter() - started

    for split, X, y in (
        ("val", data.X_val, data.y_val),
        ("test", data.X_test, data.y_test),
    ):
        preds, proba = predict_bert(
            model_obj, tokenizer, X, max_length=cfg.max_length, device=dev,
        )
        metrics = _metrics_from_predictions(y, preds, proba, data.label_names)
        notes = _build_notes("bert", text_source, n_classes=data.n_classes,
                             model_name=cfg.model_name)
        _append_metrics_row(
            task=task, model="bert", input_source=text_source,
            split=split, metrics=metrics,
            train_seconds=train_seconds, notes=notes,
        )
        _save_classification_report(metrics, task=task, model="bert", split=split)
        if split == "test":
            _plot_confusion(metrics.confusion, data.label_names,
                            task=task, model="bert", split=split)
        logger.info(
            "  [%s] acc=%.4f f1_macro=%.4f auc=%s",
            split, metrics.accuracy, metrics.f1_macro,
            "n/a" if metrics.auc_roc is None else f"{metrics.auc_roc:.4f}",
        )


def run(
    task: Task,
    model: str,
    text_source: TextSource = "long_ref",
    *,
    seed: int = 42,
    device: str | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    max_len: int | None = None,
    bert_model_name: str | None = None,
) -> None:
    """Dispatch to the classical or deep training path based on ``model``."""
    set_seed(seed)
    if model not in CLASSICAL_TRAINERS and model not in DEEP_MODELS:
        raise ValueError(f"Unknown --model {model!r}; choices: {ALL_MODELS}")

    data = load_classification_data(task=task, text_source=text_source)
    logger.info("Training %s on task=%s text_source=%s …", model, task, text_source)

    if model in CLASSICAL_TRAINERS:
        _train_classical(task=task, model=model, data=data, text_source=text_source)
    elif model == "lstm":
        _train_lstm(
            task=task, data=data, text_source=text_source, device=device,
            epochs=epochs or 10, batch_size=batch_size or 16, max_len=max_len or 400,
        )
    elif model == "bert":
        _train_bert(
            task=task, data=data, text_source=text_source, device=device,
            epochs=epochs or 3, batch_size=batch_size or 8,
            max_length=max_len or 512, model_name=bert_model_name,
        )


def _build_notes(model: str, text_source: str, *, n_classes: int,
                 model_name: str | None = None) -> str:
    """Short human-readable note stored in metrics.csv — used by slides 7–9."""
    pieces = []
    if model == "nb":
        pieces.append("ComplementNB(alpha=0.3)")
    elif model == "lr":
        pieces.append("LR L2 saga class_weight=balanced")
    elif model == "lstm":
        pieces.append("BiLSTM(128)+Word2Vec(300)")
    elif model == "bert":
        pieces.append(f"BERT={model_name or 'legal-bert-base'}")
    pieces.append(f"text={text_source}")
    pieces.append(f"{n_classes}-class")
    return " | ".join(pieces)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train + evaluate one §3 classifier.")
    parser.add_argument("--task", required=True, choices=list(TASK_TAG))
    parser.add_argument("--model", required=True, choices=ALL_MODELS)
    parser.add_argument(
        "--text-source", default="long_ref",
        choices=["long_ref", "long_pred", "source_text"],
        help="Which column from the cleaned parquet feeds the model.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"],
                        help="Override auto-detected device (lstm/bert only).")
    parser.add_argument("--epochs", type=int,
                        help="LSTM default=10, BERT default=3. Classical ignores.")
    parser.add_argument("--batch-size", type=int,
                        help="LSTM default=16, BERT default=8. Classical ignores.")
    parser.add_argument("--max-len", type=int,
                        help="LSTM default=400, BERT default=512.")
    parser.add_argument("--bert-model-name",
                        help="HF model id for --model bert (default: legal-bert-base-uncased).")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run(
        task=args.task, model=args.model, text_source=args.text_source,
        seed=args.seed, device=args.device,
        epochs=args.epochs, batch_size=args.batch_size, max_len=args.max_len,
        bert_model_name=args.bert_model_name,
    )


if __name__ == "__main__":
    main()
