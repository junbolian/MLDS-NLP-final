"""Export per-case prediction CSVs for Jianong's error-analysis workflow.

This tool sits on top of the team's existing trained artifacts without
modifying the original training code. By default it uses the baseline
`long_ref` setup described in README §3 and writes CSVs under
`results/predictions/` for notebook 06 and `results/error_cases.md`.
"""

from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path

import numpy as np

from src.classify.bert import predict_bert
from src.classify.data import ClassificationData, load_classification_data
from src.classify.lstm import Vocab, _build_model, predict_lstm
from src.utils import MODELS_DIR, RESULTS_DIR, get_logger

logger = get_logger(__name__)

TASK_TAG = {
    "class_action": "classaction",
    "case_type": "casetype",
}
PREDICTIONS_DIR = RESULTS_DIR / "predictions"


def _artifact_path(task: str, model: str) -> Path:
    stem = f"{model}_{TASK_TAG[task]}"
    if model in {"lr", "nb"}:
        suffix = ".pkl"
    elif model == "lstm":
        suffix = ".pt"
    else:
        suffix = ""
    return MODELS_DIR / f"{stem}{suffix}"


def _load_lr_pipeline(path: Path):
    with path.open("rb") as fh:
        return pickle.load(fh)


def _load_lstm_artifact(path: Path, *, device: str | None = None):
    try:
        import torch
    except ImportError as exc:
        raise ImportError("Install torch before exporting Bi-LSTM predictions.") from exc

    bundle = torch.load(path, map_location=device or "cpu")
    config = bundle["config"]
    vocab_itos = bundle["vocab_itos"]
    vocab = Vocab(
        stoi={tok: idx for idx, tok in enumerate(vocab_itos)},
        itos=vocab_itos,
    )
    embedding_dim = int(bundle["embedding_dim"])
    n_classes = len(bundle["label_names"])
    embedding_matrix = np.zeros((len(vocab_itos), embedding_dim), dtype=np.float32)
    model = _build_model(
        embedding_matrix=embedding_matrix,
        n_classes=n_classes,
        hidden_dim=int(config["hidden_dim"]),
        num_layers=int(config["num_layers"]),
        dropout=float(config["dropout"]),
    )
    model.load_state_dict(bundle["state_dict"])
    if device is not None:
        model = model.to(device)
    model.eval()
    return model, vocab, int(config["max_len"])


def _load_bert_artifact(path: Path):
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise ImportError("Install transformers and torch before exporting BERT predictions.") from exc

    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    return model, tokenizer


def _split_arrays(data: ClassificationData, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if split == "val":
        return data.X_val, data.y_val, data.case_ids_val
    if split == "test":
        return data.X_test, data.y_test, data.case_ids_test
    raise ValueError(f"Unsupported split {split!r}; choose from val/test.")


def _write_predictions(
    *,
    output_path: Path,
    case_ids: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    proba: np.ndarray | None,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for idx, (case_id, true_label, pred_label) in enumerate(zip(case_ids, y_true, y_pred)):
        row: dict[str, object] = {
            "case_id": str(case_id),
            "y_true": int(true_label),
            "y_pred": int(pred_label),
        }
        if proba is not None and idx < len(proba):
            for class_idx, value in enumerate(proba[idx]):
                row[f"y_proba_{class_idx}"] = float(value)
        rows.append(row)

    with output_path.open("w", newline="") as fh:
        fieldnames = list(rows[0].keys()) if rows else ["case_id", "y_true", "y_pred"]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote predictions -> %s", output_path)
    return output_path


def export_predictions(
    *,
    task: str,
    model: str,
    split: str,
    text_source: str = "long_ref",
    device: str | None = None,
) -> Path:
    data = load_classification_data(task=task, text_source=text_source)
    X, y, case_ids = _split_arrays(data, split)
    artifact_path = _artifact_path(task, model)
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"Missing artifact {artifact_path}. Train the team's existing {model} model first."
        )

    if model in {"lr", "nb"}:
        pipe = _load_lr_pipeline(artifact_path)
        y_pred = pipe.predict(X)
        proba = pipe.predict_proba(X) if hasattr(pipe, "predict_proba") else None
    elif model == "lstm":
        model_obj, vocab, max_len = _load_lstm_artifact(artifact_path, device=device)
        y_pred, proba = predict_lstm(
            model_obj,
            vocab,
            X,
            max_len=max_len,
            device=device,
        )
    elif model == "bert":
        model_obj, tokenizer = _load_bert_artifact(artifact_path)
        y_pred, proba = predict_bert(
            model_obj,
            tokenizer,
            X,
            device=device,
        )
    else:
        raise ValueError(f"Unsupported model {model!r}; choose from nb/lr/lstm/bert.")

    output_path = PREDICTIONS_DIR / f"{model}_{TASK_TAG[task]}_{split}.csv"
    return _write_predictions(
        output_path=output_path,
        case_ids=case_ids,
        y_true=y,
        y_pred=y_pred,
        proba=proba,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export per-case prediction CSVs for app error analysis.")
    parser.add_argument("--task", choices=["class_action", "case_type"], help="Single task to export.")
    parser.add_argument("--model", choices=["nb", "lr", "lstm", "bert"], help="Single model to export.")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"])
    parser.add_argument(
        "--all",
        action="store_true",
        help="Export NB, LR, Bi-LSTM, and BERT for both tasks using the selected split.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    jobs: list[tuple[str, str]]
    if args.all:
        jobs = [
            ("class_action", "nb"),
            ("class_action", "lr"),
            ("class_action", "lstm"),
            ("class_action", "bert"),
            ("case_type", "nb"),
            ("case_type", "lr"),
            ("case_type", "lstm"),
            ("case_type", "bert"),
        ]
    else:
        if not args.task or not args.model:
            raise SystemExit("Provide --all or both --task and --model.")
        jobs = [(args.task, args.model)]

    for task, model in jobs:
        export_predictions(
            task=task,
            model=model,
            split=args.split,
            device=args.device,
        )


if __name__ == "__main__":
    main()
