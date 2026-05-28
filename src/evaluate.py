"""Evaluation helpers for project outputs.

Two task families share one CLI:
  * ``--task summarization``  : ROUGE-1/2/L + BERTScore against Multi-LexSum
                                references (Yujun's §2).
  * ``--task classification`` : accuracy / F1 / AUC / per-class report for
                                the §3 classifiers. Reads a predictions CSV
                                with columns ``case_id, y_true, y_pred`` plus
                                an optional ``y_proba_*`` per class for AUC.

The classification path is the unified evaluation entry that the §4 Gradio
app's "Error Analysis" tab calls into.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import pandas as pd
from rouge_score import rouge_scorer
from tqdm import tqdm

from src.utils import RESULTS_DIR, get_logger

logger = get_logger(__name__)
matplotlib.use("Agg")

SUMMARY_GRANULARITIES = ("long", "short", "tiny")


def _safe_text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def _rouge_rows(preds: list[str], refs: list[str]) -> list[dict[str, float]]:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rows = []
    for pred, ref in zip(preds, refs):
        scores = scorer.score(ref, pred)
        rows.append(
            {
                "rouge1": scores["rouge1"].fmeasure,
                "rouge2": scores["rouge2"].fmeasure,
                "rougeL": scores["rougeL"].fmeasure,
            }
        )
    return rows


def _bertscore_rows(
    preds: list[str],
    refs: list[str],
    *,
    model_type: str,
    batch_size: int,
    device: str | None,
) -> list[dict[str, float]]:
    from bert_score import score as bert_score

    precision, recall, f1 = bert_score(
        preds,
        refs,
        lang="en",
        model_type=model_type,
        batch_size=batch_size,
        device=device,
        verbose=False,
    )
    return [
        {
            "bertscore_p": float(p),
            "bertscore_r": float(r),
            "bertscore_f1": float(f),
        }
        for p, r, f in zip(precision, recall, f1)
    ]


def evaluate_summaries(
    predictions: pd.DataFrame,
    references: pd.DataFrame,
    *,
    split: str | None = None,
    include_bertscore: bool = True,
    bertscore_model: str = "distilbert-base-uncased",
    bertscore_batch_size: int = 8,
    device: str | None = None,
) -> pd.DataFrame:
    """Return per-case x granularity ROUGE and BERTScore metrics."""

    if "case_id" not in predictions.columns:
        raise ValueError("Predictions CSV must contain a `case_id` column.")
    if split is not None:
        if "split" not in predictions.columns:
            raise ValueError("Predictions CSV must contain `split` when --split is used.")
        predictions = predictions[predictions["split"] == split].copy()

    merged = predictions.merge(
        references[["case_id", "long_ref", "short_ref", "tiny_ref"]],
        on="case_id",
        how="inner",
        validate="many_to_one",
    )
    if merged.empty:
        raise ValueError("No predictions matched references by case_id.")

    all_rows: list[dict] = []
    for granularity in SUMMARY_GRANULARITIES:
        pred_col = f"{granularity}_pred"
        ref_col = f"{granularity}_ref"
        if pred_col not in merged.columns:
            logger.warning("Skipping %s: missing %s", granularity, pred_col)
            continue

        work = merged[["case_id", pred_col, ref_col]].copy()
        preds = [_safe_text(x) for x in work[pred_col].tolist()]
        refs = [_safe_text(x) for x in work[ref_col].tolist()]
        rouge = _rouge_rows(preds, refs)
        bert = (
            _bertscore_rows(
                preds,
                refs,
                model_type=bertscore_model,
                batch_size=bertscore_batch_size,
                device=device,
            )
            if include_bertscore
            else [{} for _ in preds]
        )

        for case_id, r_scores, b_scores in tqdm(
            zip(work["case_id"], rouge, bert),
            total=len(work),
            desc=f"metrics:{granularity}",
        ):
            all_rows.append(
                {
                    "case_id": case_id,
                    "granularity": granularity,
                    **r_scores,
                    **b_scores,
                }
            )

    return pd.DataFrame(all_rows)


def summarize_metric_table(metrics: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-case rows into one table per granularity."""

    metric_cols = [c for c in metrics.columns if c not in {"case_id", "granularity"}]
    return metrics.groupby("granularity", as_index=False)[metric_cols].mean(numeric_only=True)


def evaluate_classification(
    predictions: pd.DataFrame,
    *,
    label_names: list[str] | None = None,
) -> dict:
    """Compute classification metrics from a predictions DataFrame.

    Required columns:
      - ``y_true`` (int)
      - ``y_pred`` (int)

    Optional columns:
      - ``y_proba_<class>`` per class for AUC. Binary may instead supply
        ``y_proba_1`` (positive-class probability) only.

    Returns a dict with: accuracy, f1_macro, f1_weighted, auc_roc,
    confusion (nested list), per_class_report (sklearn dict).
    """
    from sklearn.metrics import (  # local import keeps top of file light
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        roc_auc_score,
    )

    for required in ("y_true", "y_pred"):
        if required not in predictions.columns:
            raise ValueError(f"Predictions CSV missing required column {required!r}.")

    y_true = predictions["y_true"].to_numpy()
    y_pred = predictions["y_pred"].to_numpy()
    n_classes = int(max(y_true.max(), y_pred.max()) + 1)
    labels = list(range(n_classes))
    target_names = label_names or [f"class_{i}" for i in labels]

    auc: float | None = None
    proba_cols = [c for c in predictions.columns if c.startswith("y_proba_")]
    if proba_cols:
        try:
            if n_classes == 2 and "y_proba_1" in proba_cols:
                auc = float(roc_auc_score(y_true, predictions["y_proba_1"]))
            elif len(proba_cols) == n_classes:
                proba_cols_sorted = sorted(proba_cols, key=lambda c: int(c.split("_")[-1]))
                proba = predictions[proba_cols_sorted].to_numpy()
                auc = float(roc_auc_score(y_true, proba, multi_class="ovr", average="macro"))
        except ValueError as exc:
            logger.warning("AUC computation skipped: %s", exc)

    report = classification_report(
        y_true, y_pred, labels=labels, target_names=target_names,
        output_dict=True, zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "auc_roc": auc,
        "confusion": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "per_class_report": report,
    }


def save_confusion_matrix_png(
    confusion: list[list[int]] | pd.DataFrame,
    *,
    label_names: list[str],
    output_path: str | Path,
    title: str = "Confusion Matrix",
) -> Path:
    """Render a confusion matrix PNG from classification metrics output."""
    import matplotlib.pyplot as plt
    import numpy as np

    cm = np.asarray(confusion)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 4) if len(label_names) == 2 else (7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(label_names)))
    ax.set_yticks(range(len(label_names)))
    ax.set_xticklabels(label_names, rotation=30, ha="right")
    ax.set_yticklabels(label_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    vmax = cm.max() if cm.size else 1
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > vmax / 2 else "black"
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color=color, fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    logger.info("Wrote confusion matrix PNG -> %s", output)
    return output


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate summarization or classification outputs.")
    parser.add_argument("--task", choices=["summarization", "classification"], default="summarization")
    parser.add_argument("--predictions", required=True, help="CSV with case_id and *_pred columns.")
    parser.add_argument("--references", default="data/multilexsum_clean.parquet")
    parser.add_argument("--output", default=str(RESULTS_DIR / "summary_eval.csv"))
    parser.add_argument("--summary-output", default=str(RESULTS_DIR / "summary_eval_by_granularity.csv"))
    parser.add_argument("--split", choices=["train", "val", "test"], help="Optional split filter.")
    parser.add_argument("--skip-bertscore", action="store_true")
    parser.add_argument("--bertscore-model", default="distilbert-base-uncased")
    parser.add_argument("--bertscore-batch-size", type=int, default=8)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--label-names", nargs="+",
                        help="Class names (classification only) — used in per-class report.")
    parser.add_argument(
        "--confusion-output",
        help="Optional PNG path for the classification confusion matrix.",
    )
    return parser


def _run_classification(args: argparse.Namespace) -> None:
    predictions = pd.read_csv(args.predictions)
    result = evaluate_classification(predictions, label_names=args.label_names)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    json_path = output.with_suffix(".json")
    with json_path.open("w") as fh:
        json.dump(result, fh, indent=2)
    logger.info("Wrote classification metrics → %s", json_path)
    if args.confusion_output and args.label_names:
        save_confusion_matrix_png(
            result["confusion"],
            label_names=args.label_names,
            output_path=args.confusion_output,
            title=f"Confusion Matrix · {Path(args.predictions).stem}",
        )


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.task == "classification":
        _run_classification(args)
        return

    predictions = pd.read_csv(args.predictions)
    references = pd.read_parquet(args.references)
    metrics = evaluate_summaries(
        predictions,
        references,
        split=args.split,
        include_bertscore=not args.skip_bertscore,
        bertscore_model=args.bertscore_model,
        bertscore_batch_size=args.bertscore_batch_size,
        device=args.device,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output, index=False)
    logger.info("Wrote per-case summary evaluation -> %s", output)

    summary_output = Path(args.summary_output)
    summarize_metric_table(metrics).to_csv(summary_output, index=False)
    logger.info("Wrote aggregate summary evaluation -> %s", summary_output)


if __name__ == "__main__":
    main()
