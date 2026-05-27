"""SHAP-based top-token explanations for the linear models (§3.4).

Why only Logistic Regression and only top-token bar charts?
-----------------------------------------------------------
- README §3 deliverables list ``lr_shap_classaction.png`` and
  ``lr_shap_casetype.png`` — both are LR.
- For a sparse linear model on TF-IDF features, SHAP's ``LinearExplainer``
  reduces to the model's own (mean-centred) coefficients × feature value,
  so we get exact, fast attributions without sampling. This is the
  right tool — kernel/tree explainers would be slow and stochastic here.
- Bar chart of the top-k positive- and negative-impact tokens is what
  slide 8 needs: a story like "the words 'class' and 'representative'
  push the model toward 'class action sought'".

For the multi-class case-type plot we tile per-class bar charts so each
of the 5 case-type groups gets its own most-distinctive vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.utils import RESULTS_DIR, get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class TokenImpact:
    """One row of the bar chart: a token + its average signed SHAP value."""

    token: str
    mean_shap: float


def _ensure_linear(pipe: Pipeline) -> tuple[TfidfVectorizer, LogisticRegression]:
    """Defensive unpack — only LR + TF-IDF pipelines are supported here."""
    vectorizer = pipe.named_steps.get("tfidf")
    classifier = pipe.named_steps.get("clf")
    if not isinstance(vectorizer, TfidfVectorizer):
        raise TypeError("explain.py expects a 'tfidf' TfidfVectorizer step.")
    if not isinstance(classifier, LogisticRegression):
        raise TypeError(
            f"SHAP top-token explanation is implemented for LogisticRegression "
            f"only (got {type(classifier).__name__})."
        )
    return vectorizer, classifier


def _shap_values(
    vectorizer: TfidfVectorizer,
    classifier: LogisticRegression,
    X_texts: np.ndarray,
    *,
    background_size: int = 100,
) -> tuple[np.ndarray, list[str]]:
    """Return (shap_values, vocabulary) where shap_values has shape (n_samples, n_features).

    For multi-class LR with K classes, shap returns a list of K arrays; we
    stack into shape (K, n_samples, n_features) so downstream code can
    pick a class index.
    """
    import shap  # local import: optional dependency

    X = vectorizer.transform(X_texts)
    background_size = min(background_size, X.shape[0])
    rng = np.random.default_rng(42)
    bg_idx = rng.choice(X.shape[0], size=background_size, replace=False)
    background = X[bg_idx]

    explainer = shap.LinearExplainer(classifier, background)
    values = explainer.shap_values(X)
    vocab = vectorizer.get_feature_names_out().tolist()

    # Normalize shape to (K, n_samples, n_features) across SHAP versions:
    #   - old (<=0.42)        : list of K arrays, each (N, F)
    #   - new multi-class     : ndarray (N, F, K)
    #   - binary              : ndarray (N, F)
    if isinstance(values, list):
        values = np.stack(values, axis=0)
    elif values.ndim == 3:
        values = np.transpose(values, (2, 0, 1))
    else:
        values = values[np.newaxis, ...]
    return values, vocab


def top_tokens_per_class(
    shap_values: np.ndarray,
    vocab: list[str],
    *,
    top_k: int = 15,
) -> list[list[TokenImpact]]:
    """Return, for each class, the top-k tokens by absolute mean SHAP impact.

    Each token's score keeps its sign so the bar chart shows direction:
    positive bars push *toward* the class, negative bars push *away*.
    """
    out: list[list[TokenImpact]] = []
    n_classes = shap_values.shape[0]
    for cls in range(n_classes):
        mean_shap = shap_values[cls].mean(axis=0)
        # Some sklearn versions emit a 1-D ndarray, others a matrix-like; flatten safely.
        mean_shap = np.asarray(mean_shap).ravel()
        idx = np.argsort(-np.abs(mean_shap))[:top_k]
        rows = [TokenImpact(token=vocab[i], mean_shap=float(mean_shap[i])) for i in idx]
        out.append(rows)
    return out


def _plot_binary(
    tokens: list[TokenImpact],
    *,
    positive_label: str,
    negative_label: str,
    title: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    tokens_sorted = sorted(tokens, key=lambda t: t.mean_shap)
    labels = [t.token for t in tokens_sorted]
    values = [t.mean_shap for t in tokens_sorted]
    colors = ["#1f77b4" if v >= 0 else "#d62728" for v in values]
    ax.barh(labels, values, color=colors)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel(f"← pushes toward '{negative_label}'    |    pushes toward '{positive_label}' →")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved SHAP plot → %s", output_path)


def _plot_multiclass(
    tokens_per_class: list[list[TokenImpact]],
    label_names: list[str],
    *,
    title: str,
    output_path: Path,
) -> None:
    """Tile one mini bar chart per class so each row of slide 9 has its own vocabulary."""
    n_classes = len(tokens_per_class)
    n_cols = min(3, n_classes)
    n_rows = (n_classes + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = np.atleast_1d(axes).ravel()
    for idx, (cls_name, tokens) in enumerate(zip(label_names, tokens_per_class)):
        tokens_sorted = sorted(tokens, key=lambda t: t.mean_shap)
        labels = [t.token for t in tokens_sorted]
        values = [t.mean_shap for t in tokens_sorted]
        colors = ["#1f77b4" if v >= 0 else "#d62728" for v in values]
        axes[idx].barh(labels, values, color=colors)
        axes[idx].axvline(0, color="black", linewidth=0.5)
        axes[idx].set_title(cls_name, fontsize=10)
    # Hide any leftover panels in the grid.
    for k in range(len(label_names), len(axes)):
        axes[k].set_visible(False)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved SHAP plot → %s", output_path)


def explain_lr(
    pipe: Pipeline,
    X_test: np.ndarray,
    label_names: list[str],
    *,
    task: str,
    top_k: int = 15,
    output_dir: Path | None = None,
) -> Path:
    """Top entry point: pickled LR pipeline → SHAP PNG under results/."""
    vectorizer, classifier = _ensure_linear(pipe)
    shap_values, vocab = _shap_values(vectorizer, classifier, X_test)
    tokens_per_class = top_tokens_per_class(shap_values, vocab, top_k=top_k)

    out_dir = output_dir or RESULTS_DIR
    tag = "classaction" if task == "class_action" else "casetype"
    output_path = out_dir / f"lr_shap_{tag}.png"

    if len(label_names) == 2:
        # Binary LR has 1 row of shap values, signed toward class index 1.
        _plot_binary(
            tokens_per_class[-1],
            positive_label=label_names[1],
            negative_label=label_names[0],
            title=f"SHAP top tokens — LR · {task}",
            output_path=output_path,
        )
    else:
        _plot_multiclass(
            tokens_per_class,
            label_names,
            title=f"SHAP top tokens — LR · {task}",
            output_path=output_path,
        )
    return output_path


def build_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(description="Generate SHAP top-token plot for an LR pipeline.")
    parser.add_argument("--task", required=True, choices=["class_action", "case_type"])
    parser.add_argument("--pipeline", help="Path to pickled LR pipeline. Defaults to models/lr_{tag}.pkl.")
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--text-source", default="long_ref",
                        choices=["long_ref", "long_pred", "source_text"])
    return parser


def main() -> None:
    import pickle

    from src.classify.data import load_classification_data
    from src.utils import MODELS_DIR

    args = build_arg_parser().parse_args()
    tag = "classaction" if args.task == "class_action" else "casetype"
    pipeline_path = Path(args.pipeline) if args.pipeline else MODELS_DIR / f"lr_{tag}.pkl"
    with pipeline_path.open("rb") as fh:
        pipe = pickle.load(fh)

    data = load_classification_data(task=args.task, text_source=args.text_source)
    explain_lr(pipe, data.X_test, data.label_names, task=args.task, top_k=args.top_k)


if __name__ == "__main__":
    main()
