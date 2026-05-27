"""Classical models for §3: Naive Bayes + Logistic Regression.

Each ``train_*`` function returns a fitted ``sklearn.pipeline.Pipeline``
containing the TF-IDF vectorizer plus the classifier, so the saved pickle
is self-contained — load it and call ``.predict`` on raw strings.

The ``evaluate_classifier`` helper computes the metrics that go into
``results/classification_metrics.csv`` and onto slides 8–9.

Why these specific variants?
----------------------------
- ``ComplementNB`` (instead of MultinomialNB): much more stable on the
  multi-class case-type task where the smallest class is 8× smaller than
  the largest. Drop-in API.
- ``LogisticRegression(class_weight='balanced')``: re-weights samples by
  inverse class frequency, the same trick the CRLC labelling rationale
  in ``docs/case_type_grouping.md`` calls out for the 8:1 imbalance.
  ``saga`` solver handles both L2 and the high-dimensional TF-IDF input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline

from src.features import build_tfidf_vectorizer
from src.utils import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ClassifierMetrics:
    """Lightweight metrics container — converted to a CSV row by ``train.py``."""

    accuracy: float
    f1_macro: float
    f1_weighted: float
    auc_roc: float | None
    per_class_report: dict[str, Any]
    confusion: np.ndarray


def _build_pipeline(classifier) -> Pipeline:
    """TF-IDF + classifier, sharing one vocab end-to-end."""
    return Pipeline(
        steps=[
            ("tfidf", build_tfidf_vectorizer()),
            ("clf", classifier),
        ]
    )


def train_naive_bayes(X_train, y_train) -> Pipeline:
    """ComplementNB — robust default for sparse TF-IDF + imbalanced classes."""
    pipe = _build_pipeline(ComplementNB(alpha=0.3))
    pipe.fit(X_train, y_train)
    return pipe


def train_logistic_regression(X_train, y_train) -> Pipeline:
    """L2 logistic regression with class balancing — strong sparse-text baseline."""
    pipe = _build_pipeline(
        LogisticRegression(
            solver="saga",
            penalty="l2",
            C=1.0,
            class_weight="balanced",
            max_iter=2000,
            n_jobs=-1,
            random_state=42,
        )
    )
    pipe.fit(X_train, y_train)
    return pipe


def _compute_auc(pipe: Pipeline, X, y_true, n_classes: int) -> float | None:
    """ROC-AUC works for binary directly; for multi-class use one-vs-rest macro.

    Returns ``None`` if a class is missing from the eval split, which would
    crash ``roc_auc_score`` — better to leave the cell blank in the CSV.
    """
    if not hasattr(pipe, "predict_proba"):
        return None
    try:
        proba = pipe.predict_proba(X)
    except Exception as exc:  # pragma: no cover
        logger.warning("predict_proba unavailable: %s", exc)
        return None

    unique = set(np.unique(y_true).tolist())
    if n_classes == 2:
        if unique != {0, 1}:
            return None
        return float(roc_auc_score(y_true, proba[:, 1]))

    # Multi-class: need every class represented at least once
    if len(unique) < n_classes:
        return None
    return float(roc_auc_score(y_true, proba, multi_class="ovr", average="macro"))


def evaluate_classifier(
    pipe: Pipeline,
    X,
    y_true,
    label_names: list[str],
) -> ClassifierMetrics:
    """Compute the metric bundle ``train.py`` writes to CSV."""
    y_pred = pipe.predict(X)
    n_classes = len(label_names)

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(n_classes)),
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )

    return ClassifierMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        f1_macro=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        f1_weighted=float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        auc_roc=_compute_auc(pipe, X, y_true, n_classes),
        per_class_report=report,
        confusion=confusion_matrix(y_true, y_pred, labels=list(range(n_classes))),
    )
