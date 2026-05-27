"""Unified data loader for §3 classification.

This is the *single* place that knows how to translate
`(task, text_source)` into `(X_train, y_train, ...)` arrays. Every
classifier — classical, LSTM, or BERT — calls `load_classification_data`
so we never accidentally evaluate two models on different splits.

Tasks
-----
- ``class_action``  : binary, target column ``class_action_sought`` (bool)
- ``case_type``     : multi-class (5 groups), target column ``case_type_grouped``

Text sources
------------
- ``long_ref``  : the human-written long reference summary (W7 default)
- ``long_pred`` : Yujun's BART-generated long summary
                  (read from ``results/abstractive_summaries.csv``).
                  Used in W8 for the end-to-end "summarize-then-classify"
                  story.
- ``source_text``: full cleaned source text. Not recommended for transformer
                   models (>>512 tokens) but useful as an extra-strong
                   classical baseline.

Usage
-----
    from src.classify.data import load_classification_data
    data = load_classification_data(task="class_action", text_source="long_ref")
    print(data.X_train.shape, data.label_names)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from src.case_type_grouping import list_groups
from src.utils import DATA_DIR, PROJECT_ROOT, get_logger

logger = get_logger(__name__)

CLEAN_PARQUET = DATA_DIR / "multilexsum_clean.parquet"
ABSTRACTIVE_CSV = PROJECT_ROOT / "results" / "abstractive_summaries.csv"

Task = Literal["class_action", "case_type"]
TextSource = Literal["long_ref", "long_pred", "source_text"]

TASK_TARGETS: dict[str, str] = {
    "class_action": "class_action_sought",
    "case_type": "case_type_grouped",
}


@dataclass(frozen=True)
class ClassificationData:
    """Container for one task × one text-source slice of the dataset.

    Numpy/object arrays are used so downstream sklearn / torch code can
    consume them uniformly. Labels are kept as integer-encoded arrays plus
    a ``label_names`` mapping so multi-class results stay interpretable.
    """

    task: str
    text_source: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    case_ids_train: np.ndarray
    case_ids_val: np.ndarray
    case_ids_test: np.ndarray
    label_names: list[str]

    @property
    def n_classes(self) -> int:
        return len(self.label_names)

    @property
    def is_binary(self) -> bool:
        return self.n_classes == 2


def _attach_long_pred(df: pd.DataFrame) -> pd.DataFrame:
    """Inner-join cleaned parquet with the abstractive predictions CSV.

    Rows missing a ``long_pred`` are dropped, which usually means Yujun
    has not yet run the abstractive pipeline on that case. The caller
    is warned how many cases were lost.
    """
    if not ABSTRACTIVE_CSV.exists():
        raise FileNotFoundError(
            f"Need {ABSTRACTIVE_CSV} for text_source='long_pred'. "
            "Run `python -m src.summarize.abstractive` first, or use "
            "text_source='long_ref'."
        )
    preds = pd.read_csv(ABSTRACTIVE_CSV)[["case_id", "long_pred"]]
    before = len(df)
    merged = df.merge(preds, on="case_id", how="inner", validate="one_to_one")
    after = len(merged)
    if after < before:
        logger.warning(
            "Dropped %d cases without a long_pred (kept %d).", before - after, after
        )
    return merged


def load_classification_data(
    task: Task,
    text_source: TextSource = "long_ref",
    parquet_path: str | None = None,
) -> ClassificationData:
    """Return train/val/test arrays for one task × one text source."""
    if task not in TASK_TARGETS:
        raise ValueError(f"Unknown task {task!r}; choices: {list(TASK_TARGETS)}")

    path = parquet_path or CLEAN_PARQUET
    df = pd.read_parquet(path)
    logger.info("Loaded %d rows from %s", len(df), path)

    if text_source == "long_pred":
        df = _attach_long_pred(df)

    if text_source not in df.columns:
        raise ValueError(
            f"text_source={text_source!r} not in dataframe columns {df.columns.tolist()!r}"
        )

    target_col = TASK_TARGETS[task]
    df = df.dropna(subset=[text_source, target_col, "split"]).reset_index(drop=True)

    # Build label encoding
    if task == "class_action":
        # bool -> int(0/1) so downstream models always see numeric labels
        label_names = ["No", "Yes"]
        df["_label"] = df[target_col].astype(bool).astype(int)
    else:  # case_type — fix the label order so different models share it
        label_names = list_groups()
        name_to_idx = {name: i for i, name in enumerate(label_names)}
        df = df[df[target_col].isin(label_names)].reset_index(drop=True)
        df["_label"] = df[target_col].map(name_to_idx).astype(int)

    def _slice(split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        sub = df[df["split"] == split]
        return (
            sub[text_source].to_numpy(),
            sub["_label"].to_numpy(),
            sub["case_id"].to_numpy(),
        )

    X_train, y_train, ids_train = _slice("train")
    X_val, y_val, ids_val = _slice("val")
    X_test, y_test, ids_test = _slice("test")

    logger.info(
        "task=%s text_source=%s | train=%d val=%d test=%d | classes=%s",
        task, text_source, len(X_train), len(X_val), len(X_test), label_names,
    )

    return ClassificationData(
        task=task,
        text_source=text_source,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        case_ids_train=ids_train,
        case_ids_val=ids_val,
        case_ids_test=ids_test,
        label_names=label_names,
    )


if __name__ == "__main__":
    # Smoke check both tasks once a clean parquet exists.
    for task in ("class_action", "case_type"):
        try:
            d = load_classification_data(task=task)
            print(f"\n=== {task} ===")
            print(f"  X_train: {d.X_train.shape}  y_train dist: {np.bincount(d.y_train)}")
            print(f"  X_val:   {d.X_val.shape}    y_val dist:   {np.bincount(d.y_val)}")
            print(f"  X_test:  {d.X_test.shape}   y_test dist:  {np.bincount(d.y_test)}")
            print(f"  label_names: {d.label_names}")
        except FileNotFoundError as e:
            print(f"\n{task}: {e}")
