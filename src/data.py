"""Download and cache the Multi-LexSum dataset as a flat parquet file.

The HuggingFace dataset returns nested records with optional fields.
This module:
  1. Downloads the dataset (cached by HF in ~/.cache/huggingface/)
  2. Flattens nested fields into a tabular schema
  3. Writes a single parquet file under `data/multilexsum_raw.parquet`

After this step, all downstream modules read the parquet — no one calls
`load_dataset` again. This is the single source of truth.

Usage
-----
From command line:
    python -m src.data            # download + cache (no-op if cached)
    python -m src.data --force    # re-download even if cache exists

From Python:
    from src.data import load_and_cache_raw
    df = load_and_cache_raw()
"""

from __future__ import annotations

import argparse
from typing import Any

import pandas as pd

from src.utils import DATA_DIR, get_logger

logger = get_logger(__name__)

DATASET_NAME = "allenai/multi_lexsum"
DATASET_VERSION = "v20230518"
RAW_CACHE_PATH = DATA_DIR / "multilexsum_raw.parquet"

# Separator used to join the multiple source docs of a single case.
# Cleaning step strips this back out.
SOURCE_DOC_SEP = "\n\n---DOC---\n\n"


def _safe_get(d: Any, key: str, default: Any = None) -> Any:
    """Defensive getter: dict-style if dict, attribute-style otherwise, else default."""
    if d is None:
        return default
    if isinstance(d, dict):
        return d.get(key, default)
    return getattr(d, key, default)


def _flatten_record(record: dict) -> dict:
    """Flatten one HuggingFace record into a flat dict.

    Multi-LexSum schema (v20230518):
      - id: str
      - sources: List[str]                 (full case docs, can be many)
      - summary/long, summary/short, summary/tiny: Optional[str]
      - case_metadata: dict with case_type, class_action_sought, ...
    """
    metadata = record.get("case_metadata") or {}
    sources_list = record.get("sources") or []

    # Join all source docs of this case with our separator
    source_text = SOURCE_DOC_SEP.join(s for s in sources_list if s)

    return {
        "case_id": record.get("id"),
        "source_text": source_text,
        "n_source_docs": len(sources_list),
        # Reference summaries (the "ground truth" for the summarization task)
        "long_ref": record.get("summary/long"),
        "short_ref": record.get("summary/short"),
        "tiny_ref": record.get("summary/tiny"),
        # Classification targets
        "class_action_sought": _safe_get(metadata, "class_action_sought"),
        "case_type_raw": _safe_get(metadata, "case_type"),
        # Useful auxiliary metadata
        "filing_date": _safe_get(metadata, "filing_date"),
        "court": _safe_get(metadata, "court"),
        "state": _safe_get(metadata, "state"),
    }


def load_and_cache_raw(force: bool = False) -> pd.DataFrame:
    """Download Multi-LexSum and cache as parquet. Returns the DataFrame.

    Parameters
    ----------
    force : bool
        If True, re-download even if cached parquet exists.
    """
    if RAW_CACHE_PATH.exists() and not force:
        logger.info(f"Loading cached raw data from {RAW_CACHE_PATH}")
        return pd.read_parquet(RAW_CACHE_PATH)

    logger.info(f"Downloading {DATASET_NAME} ({DATASET_VERSION})...")
    logger.info("First run will take several minutes (~2 GB to download).")

    # Lazy import: don't require `datasets` for downstream modules that
    # already have the parquet cached.
    from datasets import load_dataset

    ds = load_dataset(DATASET_NAME, name=DATASET_VERSION, trust_remote_code=True)

    frames = []
    for hf_split, our_split in [("train", "train"), ("validation", "val"), ("test", "test")]:
        if hf_split not in ds:
            logger.warning(f"Split '{hf_split}' not present in dataset; skipping.")
            continue
        records = [_flatten_record(dict(r)) for r in ds[hf_split]]
        df = pd.DataFrame(records)
        df["split"] = our_split
        frames.append(df)
        logger.info(f"  {hf_split:<10}: {len(df):>5} records")

    if not frames:
        raise RuntimeError("No splits loaded — check dataset name/version.")

    raw = pd.concat(frames, ignore_index=True)

    # Coerce class_action_sought to a clean bool when possible.
    # Multi-LexSum stores this as a string ("Yes"/"No") or bool depending on version.
    raw["class_action_sought"] = raw["class_action_sought"].map(_coerce_bool)

    raw.to_parquet(RAW_CACHE_PATH, index=False)
    logger.info(f"Cached raw parquet → {RAW_CACHE_PATH}")
    logger.info(f"Total rows: {len(raw)} | columns: {raw.shape[1]}")
    return raw


def _coerce_bool(v: Any) -> Any:
    """Convert various truthy/falsy encodings to a Python bool. Returns None if unknown."""
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v) if v in (0, 1) else None
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("yes", "true", "y", "1"):
            return True
        if s in ("no", "false", "n", "0"):
            return False
    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Download/cache Multi-LexSum.")
    parser.add_argument(
        "--download", action="store_true",
        help="Compatibility alias for the README quick-start; download/cache if needed.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download and re-cache even if parquet exists.",
    )
    parser.add_argument(
        "--sample", action="store_true",
        help="Print the first 3 rows after loading.",
    )
    args = parser.parse_args()

    df = load_and_cache_raw(force=args.force)

    print(f"\nShape: {df.shape}")
    print(f"Splits: {df['split'].value_counts().to_dict()}")
    print(f"Columns: {df.columns.tolist()}")

    if args.sample:
        with pd.option_context("display.max_colwidth", 200):
            print("\nFirst 3 rows (truncated):")
            print(df[["case_id", "n_source_docs", "case_type_raw",
                      "class_action_sought", "split"]].head(3))


if __name__ == "__main__":
    main()
