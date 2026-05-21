"""Clean Multi-LexSum source texts and produce the canonical parquet.

Pipeline:
  raw parquet  →  apply regex cleaning  →  apply case_type grouping
              →  drop rows missing references/metadata
              →  write `data/multilexsum_clean.parquet`

This canonical parquet is the *only* data file the rest of the project
reads. Teammates downstream should NOT re-clean from raw.

Usage
-----
From command line:
    python -m src.cleaning            # build clean parquet (no-op if exists)
    python -m src.cleaning --force    # rebuild even if cached

From Python:
    from src.cleaning import build_clean_dataframe, clean_text
    df_clean = build_clean_dataframe(force=False)
    cleaned_str = clean_text(raw_string)
"""

from __future__ import annotations

import argparse
import re
from typing import Optional

import pandas as pd

from src.case_type_grouping import group_case_type
from src.data import SOURCE_DOC_SEP, load_and_cache_raw
from src.utils import DATA_DIR, get_logger

logger = get_logger(__name__)

CLEAN_CACHE_PATH = DATA_DIR / "multilexsum_clean.parquet"

# ---------------------------------------------------------------------------
# Regex patterns for legal-document noise
# ---------------------------------------------------------------------------
# Inter-document separator we ourselves inserted in src.data
_DOC_SEP_RE = re.compile(re.escape(SOURCE_DOC_SEP))

# Pagination markers: "[Page 3 of 27]", "Page 3 of 27", "page 1"
_PAGE_MARKER_RE = re.compile(
    r"\[?\s*page\s+\d+(?:\s+of\s+\d+)?\s*\]?", re.IGNORECASE
)

# Legal reporter citations: e.g. "123 F.3d 456", "519 U.S. 357", "789 So.2d 123"
_CITATION_RE = re.compile(
    r"\b\d{1,4}\s+[A-Z][a-zA-Z\.]{1,8}\.?\s?\d?[a-z]?\s+\d{1,4}\b"
)

# U.S. Code references: "42 U.S.C. § 1983"
_USC_RE = re.compile(
    r"\b\d+\s+U\.?\s?S\.?\s?C\.?\s*§+\s*\d+[a-zA-Z\-]*"
)

# Code of Federal Regulations: "29 C.F.R. § 1604.11"
_CFR_RE = re.compile(
    r"\b\d+\s+C\.?\s?F\.?\s?R\.?\s*§+\s*\d+(?:\.\d+)*"
)

# Footnote markers: "[1]", "[fn 2]", "* * *"
_FOOTNOTE_RE = re.compile(r"\[\s*(?:fn\s*)?\d{1,3}\s*\]", re.IGNORECASE)

# URLs and emails (rare but appear in some cases)
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")

# Repeated dashes/underscores used as section dividers
_DIVIDER_RE = re.compile(r"[-_*=]{3,}")

# Excess whitespace
_WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: Optional[str]) -> str:
    """Apply the full cleaning pipeline to a single source string.

    Order matters: remove structural noise FIRST (separators, pages),
    then patterned content (citations), THEN whitespace normalize LAST.
    """
    if not text:
        return ""
    t = text

    # Structural noise
    t = _DOC_SEP_RE.sub(" ", t)
    t = _PAGE_MARKER_RE.sub(" ", t)
    t = _FOOTNOTE_RE.sub(" ", t)
    t = _DIVIDER_RE.sub(" ", t)

    # URLs & emails
    t = _URL_RE.sub(" ", t)
    t = _EMAIL_RE.sub(" ", t)

    # Legal citations (order: longer/more-specific patterns first)
    t = _USC_RE.sub(" ", t)
    t = _CFR_RE.sub(" ", t)
    t = _CITATION_RE.sub(" ", t)

    # Whitespace normalize last
    t = _WHITESPACE_RE.sub(" ", t)
    return t.strip()


def build_clean_dataframe(force: bool = False) -> pd.DataFrame:
    """Build the canonical cleaned parquet. Returns the DataFrame.

    Steps:
      1. Load raw parquet (downloads if missing)
      2. Clean `source_text` via regex pipeline
      3. Add length columns (chars + tokens)
      4. Add `case_type_grouped` via mapping
      5. Drop rows missing any reference summary or classification target
      6. Drop near-empty cases (< 100 chars after cleaning)
      7. Write parquet
    """
    if CLEAN_CACHE_PATH.exists() and not force:
        logger.info(f"Loading cached clean data from {CLEAN_CACHE_PATH}")
        return pd.read_parquet(CLEAN_CACHE_PATH)

    raw = load_and_cache_raw()
    logger.info(f"Starting cleaning on {len(raw)} rows...")

    df = raw.copy()

    # Step 1: clean text
    df["source_text"] = df["source_text"].apply(clean_text)

    # Step 2: add length columns
    df["source_n_chars"] = df["source_text"].str.len()
    df["source_n_tokens"] = df["source_text"].str.split().str.len().fillna(0).astype(int)

    # Step 3: group case_type
    df["case_type_grouped"] = df["case_type_raw"].apply(group_case_type)

    # Step 4: drop incomplete rows
    before = len(df)
    df = df.dropna(subset=[
        "long_ref", "short_ref", "tiny_ref",
        "class_action_sought", "case_type_raw",
    ])
    after_na = len(df)
    df = df[df["source_n_chars"] >= 100]
    after_len = len(df)

    logger.info(
        f"Dropped {before - after_na} rows missing summaries/metadata; "
        f"dropped {after_na - after_len} near-empty cases. "
        f"Final: {after_len} rows."
    )

    df = df.reset_index(drop=True)

    # Sanity check: warn if too many rows fell into "Other" case-type group
    other_pct = (df["case_type_grouped"] == "Other").mean() * 100
    if other_pct > 5.0:
        logger.warning(
            f"{other_pct:.1f}% of cases routed to 'Other' — "
            f"check `src/case_type_grouping.py` for missing raw labels."
        )
    else:
        logger.info(f"'Other' group is {other_pct:.1f}% (good).")

    df.to_parquet(CLEAN_CACHE_PATH, index=False)
    logger.info(f"Wrote canonical clean parquet → {CLEAN_CACHE_PATH}")
    logger.info(f"Split sizes: {df['split'].value_counts().to_dict()}")
    logger.info(
        f"class_action_sought distribution: "
        f"{df['class_action_sought'].value_counts(dropna=False).to_dict()}"
    )
    logger.info(
        f"case_type_grouped distribution: "
        f"{df['case_type_grouped'].value_counts().to_dict()}"
    )

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Build cleaned Multi-LexSum parquet.")
    parser.add_argument("--force", action="store_true",
                        help="Rebuild even if cached parquet exists.")
    args = parser.parse_args()

    df = build_clean_dataframe(force=args.force)
    print(f"\nFinal cleaned shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")


if __name__ == "__main__":
    main()
