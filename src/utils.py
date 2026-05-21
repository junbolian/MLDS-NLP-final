"""Shared utilities: project paths, logging, deterministic seeding.

This module is the single source of truth for *where files live* in the
project. All other modules import path constants from here rather than
hard-coding paths, so the project is portable across machines/OSes.
"""

from __future__ import annotations

import logging
import os
import random
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# PROJECT_ROOT = parent of `src/`. Works on Windows, macOS, Linux.
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
MODELS_DIR: Path = PROJECT_ROOT / "models"
RESULTS_DIR: Path = PROJECT_ROOT / "results"
DOCS_DIR: Path = PROJECT_ROOT / "docs"
NOTEBOOKS_DIR: Path = PROJECT_ROOT / "notebooks"

# Ensure standard directories exist (idempotent)
for _d in (DATA_DIR, MODELS_DIR, RESULTS_DIR, DOCS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger with a consistent format across modules.

    Parameters
    ----------
    name : str
        Logger name, usually `__name__` from the calling module.
    level : int
        Logging level (default: INFO).
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int = 42) -> None:
    """Seed all RNGs we use (Python, NumPy, optional PyTorch/TF)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------
def human_bytes(n: int) -> str:
    """Format a byte count as KB/MB/GB."""
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


if __name__ == "__main__":
    log = get_logger("utils")
    log.info(f"PROJECT_ROOT = {PROJECT_ROOT}")
    log.info(f"DATA_DIR     = {DATA_DIR}")
    log.info(f"MODELS_DIR   = {MODELS_DIR}")
    log.info(f"RESULTS_DIR  = {RESULTS_DIR}")
