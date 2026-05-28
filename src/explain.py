"""Unified explainability helpers for the demo app and README §4.

This module intentionally sits above the task-specific `src.classify.explain`
implementation so the app can import one stable surface area for:
  * LR token-level explanations
  * lightweight BERT attention visualizations
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.classify.explain import explain_lr
from src.utils import MODELS_DIR, RESULTS_DIR, get_logger

logger = get_logger(__name__)


def load_pickle_model(path: str | Path):
    """Load a pickled sklearn pipeline."""
    with Path(path).open("rb") as fh:
        return pickle.load(fh)


def explain_lr_model(
    *,
    pipeline_path: str | Path,
    texts: list[str],
    label_names: list[str],
    task: str,
    top_k: int = 15,
    output_dir: str | Path | None = None,
) -> Path:
    """Render a SHAP explanation plot for the supplied texts."""
    pipe = load_pickle_model(pipeline_path)
    return explain_lr(
        pipe,
        np.asarray(texts, dtype=object),
        label_names,
        task=task,
        top_k=top_k,
        output_dir=Path(output_dir) if output_dir else None,
    )


def render_attention_heatmap(
    tokens: list[str],
    attention: np.ndarray,
    *,
    title: str,
    output_path: str | Path,
    max_tokens: int = 20,
) -> Path:
    """Persist a compact attention heatmap for one BERT example."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    usable_tokens = tokens[:max_tokens]
    usable_attention = np.asarray(attention, dtype=float)[: len(usable_tokens), : len(usable_tokens)]

    fig, ax = plt.subplots(figsize=(max(6, len(usable_tokens) * 0.45), max(5, len(usable_tokens) * 0.45)))
    im = ax.imshow(usable_attention, cmap="magma")
    ax.set_xticks(range(len(usable_tokens)))
    ax.set_yticks(range(len(usable_tokens)))
    ax.set_xticklabels(usable_tokens, rotation=60, ha="right", fontsize=8)
    ax.set_yticklabels(usable_tokens, fontsize=8)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    logger.info("Saved BERT attention heatmap → %s", output)
    return output


def export_bert_attention_example(
    *,
    model_dir: str | Path,
    text: str,
    output_path: str | Path,
    max_length: int = 128,
) -> dict[str, Any]:
    """Compute a representative attention map for one text and save it as an image."""
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise ImportError("Install transformers and torch for BERT attention export.") from exc

    model_path = Path(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, output_attentions=True)
    model.eval()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    with torch.no_grad():
        outputs = model(**inputs)

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    attentions = outputs.attentions
    if not attentions:
        raise RuntimeError("Model did not return attentions.")

    last_layer = attentions[-1][0].mean(dim=0).cpu().numpy()
    image_path = render_attention_heatmap(
        tokens,
        last_layer,
        title="BERT attention example",
        output_path=output_path,
    )
    return {
        "image_path": str(image_path),
        "tokens": tokens[:20],
    }


def load_label_names(model_dir: str | Path) -> list[str]:
    """Read label names for a saved BERT model directory."""
    path = Path(model_dir) / "label_names.json"
    return json.loads(path.read_text(encoding="utf-8"))


__all__ = [
    "RESULTS_DIR",
    "MODELS_DIR",
    "explain_lr_model",
    "export_bert_attention_example",
    "load_label_names",
    "load_pickle_model",
    "render_attention_heatmap",
]
