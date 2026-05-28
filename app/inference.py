"""End-to-end inference entrypoint for the Multi-LexSum demo app.

Contract
--------
`predict(case_text: str) -> dict`

The returned payload is intentionally verbose so the Gradio UI, notebooks,
and future deployment wrappers can all consume the same structure.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.explain import explain_lr_model, export_bert_attention_example, load_label_names
from src.summarize.abstractive import AbstractiveSummarizer
from src.summarize.extractive import extractive_summarize
from src.summarize.tiny import DEFAULT_MODEL_DIR, TinyT5Summarizer
from src.utils import MODELS_DIR, RESULTS_DIR, get_logger

logger = get_logger(__name__)

APP_RESULTS_DIR = RESULTS_DIR / "app"
APP_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_ACTION_LABELS = ["No", "Yes"]
CASE_TYPE_LABELS = [
    "Criminal Justice",
    "Civil Rights & Equality",
    "Healthcare & Disability",
    "Immigration & Education",
    "Speech & Voting",
]
TASK_TAG = {
    "class_action": "classaction",
    "case_type": "casetype",
}


@dataclass(frozen=True)
class PredictionConfig:
    extractive_method: str = "lexrank"
    extractive_tokens: int = 3500
    abstractive_model_key: str = "bart-large-cnn"
    abstractive_model_name: str | None = None
    tiny_model_dir: Path = DEFAULT_MODEL_DIR
    classifier_text_source: str = "long_ref"


@dataclass(frozen=True)
class ResolvedArtifact:
    task: str
    model_kind: str
    path: Path


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=float)
    logits = logits - logits.max()
    exp = np.exp(logits)
    return exp / exp.sum()


def _predict_with_bert(model_dir: Path, text: str) -> tuple[str, float, list[float], list[str]]:
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise ImportError("Install transformers and torch before running BERT inference.") from exc

    if not model_dir.exists():
        raise FileNotFoundError(
            f"Missing BERT model artifact: {model_dir}. "
            "Train it with `python -m src.classify.train --model bert ...` first."
        )

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    label_names = load_label_names(model_dir)
    model.eval()

    inputs = tokenizer(text, truncation=True, padding="max_length", max_length=512, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits[0].cpu().numpy()
    proba = _softmax(logits)
    pred_idx = int(np.argmax(proba))
    return label_names[pred_idx], float(proba[pred_idx]), proba.tolist(), label_names


def _predict_with_lr(pipeline_path: Path, text: str, label_names: list[str]) -> tuple[str, float, list[float]]:
    from src.explain import load_pickle_model

    if not pipeline_path.exists():
        raise FileNotFoundError(
            f"Missing LR model artifact: {pipeline_path}. "
            "Train it with `python -m src.classify.train --model lr ...` first."
        )

    pipe = load_pickle_model(pipeline_path)
    if not hasattr(pipe, "predict_proba"):
        raise RuntimeError(f"Pipeline at {pipeline_path} does not expose predict_proba().")
    proba = pipe.predict_proba([text])[0]
    pred_idx = int(np.argmax(proba))
    return label_names[pred_idx], float(proba[pred_idx]), proba.tolist()


def _artifact_path(task: str, model_kind: str) -> Path:
    stem = f"{model_kind}_{TASK_TAG[task]}"
    suffix = ".pkl" if model_kind == "lr" else ""
    return MODELS_DIR / f"{stem}{suffix}"


def _resolve_artifact(task: str, model_kind: str) -> ResolvedArtifact | None:
    path = _artifact_path(task, model_kind)
    if path.exists():
        return ResolvedArtifact(task=task, model_kind=model_kind, path=path)
    return None


def _resolve_best_classifier(task: str) -> ResolvedArtifact:
    artifact = _resolve_artifact(task, "bert") or _resolve_artifact(task, "lr")
    if artifact is None:
        raise FileNotFoundError(
            "No trained classifier artifacts were found for task="
            f"{task!r}. Train one of: `python -m src.classify.train --task {task} --model lr`, "
            f"or `python -m src.classify.train --task {task} --model bert`."
        )
    return artifact


def _resolve_lr_explainer(task: str) -> ResolvedArtifact | None:
    return _resolve_artifact(task, "lr")


class DemoPredictor:
    """Lazy-loaded app inference wrapper."""

    def __init__(self, config: PredictionConfig | None = None) -> None:
        self.config = config or PredictionConfig()
        local_bart_dir = MODELS_DIR / "bart-large-cnn"
        self.abstractive_model_name = (
            self.config.abstractive_model_name
            or (str(local_bart_dir) if (local_bart_dir / "config.json").exists() else None)
        )
        self.classifiers = {
            "class_action": _resolve_best_classifier("class_action"),
            "case_type": _resolve_best_classifier("case_type"),
        }
        self.explainers = {
            task: _resolve_lr_explainer(task)
            for task in self.classifiers
        }
        self.abstractive = AbstractiveSummarizer(
            model_key=self.config.abstractive_model_key,
            model_name=self.abstractive_model_name,
        )
        self.tiny = TinyT5Summarizer(model_dir=self.config.tiny_model_dir)

    def summarize(self, case_text: str) -> dict[str, Any]:
        reduction = extractive_summarize(
            case_text,
            method=self.config.extractive_method,  # type: ignore[arg-type]
            target_tokens=self.config.extractive_tokens,
        )
        abstractive = self.abstractive.summarize(case_text, reduced_text=reduction.summary_text)
        tiny = self.tiny.summarize(abstractive.short)
        return {
            "long": abstractive.long,
            "short": abstractive.short,
            "tiny": tiny,
            "reduction": {
                "method": reduction.method,
                "original_tokens": reduction.original_tokens,
                "reduced_tokens": reduction.reduced_tokens,
                "selected_sentences": reduction.selected_sentences,
                "candidate_sentences": reduction.candidate_sentences,
                "reduction_pct": reduction.reduction_pct,
            },
        }

    def classify(self, long_summary: str) -> dict[str, Any]:
        def run_classifier(task: str, label_names: list[str]) -> tuple[tuple[str, float, list[float], list[str]], ResolvedArtifact]:
            artifact = self.classifiers[task]
            if artifact.model_kind == "bert":
                prediction = _predict_with_bert(artifact.path, long_summary)
            else:
                label, conf, probs = _predict_with_lr(artifact.path, long_summary, label_names)
                prediction = (label, conf, probs, label_names)
            return prediction, artifact

        class_action, class_action_artifact = run_classifier("class_action", CLASS_ACTION_LABELS)
        case_type, case_type_artifact = run_classifier("case_type", CASE_TYPE_LABELS)
        lr_class_action = None
        if self.explainers["class_action"] is not None:
            lr_class_action = _predict_with_lr(
                self.explainers["class_action"].path, long_summary, CLASS_ACTION_LABELS
            )
        lr_case_type = None
        if self.explainers["case_type"] is not None:
            lr_case_type = _predict_with_lr(
                self.explainers["case_type"].path, long_summary, CASE_TYPE_LABELS
            )
        return {
            "class_action": {
                "label": class_action[0],
                "confidence": class_action[1],
                "probabilities": dict(zip(class_action[3], class_action[2])),
                "model": class_action_artifact.path.name,
                "model_kind": class_action_artifact.model_kind,
                "text_source": self.config.classifier_text_source,
            },
            "case_type": {
                "label": case_type[0],
                "confidence": case_type[1],
                "probabilities": dict(zip(case_type[3], case_type[2])),
                "model": case_type_artifact.path.name,
                "model_kind": case_type_artifact.model_kind,
                "text_source": self.config.classifier_text_source,
            },
            "explainability_models": {
                "class_action_lr": (
                    self.explainers["class_action"].path.name if self.explainers["class_action"] else None
                ),
                "case_type_lr": (
                    self.explainers["case_type"].path.name if self.explainers["case_type"] else None
                ),
            },
            "lr_predictions": {
                "class_action": (
                    {
                        "label": lr_class_action[0],
                        "confidence": lr_class_action[1],
                        "probabilities": dict(zip(CLASS_ACTION_LABELS, lr_class_action[2])),
                        "text_source": self.config.classifier_text_source,
                    }
                    if lr_class_action is not None
                    else None
                ),
                "case_type": (
                    {
                        "label": lr_case_type[0],
                        "confidence": lr_case_type[1],
                        "probabilities": dict(zip(CASE_TYPE_LABELS, lr_case_type[2])),
                        "text_source": self.config.classifier_text_source,
                    }
                    if lr_case_type is not None
                    else None
                ),
            },
        }

    def explain(self, long_summary: str) -> dict[str, Any]:
        explain_dir = APP_RESULTS_DIR / "explanations"
        explain_dir.mkdir(parents=True, exist_ok=True)
        lr_class_action_path = None
        if self.explainers["class_action"] is not None:
            lr_class_action_path = explain_lr_model(
                pipeline_path=self.explainers["class_action"].path,
                texts=[long_summary],
                label_names=CLASS_ACTION_LABELS,
                task="class_action",
                output_dir=explain_dir,
            )
        lr_case_type_path = None
        if self.explainers["case_type"] is not None:
            lr_case_type_path = explain_lr_model(
                pipeline_path=self.explainers["case_type"].path,
                texts=[long_summary],
                label_names=CASE_TYPE_LABELS,
                task="case_type",
                output_dir=explain_dir,
            )
        bert_case_type = self.classifiers["case_type"]
        if bert_case_type.model_kind == "bert":
            bert_attention = export_bert_attention_example(
                model_dir=bert_case_type.path,
                text=long_summary,
                output_path=explain_dir / "bert_case_type_attention.png",
            )
        else:
            bert_attention = {
                "image_path": None,
                "tokens": [],
                "weights": [],
                "warning": "No BERT artifact available yet; attention view is disabled.",
            }
        return {
            "lr_class_action_plot": str(lr_class_action_path) if lr_class_action_path else None,
            "lr_case_type_plot": str(lr_case_type_path) if lr_case_type_path else None,
            "bert_attention": bert_attention,
        }

    def predict(self, case_text: str) -> dict[str, Any]:
        case_text = str(case_text or "").strip()
        if not case_text:
            raise ValueError("Please provide case text.")

        summaries = self.summarize(case_text)
        predictions = self.classify(summaries["long"])
        explainability = self.explain(summaries["long"])
        warnings = [
            "The checked-in classifiers were trained on `long_ref` reference summaries from §3.",
            "In the app, those same classifiers are applied to the generated long summary as a lightweight end-to-end demo approximation.",
            "BERT attention is illustrative, not causal proof.",
        ]
        for task in ("class_action", "case_type"):
            pred = predictions[task]
            if pred["model_kind"] != "bert":
                warnings.append(f"{task} is using LR fallback because no BERT artifact is available.")
            if self.explainers[task] is None:
                warnings.append(f"{task} has no LR explanation artifact available yet.")
        if explainability["bert_attention"]["image_path"] is None:
            warnings.append("BERT attention is unavailable until a trained BERT artifact is present.")
        return {
            "summaries": {
                "long": summaries["long"],
                "short": summaries["short"],
                "tiny": summaries["tiny"],
            },
            "predictions": {
                "class_action": predictions["class_action"],
                "case_type": predictions["case_type"],
            },
            "explainability": explainability,
            "metadata": {
                "input_text_chars": len(case_text),
                "models": {
                    "summarization": self.abstractive.model_name,
                    "summarization_key": self.config.abstractive_model_key,
                    "tiny": str(self.config.tiny_model_dir),
                    "class_action_classifier": predictions["class_action"]["model"],
                    "case_type_classifier": predictions["case_type"]["model"],
                    "classifier_text_source": self.config.classifier_text_source,
                },
                "reduction": summaries["reduction"],
                "warnings": warnings,
            },
        }


_PREDICTOR: DemoPredictor | None = None


def get_predictor() -> DemoPredictor:
    global _PREDICTOR
    if _PREDICTOR is None:
        _PREDICTOR = DemoPredictor()
    return _PREDICTOR


def predict(case_text: str) -> dict[str, Any]:
    """Stable public entrypoint used by the app and notebooks."""
    return get_predictor().predict(case_text)


def predict_json(case_text: str) -> str:
    """Convenience wrapper for CLI or debugging."""
    return json.dumps(predict(case_text), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the Multi-LexSum end-to-end predictor.")
    parser.add_argument("--text", help="Case text to analyze.")
    args = parser.parse_args()
    if not args.text:
        raise SystemExit("Provide --text.")
    print(predict_json(args.text))
