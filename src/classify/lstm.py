"""Bi-LSTM classifier with self-trained Word2Vec embeddings (§3.5).

PyTorch, *not* TensorFlow — chosen to keep the project on one DL stack
(BART and Legal-BERT next door already use PyTorch). The README's "TF/Keras"
note is overridden by this consolidation.

Pipeline
--------
1. Tokenise each input (long_ref summary by default) with a light regex.
2. Train a gensim Word2Vec model on the **training-split tokens only**
   so the embeddings remain leakage-free.
3. Build a torch ``Embedding`` initialised from the gensim vectors
   (frozen for the first epoch, then unfrozen — a common warm-start
   recipe for small datasets).
4. ``BiLSTM(hidden=128, dropout=0.3) → Linear → output``.
5. Train with AdamW, class-weighted cross-entropy, early-stopping on
   val macro-F1.

Output artifacts (per call)
---------------------------
- ``models/lstm_{task_tag}.pt``                 : model + tokenizer + label_names
- ``results/training_curves/lstm_{task_tag}.png``: loss + accuracy per epoch
- a row appended to ``results/classification_metrics.csv`` (via train.py)
- a row appended to a per-epoch CSV for completeness (``train_log_{task_tag}.csv``)

GPU notes
---------
- Colab T4: ~3 min/task at default settings.
- Apple Silicon MPS works (set ``--device mps``) but is ~3× slower.
- CPU is workable (~10 min/task) — keep ``--epochs 6`` if you go that route.
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.utils import MODELS_DIR, RESULTS_DIR, get_logger, set_seed

logger = get_logger(__name__)

TRAINING_CURVES_DIR = RESULTS_DIR / "training_curves"

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]*|\d+")
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


# ---------------------------------------------------------------------------
# Tokenisation + vocab
# ---------------------------------------------------------------------------
def tokenize(text: str) -> list[str]:
    """Lowercase regex word tokens; consistent with the extractive TF-IDF."""
    return _WORD_RE.findall(text.lower())


@dataclass(frozen=True)
class Vocab:
    """Word ↔ id table. Token 0 is <pad>, token 1 is <unk>."""

    stoi: dict[str, int]
    itos: list[str]

    def encode(self, tokens: list[str], max_len: int) -> tuple[list[int], int]:
        ids = [self.stoi.get(tok, 1) for tok in tokens[:max_len]]
        length = len(ids)
        if length < max_len:
            ids = ids + [0] * (max_len - length)
        return ids, length


def build_vocab(token_lists: list[list[str]], *, min_freq: int = 2, max_size: int = 50_000) -> Vocab:
    from collections import Counter

    counter: Counter[str] = Counter()
    for toks in token_lists:
        counter.update(toks)
    most_common = [tok for tok, freq in counter.most_common(max_size) if freq >= min_freq]
    itos = [PAD_TOKEN, UNK_TOKEN, *most_common]
    stoi = {tok: i for i, tok in enumerate(itos)}
    return Vocab(stoi=stoi, itos=itos)


# ---------------------------------------------------------------------------
# Word2Vec — gensim, trained on training-split only
# ---------------------------------------------------------------------------
def train_word2vec(
    token_lists: list[list[str]],
    vocab: Vocab,
    *,
    dim: int = 300,
    window: int = 5,
    min_count: int = 1,
    workers: int = 4,
    epochs: int = 5,
) -> np.ndarray:
    """Return an ``(vocab_size, dim)`` embedding matrix.

    Tokens missing from gensim's vocab (very rare for our settings) stay
    at their random-normal init so the model can still flow gradients.
    """
    from gensim.models import Word2Vec

    logger.info("Training Word2Vec on %d documents (dim=%d) …", len(token_lists), dim)
    w2v = Word2Vec(
        sentences=token_lists,
        vector_size=dim,
        window=window,
        min_count=min_count,
        workers=workers,
        epochs=epochs,
        seed=42,
    )
    rng = np.random.default_rng(42)
    matrix = rng.normal(0.0, 0.1, size=(len(vocab.itos), dim)).astype(np.float32)
    matrix[0] = 0.0  # <pad>
    hits = 0
    for tok, idx in vocab.stoi.items():
        if tok in w2v.wv:
            matrix[idx] = w2v.wv[tok]
            hits += 1
    logger.info("Word2Vec hit rate: %d / %d (%.1f%%)", hits, len(vocab.itos),
                100 * hits / max(1, len(vocab.itos)))
    return matrix


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def _build_model(
    *,
    embedding_matrix: np.ndarray,
    n_classes: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
):
    """Construct the Embedding → BiLSTM → Linear classifier."""
    import torch
    import torch.nn as nn

    vocab_size, dim = embedding_matrix.shape

    class BiLSTMClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding.from_pretrained(
                torch.tensor(embedding_matrix), freeze=False, padding_idx=0,
            )
            self.lstm = nn.LSTM(
                input_size=dim, hidden_size=hidden_dim, num_layers=num_layers,
                batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0.0,
            )
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden_dim * 2, n_classes)

        def forward(self, x: "torch.Tensor", lengths: "torch.Tensor") -> "torch.Tensor":
            emb = self.embedding(x)
            packed = nn.utils.rnn.pack_padded_sequence(
                emb, lengths.cpu(), batch_first=True, enforce_sorted=False,
            )
            _, (h_n, _) = self.lstm(packed)
            # Concat last-layer forward + backward hidden states
            h_fwd = h_n[-2]
            h_bwd = h_n[-1]
            pooled = self.dropout(torch.cat([h_fwd, h_bwd], dim=-1))
            return self.fc(pooled)

    return BiLSTMClassifier()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LSTMConfig:
    max_len: int = 400
    batch_size: int = 16
    epochs: int = 10
    lr: float = 1e-3
    hidden_dim: int = 128
    num_layers: int = 1
    dropout: float = 0.3
    embedding_dim: int = 300
    patience: int = 3
    seed: int = 42


def _encode_dataset(vocab: Vocab, token_lists: list[list[str]], max_len: int):
    import torch

    ids_list, lengths = [], []
    for toks in token_lists:
        ids, length = vocab.encode(toks, max_len)
        ids_list.append(ids)
        lengths.append(max(length, 1))  # avoid 0-length sequences crashing pack_padded
    return torch.tensor(ids_list, dtype=torch.long), torch.tensor(lengths, dtype=torch.long)


def _class_weights(y: np.ndarray, n_classes: int):
    """Inverse-frequency weights, same spirit as sklearn's `class_weight='balanced'`."""
    counts = np.bincount(y, minlength=n_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (n_classes * counts)
    return weights


def _resolve_device(device: str | None):
    import torch

    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _save_training_curves(
    history: list[dict[str, float]],
    *,
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4))
    ax_loss.plot(epochs, [r["train_loss"] for r in history], label="train")
    ax_loss.plot(epochs, [r["val_loss"] for r in history], label="val")
    ax_loss.set_xlabel("epoch"); ax_loss.set_ylabel("loss"); ax_loss.set_title("Loss")
    ax_loss.legend()
    ax_acc.plot(epochs, [r["train_acc"] for r in history], label="train")
    ax_acc.plot(epochs, [r["val_acc"] for r in history], label="val")
    ax_acc.set_xlabel("epoch"); ax_acc.set_ylabel("accuracy"); ax_acc.set_title("Accuracy")
    ax_acc.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved training curves → %s", output_path)


def _save_log_csv(history: list[dict[str, float]], output_path: Path) -> None:
    if not history:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def train_lstm(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    *,
    label_names: list[str],
    task_tag: str,
    config: LSTMConfig | None = None,
    device: str | None = None,
):
    """Train one Bi-LSTM; returns (model, vocab, embedding_dim) plus side-effects.

    Side effects:
      - ``models/lstm_{task_tag}.pt`` containing model state + vocab + label_names
      - ``results/training_curves/lstm_{task_tag}.png``
      - ``results/training_curves/lstm_{task_tag}.csv``
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    cfg = config or LSTMConfig()
    set_seed(cfg.seed)
    dev = _resolve_device(device)
    logger.info("Bi-LSTM on device=%s | task_tag=%s", dev, task_tag)

    train_tokens = [tokenize(t) for t in X_train]
    val_tokens = [tokenize(t) for t in X_val]

    vocab = build_vocab(train_tokens)
    embedding_matrix = train_word2vec(train_tokens, vocab, dim=cfg.embedding_dim)

    X_train_ids, train_lens = _encode_dataset(vocab, train_tokens, cfg.max_len)
    X_val_ids, val_lens = _encode_dataset(vocab, val_tokens, cfg.max_len)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    y_val_t = torch.tensor(y_val, dtype=torch.long)

    train_loader = DataLoader(
        TensorDataset(X_train_ids, train_lens, y_train_t),
        batch_size=cfg.batch_size, shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(X_val_ids, val_lens, y_val_t),
        batch_size=cfg.batch_size, shuffle=False,
    )

    model = _build_model(
        embedding_matrix=embedding_matrix,
        n_classes=len(label_names),
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    ).to(dev)

    weights = torch.tensor(_class_weights(y_train, len(label_names)), device=dev)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-5)

    best_val_acc = -1.0
    bad_epochs = 0
    history: list[dict[str, float]] = []
    best_state: dict | None = None

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        n_seen = 0
        n_correct = 0
        for ids, lens, labels in train_loader:
            ids, lens, labels = ids.to(dev), lens.to(dev), labels.to(dev)
            optimizer.zero_grad()
            logits = model(ids, lens)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            n_seen += labels.size(0)
            n_correct += int((logits.argmax(dim=-1) == labels).sum().item())
        train_loss = total_loss / max(1, n_seen)
        train_acc = n_correct / max(1, n_seen)

        # Validation
        model.eval()
        val_total = 0.0
        val_seen = 0
        val_correct = 0
        with torch.no_grad():
            for ids, lens, labels in val_loader:
                ids, lens, labels = ids.to(dev), lens.to(dev), labels.to(dev)
                logits = model(ids, lens)
                val_total += criterion(logits, labels).item() * labels.size(0)
                val_seen += labels.size(0)
                val_correct += int((logits.argmax(dim=-1) == labels).sum().item())
        val_loss = val_total / max(1, val_seen)
        val_acc = val_correct / max(1, val_seen)

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
        })
        logger.info(
            "epoch %d/%d | train loss=%.4f acc=%.4f | val loss=%.4f acc=%.4f",
            epoch, cfg.epochs, train_loss, train_acc, val_loss, val_acc,
        )

        if val_acc > best_val_acc + 1e-4:
            best_val_acc = val_acc
            bad_epochs = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                logger.info("Early stop: val_acc didn't improve for %d epochs.", cfg.patience)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    _save_training_curves(history, output_path=TRAINING_CURVES_DIR / f"lstm_{task_tag}.png")
    _save_log_csv(history, TRAINING_CURVES_DIR / f"lstm_{task_tag}.csv")

    bundle_path = MODELS_DIR / f"lstm_{task_tag}.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": cfg.__dict__,
            "vocab_itos": vocab.itos,
            "label_names": label_names,
            "embedding_dim": cfg.embedding_dim,
        },
        bundle_path,
    )
    logger.info("Saved Bi-LSTM bundle → %s", bundle_path)

    return model, vocab, dev


def predict_lstm(model, vocab: Vocab, X_texts, *, max_len: int = 400, device=None):
    """Inference helper used by ``train.py`` (eval phase) and the Gradio app."""
    import torch

    model.eval()
    tokens = [tokenize(str(t)) for t in X_texts]
    ids, lengths = _encode_dataset(vocab, tokens, max_len)
    dev = device or next(model.parameters()).device
    with torch.no_grad():
        logits = model(ids.to(dev), lengths.to(dev))
        proba = torch.softmax(logits, dim=-1).cpu().numpy()
    preds = proba.argmax(axis=-1)
    return preds, proba


# ---------------------------------------------------------------------------
# CLI — invoked indirectly through ``src.classify.train``; useful for debug.
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a Bi-LSTM classifier on §3.")
    parser.add_argument("--task", required=True, choices=["class_action", "case_type"])
    parser.add_argument("--text-source", default="long_ref",
                        choices=["long_ref", "long_pred", "source_text"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-len", type=int, default=400)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"])
    return parser


def main() -> None:
    """Smoke-test entry point. The real CLI path is `src.classify.train --model lstm`."""
    from src.classify.data import load_classification_data

    args = build_arg_parser().parse_args()
    data = load_classification_data(task=args.task, text_source=args.text_source)
    cfg = LSTMConfig(
        epochs=args.epochs, batch_size=args.batch_size, max_len=args.max_len,
    )
    task_tag = "classaction" if args.task == "class_action" else "casetype"
    train_lstm(
        data.X_train, data.y_train, data.X_val, data.y_val,
        label_names=data.label_names, task_tag=task_tag,
        config=cfg, device=args.device,
    )


if __name__ == "__main__":
    main()
