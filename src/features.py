"""Featurization for classification (§3).

Currently exposes a single TF-IDF builder used by the classical models
(Naive Bayes, Logistic Regression). It returns a *fresh* sklearn
``TfidfVectorizer`` that callers wrap in a ``Pipeline`` together with
their classifier; this keeps fit-time and inference-time vocab in lock
step and avoids the train/test leakage of pre-fitting on the full corpus.

The Word2Vec path used by the Bi-LSTM lives in ``src/classify/lstm.py``
to keep that gensim dependency optional.

Design notes
------------
- ``ngram_range=(1, 2)`` to capture short legal phrases like
  "class action", "due process", "summary judgment".
- ``min_df=2`` drops single-occurrence tokens (mostly OCR garbage).
- ``max_features=50_000`` caps vocab so the LR coefficient vector fits
  comfortably in memory and SHAP stays tractable.
- ``sublinear_tf=True`` dampens the effect of very long documents.
- English stop-word removal is on; legal-domain stop words (e.g. "court",
  "plaintiff") are kept because they carry signal across our 5 case
  types (criminal-justice cases mention "prosecutor"/"jail" far more,
  etc.) and SHAP can later be inspected to confirm this.
"""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer

DEFAULT_NGRAM_RANGE = (1, 2)
DEFAULT_MAX_FEATURES = 50_000
DEFAULT_MIN_DF = 2


def build_tfidf_vectorizer(
    *,
    ngram_range: tuple[int, int] = DEFAULT_NGRAM_RANGE,
    max_features: int = DEFAULT_MAX_FEATURES,
    min_df: int = DEFAULT_MIN_DF,
    stop_words: str | None = "english",
    sublinear_tf: bool = True,
) -> TfidfVectorizer:
    """Return a fresh, *unfitted* TF-IDF vectorizer.

    Always call ``.fit_transform`` on training data only, then transform
    val/test — never fit on the full corpus, or feature counts leak.
    """
    return TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=min_df,
        stop_words=stop_words,
        sublinear_tf=sublinear_tf,
        lowercase=True,
        strip_accents="unicode",
    )
