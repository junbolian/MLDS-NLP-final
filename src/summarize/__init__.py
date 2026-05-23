"""Multi-granularity summarization components."""


def summarize(text: str) -> dict[str, str]:
    """Lazy package-level convenience wrapper."""

    from src.summarize.pipeline import summarize as _summarize

    return _summarize(text)


__all__ = ["summarize"]
