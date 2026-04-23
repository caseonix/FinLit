"""LangChain integration for FinLit. Install with: pip install finlit[langchain]."""
try:
    from finlit.integrations.langchain.loader import FinLitLoader
except ImportError as exc:  # pragma: no cover - exercised via sys.modules patching
    raise ImportError(
        "finlit[langchain] extras not installed. "
        "Run: pip install finlit[langchain]"
    ) from exc

__all__ = ["FinLitLoader"]
