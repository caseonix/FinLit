"""MCP integration for FinLit. Install with: pip install finlit[mcp]."""
try:
    import mcp  # noqa: F401
except ImportError as exc:  # pragma: no cover - exercised via sys.modules patching
    raise ImportError(
        "finlit[mcp] extras not installed. "
        "Run: pip install finlit[mcp]"
    ) from exc


def serve(**kwargs):
    """Run the FinLit MCP server. Lazy import to avoid loading server at package init time."""
    from finlit.integrations.mcp.server import serve as _serve
    return _serve(**kwargs)


__all__ = ["serve"]
