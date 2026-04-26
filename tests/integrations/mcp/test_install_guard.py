"""Verify the MCP integration raises a helpful ImportError when mcp is missing."""
import sys

import pytest

_NS_PREFIXES = ("finlit.integrations.mcp", "mcp")


def _is_target(modname: str) -> bool:
    return modname == "mcp" or any(
        modname == p or modname.startswith(p + ".") for p in _NS_PREFIXES
    )


def test_missing_mcp_extra_raises_helpful_importerror():
    # Snapshot every module we may touch so we can restore after the test.
    saved = {k: v for k, v in sys.modules.items() if _is_target(k)}
    try:
        # Drop cached imports so the next `import finlit.integrations.mcp` re-runs __init__.
        for mod in list(sys.modules):
            if _is_target(mod):
                del sys.modules[mod]
        # Block `import mcp` from succeeding.
        sys.modules["mcp"] = None  # type: ignore[assignment]

        with pytest.raises(ImportError, match=r"finlit\[mcp\] extras not installed"):
            import finlit.integrations.mcp  # noqa: F401
    finally:
        # Drop anything the test (or its triggered re-imports) inserted.
        for mod in list(sys.modules):
            if _is_target(mod) and mod not in saved:
                del sys.modules[mod]
        # Restore everything we snapshot.
        sys.modules.update(saved)
