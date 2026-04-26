"""Entry point for `python -m finlit.integrations.mcp`.

Reads server-startup config from environment variables (matching the CLI
flag names), then calls serve(). Used by Claude Desktop / Cursor / any
mcpServers config that prefers `python -m` over a console script.
"""
from __future__ import annotations

import os

from finlit.integrations.mcp.server import serve

_PII_MODES = {"redact", "raw"}


def _get_pii_mode() -> str:
    mode = os.environ.get("FINLIT_PII_MODE", "redact")
    if mode not in _PII_MODES:
        raise SystemExit(f"FINLIT_PII_MODE must be one of {_PII_MODES}, got {mode!r}")
    return mode


def main() -> None:
    serve(
        extractor=os.environ.get("FINLIT_EXTRACTOR", "claude"),
        vision_extractor=os.environ.get("FINLIT_VISION_EXTRACTOR") or None,
        review_threshold=float(os.environ.get("FINLIT_REVIEW_THRESHOLD", "0.85")),
        pii_mode=_get_pii_mode(),  # type: ignore[arg-type]
    )


if __name__ == "__main__":
    main()
