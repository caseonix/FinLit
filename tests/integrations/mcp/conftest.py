"""Shared helpers for FinLit MCP integration tests."""
from __future__ import annotations

import json
from typing import Any


def call_payload(call_result: Any) -> Any:
    """Extract the structured payload from a FastMCP `call_tool` result.

    FastMCP 1.x `call_tool` returns `tuple[list[ContentBlock], dict[str, Any]]`
    where the dict has a `"result"` key holding the structured value the tool
    returned. Older tool registrations may return just a list of ContentBlock
    (no structured payload), so we fall back to JSON-parsing the first text
    block in that case.
    """
    if isinstance(call_result, tuple) and len(call_result) == 2:
        _blocks, structured = call_result
        if isinstance(structured, dict) and "result" in structured:
            return structured["result"]
        return structured
    # Older shape: just a list of content blocks
    if isinstance(call_result, list) and call_result and hasattr(call_result[0], "text"):
        return json.loads(call_result[0].text)
    return call_result
