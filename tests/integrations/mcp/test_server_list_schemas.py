"""list_schemas tool returns one entry per built-in registry schema."""
import pytest

from finlit.integrations.mcp.server import build_app
from tests.integrations.mcp.conftest import call_payload


@pytest.mark.asyncio
async def test_list_schemas_returns_all_builtins():
    app = build_app(extractor="claude", vision_extractor=None,
                    review_threshold=0.85, pii_mode="redact")

    raw = await app.call_tool("list_schemas", {})
    schemas_list = call_payload(raw)

    keys = {entry["key"] for entry in schemas_list}
    assert keys == {"cra.t4", "cra.t5", "cra.t4a", "cra.nr4", "banking.bank_statement"}


@pytest.mark.asyncio
async def test_list_schemas_entry_shape():
    app = build_app(extractor="claude", vision_extractor=None,
                    review_threshold=0.85, pii_mode="redact")

    raw = await app.call_tool("list_schemas", {})
    schemas_list = call_payload(raw)
    t4 = next(e for e in schemas_list if e["key"] == "cra.t4")

    assert t4["name"]                          # non-empty document_type string
    assert isinstance(t4["version"], str) and t4["version"]
    assert t4["field_count"] > 0
    assert isinstance(t4["required_fields"], list)
    assert isinstance(t4["description"], str)
