"""detect_pii tool: standalone Presidio + Canadian recognizers, no LLM."""
import pytest

from finlit.integrations.mcp.server import build_app
from tests.integrations.mcp.conftest import call_payload


@pytest.mark.asyncio
async def test_detect_pii_finds_sin_and_postal():
    app = build_app(extractor="claude", vision_extractor=None,
                    review_threshold=0.85, pii_mode="redact")

    text = "John Doe lives at M5V 3A8 with SIN 123-456-789."
    raw = await app.call_tool("detect_pii", {"text": text})
    payload = call_payload(raw)

    types = {e["entity_type"] for e in payload["entities"]}
    assert "CA_SIN" in types
    assert "CA_POSTAL_CODE" in types


@pytest.mark.asyncio
async def test_detect_pii_returns_redacted_when_requested():
    app = build_app(extractor="claude", vision_extractor=None,
                    review_threshold=0.85, pii_mode="redact")

    raw = await app.call_tool("detect_pii", {
        "text": "SIN 123-456-789", "return_redacted": True,
    })
    payload = call_payload(raw)

    assert "redacted_text" in payload
    assert "123-456-789" not in payload["redacted_text"]
    assert "***-***-***" in payload["redacted_text"]


@pytest.mark.asyncio
async def test_detect_pii_omits_redacted_by_default():
    app = build_app(extractor="claude", vision_extractor=None,
                    review_threshold=0.85, pii_mode="redact")

    raw = await app.call_tool("detect_pii", {"text": "SIN 123-456-789"})
    payload = call_payload(raw)

    assert payload.get("redacted_text") is None
