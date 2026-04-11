"""Tests for finlit.extractors.base_vision.BaseVisionExtractor and
finlit.extractors.vision_extractor.VisionExtractor."""
from __future__ import annotations

import pytest

from finlit import schemas
from finlit.extractors.base_vision import BaseVisionExtractor
from finlit.extractors.pydantic_ai_extractor import ExtractionOutput
from finlit.extractors.vision_extractor import VisionExtractor


def test_base_vision_extractor_is_abstract():
    """Instantiating BaseVisionExtractor directly must fail — it's an ABC."""
    with pytest.raises(TypeError):
        BaseVisionExtractor()  # type: ignore[abstract]


def test_base_vision_extractor_subclass_must_implement_extract():
    """A subclass that does not implement extract() cannot be instantiated."""
    class Incomplete(BaseVisionExtractor):
        pass

    with pytest.raises(TypeError):
        Incomplete()  # type: ignore[abstract]


def test_base_vision_extractor_subclass_with_extract_works():
    """A subclass that implements extract() can be instantiated and called."""
    class Stub(BaseVisionExtractor):
        def extract(self, images, schema, text=""):
            return ExtractionOutput(
                fields={"payer_name": "Test"},
                confidence={"payer_name": 0.9},
            )

    s = Stub()
    out = s.extract([b"fake"], schemas.CRA_T5)
    assert out.fields["payer_name"] == "Test"


# ---------------- VisionExtractor tests ----------------


class _FakeRunResult:
    """Stand-in for pydantic-ai's RunResult object."""
    def __init__(self, output):
        self.output = output


class _FakeAgent:
    """Stand-in for pydantic-ai's Agent, capturing what was passed to it
    without making any LLM calls."""
    def __init__(self, canned_output):
        self.canned_output = canned_output
        self.last_prompt = None
        self.call_count = 0

    def run_sync(self, prompt):
        self.last_prompt = prompt
        self.call_count += 1
        return _FakeRunResult(self.canned_output)


def _install_fake_agent(vision_extractor, canned_output):
    """Replace the real pydantic-ai Agent on a VisionExtractor with a fake."""
    fake = _FakeAgent(canned_output)
    vision_extractor._agent = fake
    return fake


def test_vision_extractor_default_model_is_claude():
    """The default model string must be claude-sonnet-4-6 per the design."""
    ve = VisionExtractor()
    assert ve.model == "anthropic:claude-sonnet-4-6"


def test_vision_extractor_custom_dpi_stored():
    ve = VisionExtractor(dpi=300)
    assert ve.dpi == 300


def test_vision_extractor_default_dpi_is_200():
    ve = VisionExtractor()
    assert ve.dpi == 200


def test_vision_extractor_returns_extraction_output():
    """extract() returns whatever the underlying agent produced."""
    ve = VisionExtractor()
    canned = ExtractionOutput(
        fields={"payer_name": "Bank of Canada", "tax_year": 2024},
        confidence={"payer_name": 0.95, "tax_year": 0.99},
    )
    _install_fake_agent(ve, canned)

    out = ve.extract([b"fakepng"], schemas.CRA_T5)

    assert out.fields["payer_name"] == "Bank of Canada"
    assert out.fields["tax_year"] == 2024
    assert out.confidence["tax_year"] == 0.99


def test_vision_extractor_passes_images_to_agent():
    """The prompt passed to agent.run_sync must include BinaryContent parts,
    one per page image, with media_type image/png."""
    from pydantic_ai import BinaryContent

    ve = VisionExtractor()
    canned = ExtractionOutput(fields={}, confidence={})
    fake = _install_fake_agent(ve, canned)

    ve.extract([b"page1png", b"page2png"], schemas.CRA_T5)

    # The prompt should be a list: [text_prompt, BinaryContent, BinaryContent]
    assert isinstance(fake.last_prompt, list)
    binary_parts = [p for p in fake.last_prompt if isinstance(p, BinaryContent)]
    assert len(binary_parts) == 2
    assert all(bp.media_type == "image/png" for bp in binary_parts)
    assert binary_parts[0].data == b"page1png"
    assert binary_parts[1].data == b"page2png"


def test_vision_extractor_passes_text_hint_in_prompt():
    """When text= is provided, it should appear in the text prompt sent
    to the agent."""
    ve = VisionExtractor()
    canned = ExtractionOutput(fields={}, confidence={})
    fake = _install_fake_agent(ve, canned)

    ve.extract([b"img"], schemas.CRA_T5, text="Acme Corp employer name hint")

    # First element is the text prompt
    text_prompt = fake.last_prompt[0]
    assert isinstance(text_prompt, str)
    assert "Acme Corp employer name hint" in text_prompt


def test_vision_extractor_max_pages_enforced():
    """If max_pages is set and more pages are passed, raise ValueError
    BEFORE any LLM call."""
    ve = VisionExtractor(max_pages=2)
    fake = _install_fake_agent(ve, ExtractionOutput(fields={}, confidence={}))

    with pytest.raises(ValueError, match="max_pages"):
        ve.extract([b"p1", b"p2", b"p3"], schemas.CRA_T5)

    assert fake.call_count == 0  # agent was never called


def test_vision_extractor_max_pages_none_allows_unlimited():
    """max_pages=None (default) allows any number of pages."""
    ve = VisionExtractor()  # max_pages defaults to None
    _install_fake_agent(ve, ExtractionOutput(fields={}, confidence={}))

    # Should not raise
    ve.extract([b"p1"] * 20, schemas.CRA_T5)


def test_vision_extractor_max_pages_at_limit_succeeds():
    """Exactly max_pages images should not raise — the cap is strictly
    greater-than."""
    ve = VisionExtractor(max_pages=2)
    fake = _install_fake_agent(ve, ExtractionOutput(fields={}, confidence={}))

    ve.extract([b"p1", b"p2"], schemas.CRA_T5)

    assert fake.call_count == 1
