"""Tests for finlit.extractors.base_vision.BaseVisionExtractor and
finlit.extractors.vision_extractor.VisionExtractor."""
from __future__ import annotations

import pytest

from finlit import schemas
from finlit.extractors.base_vision import BaseVisionExtractor
from finlit.extractors.pydantic_ai_extractor import ExtractionOutput


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
