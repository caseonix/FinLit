"""Stub extractor returns canned output regardless of input."""
from finlit.extractors.pydantic_ai_extractor import ExtractionOutput
from finlit.schema import Field, Schema
from tests.fixtures.stub_extractor import StubExtractor


def test_stub_returns_canned_output():
    canned = ExtractionOutput(
        fields={"sin": "123-456-789", "employee_name": "Test User"},
        confidence={"sin": 0.99, "employee_name": 0.95},
        notes="",
    )
    extractor = StubExtractor(canned)
    schema = Schema(name="x", fields=[Field(name="sin", pii=True), Field(name="employee_name")])

    out = extractor.extract("ignored text", schema)

    assert out.fields == {"sin": "123-456-789", "employee_name": "Test User"}
    assert out.confidence == {"sin": 0.99, "employee_name": 0.95}
