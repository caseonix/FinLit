"""Pins the public import surface of finlit."""
import finlit


def test_top_level_exports_present():
    assert hasattr(finlit, "DocumentPipeline")
    assert hasattr(finlit, "BatchPipeline")
    assert hasattr(finlit, "Schema")
    assert hasattr(finlit, "Field")
    assert hasattr(finlit, "ExtractionResult")
    assert hasattr(finlit, "schemas")


def test_schemas_registry_has_expected_names():
    from finlit import schemas
    assert schemas.CRA_T4.name == "cra_t4"
    assert schemas.CRA_T5.name == "cra_t5"
    assert schemas.CRA_T4A.name == "cra_t4a"
    assert schemas.CRA_NR4.name == "cra_nr4"
    assert schemas.BANK_STATEMENT.name == "bank_statement"


def test_version_string_present():
    assert isinstance(finlit.__version__, str)


def test_all_exports_restricted():
    assert set(finlit.__all__) == {
        "DocumentPipeline",
        "BatchPipeline",
        "Schema",
        "Field",
        "ExtractionResult",
        "schemas",
        "VisionExtractor",
        "BaseVisionExtractor",
    }


def test_public_api_exports_vision_extractor():
    assert hasattr(finlit, "VisionExtractor")
    assert hasattr(finlit, "BaseVisionExtractor")
    assert "VisionExtractor" in finlit.__all__
    assert "BaseVisionExtractor" in finlit.__all__


def test_can_import_vision_extractor_from_top_level():
    from finlit import BaseVisionExtractor, VisionExtractor

    # Construct a default vision extractor (network is not touched here
    # because the underlying pydantic-ai Agent is built lazily).
    ve = VisionExtractor()
    assert ve.model == "anthropic:claude-sonnet-4-6"
    assert BaseVisionExtractor is not None
