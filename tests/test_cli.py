"""CLI tests — uses typer.testing.CliRunner. No network calls."""
from __future__ import annotations

from unittest.mock import patch

from typer.testing import CliRunner

from finlit.cli.main import app
from finlit.result import ExtractionResult


runner = CliRunner()


def _fake_result() -> ExtractionResult:
    return ExtractionResult(
        fields={"employer_name": "Acme Corp", "box_14_employment_income": 87500.0},
        confidence={"employer_name": 0.99, "box_14_employment_income": 0.97},
        source_ref={},
        extraction_path="text",
    )


def test_extract_command_accepts_vision_extractor_flag(tmp_path):
    """The --vision-extractor flag is accepted and passes a VisionExtractor
    into DocumentPipeline's vision_extractor parameter."""
    fake_pdf = tmp_path / "t4.pdf"
    fake_pdf.write_bytes(b"x")

    captured = {}

    def _fake_init(self, **kwargs):
        captured.update(kwargs)
        # Don't run any real pipeline setup
        self.schema = kwargs["schema"]

    def _fake_run(self, path):
        return _fake_result()

    with patch("finlit.DocumentPipeline.__init__", _fake_init), patch(
        "finlit.DocumentPipeline.run", _fake_run
    ):
        result = runner.invoke(
            app,
            [
                "extract",
                str(fake_pdf),
                "--schema",
                "cra.t4",
                "--extractor",
                "claude",
                "--vision-extractor",
                "anthropic:claude-sonnet-4-6",
            ],
        )

    assert result.exit_code == 0, result.output
    assert captured.get("vision_extractor") is not None
    # Should be a VisionExtractor instance with the model we passed
    from finlit import VisionExtractor
    assert isinstance(captured["vision_extractor"], VisionExtractor)
    assert captured["vision_extractor"].model == "anthropic:claude-sonnet-4-6"


def test_extract_command_without_vision_extractor_flag(tmp_path):
    """When --vision-extractor is omitted, vision_extractor should be None."""
    fake_pdf = tmp_path / "t4.pdf"
    fake_pdf.write_bytes(b"x")

    captured = {}

    def _fake_init(self, **kwargs):
        captured.update(kwargs)
        self.schema = kwargs["schema"]

    def _fake_run(self, path):
        return _fake_result()

    with patch("finlit.DocumentPipeline.__init__", _fake_init), patch(
        "finlit.DocumentPipeline.run", _fake_run
    ):
        result = runner.invoke(
            app,
            ["extract", str(fake_pdf), "--schema", "cra.t4"],
        )

    assert result.exit_code == 0, result.output
    assert captured.get("vision_extractor") is None
