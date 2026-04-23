"""Tests for finlit.integrations.langchain.FinLitLoader."""
from __future__ import annotations

import inspect
from pathlib import Path

import pytest


def test_single_file_load_returns_one_document(
    t4_pipeline, patch_docling_parser, fake_t4_pdf
):
    from finlit.integrations.langchain import FinLitLoader

    loader = FinLitLoader(fake_t4_pdf, pipeline=t4_pipeline)
    docs = loader.load()

    assert len(docs) == 1
    doc = docs[0]
    assert doc.metadata["source"] == str(fake_t4_pdf)
    assert doc.metadata["finlit_fields"]["employer_name"] == "Acme Corp"
    assert "Acme Corp" in doc.page_content  # raw parsed text surfaces through


def test_metadata_contract_snapshot(
    t4_pipeline, patch_docling_parser, fake_t4_pdf
):
    """Lock the metadata schema. Adding a new field is OK (update the test);
    changing a field name is a breaking change this test is designed to
    catch before it merges."""
    from finlit.integrations.langchain import FinLitLoader

    loader = FinLitLoader(fake_t4_pdf, pipeline=t4_pipeline)
    doc = loader.load()[0]

    # page_content is the raw parsed text, not a synthesized summary
    assert "T4" in doc.page_content
    assert "Acme Corp" in doc.page_content
    assert "87500.00" in doc.page_content

    expected_keys = {
        "source",
        "finlit_schema",
        "finlit_model",
        "finlit_extraction_path",
        "finlit_needs_review",
        "finlit_extracted_field_count",
        "finlit_fields",
        "finlit_confidence",
        "finlit_source_ref",
        "finlit_warnings",
        "finlit_review_fields",
        "finlit_pii_entities",
    }
    assert set(doc.metadata.keys()) == expected_keys, (
        f"metadata drift: {set(doc.metadata.keys()) ^ expected_keys}"
    )

    assert doc.metadata["finlit_schema"] == "cra_t4"
    assert doc.metadata["finlit_extraction_path"] == "text"
    assert doc.metadata["finlit_needs_review"] is False
    assert isinstance(doc.metadata["finlit_fields"], dict)
    assert isinstance(doc.metadata["finlit_confidence"], dict)


def test_list_of_paths_preserves_order(
    t4_pipeline, patch_docling_parser, tmp_path
):
    from finlit.integrations.langchain import FinLitLoader

    p1 = tmp_path / "a.pdf"
    p1.write_bytes(b"x")
    p2 = tmp_path / "b.pdf"
    p2.write_bytes(b"x")
    p3 = tmp_path / "c.pdf"
    p3.write_bytes(b"x")

    loader = FinLitLoader([p1, p2, p3], pipeline=t4_pipeline)
    docs = loader.load()

    assert [d.metadata["source"] for d in docs] == [str(p1), str(p2), str(p3)]


def test_lazy_load_is_a_generator(
    t4_pipeline, patch_docling_parser, fake_t4_pdf
):
    from finlit.integrations.langchain import FinLitLoader

    loader = FinLitLoader(fake_t4_pdf, pipeline=t4_pipeline)
    iterator = loader.lazy_load()

    assert inspect.isgenerator(iterator)
    # Pulling one item must not require iterating the whole list
    first = next(iterator)
    assert first.metadata["source"] == str(fake_t4_pdf)


def test_loader_accepts_schema_kwarg(
    patch_docling_parser, high_confidence_t4_extractor, fake_t4_pdf
):
    """When schema is passed (and pipeline is not), the loader builds
    its own DocumentPipeline internally and uses it."""
    from finlit.integrations.langchain import FinLitLoader

    loader = FinLitLoader(
        fake_t4_pdf,
        schema="cra.t4",
        extractor=high_confidence_t4_extractor,
    )
    docs = loader.load()
    assert docs[0].metadata["finlit_schema"] == "cra_t4"
    assert docs[0].metadata["finlit_fields"]["employer_name"] == "Acme Corp"


def test_pipeline_wins_over_schema_kwarg(
    t4_pipeline, patch_docling_parser, fake_t4_pdf
):
    """When both pipeline and schema are passed, pipeline wins. Prove by
    passing a schema key that would fail to resolve if it were consulted."""
    from finlit.integrations.langchain import FinLitLoader

    loader = FinLitLoader(
        fake_t4_pdf,
        pipeline=t4_pipeline,
        schema="intentionally.broken.key",   # would raise if resolved
    )
    docs = loader.load()  # but pipeline wins, so no resolution happens
    assert len(docs) == 1


def test_missing_schema_and_pipeline_raises_at_init(fake_t4_pdf):
    from finlit.integrations.langchain import FinLitLoader

    with pytest.raises(ValueError, match="schema=... or pipeline=..."):
        FinLitLoader(fake_t4_pdf)


def test_on_error_raise_aborts_iteration(
    t4_pipeline, patch_docling_parser, tmp_path, monkeypatch
):
    """Default on_error='raise' re-raises and aborts the remaining files."""
    from finlit.integrations.langchain import FinLitLoader

    p1 = tmp_path / "ok1.pdf"
    p1.write_bytes(b"x")
    p2 = tmp_path / "boom.pdf"
    p2.write_bytes(b"x")
    p3 = tmp_path / "ok2.pdf"
    p3.write_bytes(b"x")

    original_run = t4_pipeline.run

    def _run(path):
        if Path(path).name == "boom.pdf":
            raise RuntimeError("kaboom")
        return original_run(path)

    monkeypatch.setattr(t4_pipeline, "run", _run)

    loader = FinLitLoader([p1, p2, p3], pipeline=t4_pipeline)
    with pytest.raises(RuntimeError, match="kaboom"):
        loader.load()


def test_on_error_skip_warns_and_continues(
    t4_pipeline, patch_docling_parser, tmp_path, monkeypatch, caplog
):
    """on_error='skip' logs a warning and yields only the good Documents."""
    from finlit.integrations.langchain import FinLitLoader

    p1 = tmp_path / "ok1.pdf"
    p1.write_bytes(b"x")
    p2 = tmp_path / "boom.pdf"
    p2.write_bytes(b"x")
    p3 = tmp_path / "ok2.pdf"
    p3.write_bytes(b"x")

    original_run = t4_pipeline.run

    def _run(path):
        if Path(path).name == "boom.pdf":
            raise RuntimeError("kaboom")
        return original_run(path)

    monkeypatch.setattr(t4_pipeline, "run", _run)

    import logging
    caplog.set_level(logging.WARNING, logger="finlit.integrations.langchain")

    loader = FinLitLoader([p1, p2, p3], pipeline=t4_pipeline, on_error="skip")
    docs = loader.load()

    assert [d.metadata["source"] for d in docs] == [str(p1), str(p3)]
    assert any("boom.pdf" in rec.message for rec in caplog.records)


def test_on_error_include_emits_failure_document(
    t4_pipeline, patch_docling_parser, tmp_path, monkeypatch
):
    """on_error='include' yields a Document with page_content='' and
    finlit_error / finlit_error_type in metadata."""
    from finlit.integrations.langchain import FinLitLoader

    p1 = tmp_path / "ok1.pdf"
    p1.write_bytes(b"x")
    p2 = tmp_path / "boom.pdf"
    p2.write_bytes(b"x")
    p3 = tmp_path / "ok2.pdf"
    p3.write_bytes(b"x")

    original_run = t4_pipeline.run

    def _run(path):
        if Path(path).name == "boom.pdf":
            raise RuntimeError("kaboom")
        return original_run(path)

    monkeypatch.setattr(t4_pipeline, "run", _run)

    loader = FinLitLoader([p1, p2, p3], pipeline=t4_pipeline, on_error="include")
    docs = loader.load()

    assert [d.metadata["source"] for d in docs] == [str(p1), str(p2), str(p3)]
    failure = docs[1]
    assert failure.page_content == ""
    assert failure.metadata["finlit_error_type"] == "RuntimeError"
    assert "kaboom" in failure.metadata["finlit_error"]


def test_audit_log_omitted_by_default(
    t4_pipeline, patch_docling_parser, fake_t4_pdf
):
    from finlit.integrations.langchain import FinLitLoader

    loader = FinLitLoader(fake_t4_pdf, pipeline=t4_pipeline)
    doc = loader.load()[0]
    assert "finlit_audit_log" not in doc.metadata


def test_audit_log_included_when_flag_set(
    t4_pipeline, patch_docling_parser, fake_t4_pdf
):
    from finlit.integrations.langchain import FinLitLoader

    loader = FinLitLoader(
        fake_t4_pdf, pipeline=t4_pipeline, include_audit_log=True
    )
    doc = loader.load()[0]
    assert "finlit_audit_log" in doc.metadata
    events = {e["event"] for e in doc.metadata["finlit_audit_log"]}
    assert "pipeline_complete" in events


def test_last_results_includes_none_for_skipped_failures(
    t4_pipeline, patch_docling_parser, tmp_path, monkeypatch
):
    """In skip mode, last_results gets a None placeholder so indices
    align with the input path list. Yielded docs are still only the
    successes — the user pairs them with paths via last_results."""
    from finlit.integrations.langchain import FinLitLoader
    from finlit.result import ExtractionResult

    p1 = tmp_path / "ok1.pdf"
    p1.write_bytes(b"x")
    p2 = tmp_path / "boom.pdf"
    p2.write_bytes(b"x")
    p3 = tmp_path / "ok2.pdf"
    p3.write_bytes(b"x")

    original_run = t4_pipeline.run

    def _run(path):
        if Path(path).name == "boom.pdf":
            raise RuntimeError("kaboom")
        return original_run(path)

    monkeypatch.setattr(t4_pipeline, "run", _run)

    loader = FinLitLoader([p1, p2, p3], pipeline=t4_pipeline, on_error="skip")
    _ = loader.load()

    assert len(loader.last_results) == 3
    assert isinstance(loader.last_results[0], ExtractionResult)
    assert loader.last_results[1] is None
    assert isinstance(loader.last_results[2], ExtractionResult)


def test_last_results_resets_on_each_load(
    t4_pipeline, patch_docling_parser, fake_t4_pdf
):
    from finlit.integrations.langchain import FinLitLoader

    loader = FinLitLoader(fake_t4_pdf, pipeline=t4_pipeline)
    loader.load()
    loader.load()  # second call
    assert len(loader.last_results) == 1  # not 2


def test_import_without_langchain_core_raises_helpful_error(monkeypatch):
    """If langchain-core is not installed, importing the integration
    module should raise ImportError with an install hint — not a cryptic
    ModuleNotFoundError from deep inside the loader."""
    import sys

    # Force both import caches and both find_module attempts to behave
    # as if langchain-core is absent.
    monkeypatch.setitem(sys.modules, "langchain_core", None)
    monkeypatch.setitem(sys.modules, "langchain_core.document_loaders", None)
    monkeypatch.setitem(sys.modules, "langchain_core.documents", None)

    # Also purge any already-loaded copies of our module so the import
    # re-runs.
    sys.modules.pop("finlit.integrations.langchain", None)
    sys.modules.pop("finlit.integrations.langchain.loader", None)

    with pytest.raises(ImportError, match=r"pip install finlit\[langchain\]"):
        import finlit.integrations.langchain  # noqa: F401
