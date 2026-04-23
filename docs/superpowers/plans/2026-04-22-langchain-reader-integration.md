# LangChain Reader Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `FinLitLoader`, a LangChain `BaseLoader` subclass in `finlit.integrations.langchain`, that turns any file `DocumentPipeline` can process into LangChain `Document` objects with FinLit's structured fields as namespaced metadata.

**Architecture:** New optional subpackage `finlit/integrations/langchain/` gated behind the `finlit[langchain]` extra. Core FinLit is untouched. `FinLitLoader.lazy_load()` wraps `DocumentPipeline.run()` and emits one `Document` per input file with `page_content` = raw Docling text and `metadata` = `source` + `finlit_*`-prefixed structured output. A shared `_resolve_schema` helper at `finlit/integrations/_schema_resolver.py` prepares for the eventual LlamaIndex port.

**Tech Stack:** Python 3.10+, `langchain-core>=0.3.0` (optional dep), pytest. Tests reuse the existing `StubExtractor` / `synthetic_parsed_document` / `monkeypatch`-the-parser pattern — zero network, zero LLM calls.

**Spec:** `docs/superpowers/specs/2026-04-22-langchain-llamaindex-readers-design.md`

---

## Preflight

Before starting:

```bash
cd /Users/srivatsakasagar/Development/FinLit
git status            # expect: main, clean working tree after spec commits
pytest tests/ -v      # expect: existing suite passes (sets baseline)
```

If the baseline fails, fix that first — do not start this plan on a broken main.

---

## Task 1: Add `langchain` optional dependency

**Files:**
- Modify: `pyproject.toml:35-36`

- [ ] **Step 1: Read current extras block**

Run: `grep -n "optional-dependencies" /Users/srivatsakasagar/Development/FinLit/pyproject.toml`
Expected: line `[project.optional-dependencies]` followed by `dev = [...]`.

- [ ] **Step 2: Add the `langchain` extra**

Replace the `[project.optional-dependencies]` block in `pyproject.toml` with:

```toml
[project.optional-dependencies]
dev = ["pytest", "pytest-asyncio", "ruff", "mypy"]
langchain = ["langchain-core>=0.3.0"]
```

- [ ] **Step 3: Verify the bare install still works**

Run: `pip install -e . --no-deps --force-reinstall --quiet && python -c "import finlit; print(finlit.__version__)"`
Expected: prints a version string, no errors.

- [ ] **Step 4: Verify the `langchain` extra installs**

Run: `pip install -e ".[langchain]" --quiet && python -c "from langchain_core.document_loaders import BaseLoader; from langchain_core.documents import Document; print('ok')"`
Expected: prints `ok`.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml
git commit -m "build(finlit): add optional langchain extra

Depends on langchain-core, not langchain, to keep the surface narrow
and avoid pulling in community retrievers/chains.

Install via: pip install finlit[langchain]"
```

---

## Task 2: Package skeleton + test conftest

**Files:**
- Create: `finlit/integrations/__init__.py`
- Create: `finlit/integrations/langchain/__init__.py`
- Create: `finlit/integrations/langchain/loader.py` (empty module docstring only)
- Create: `tests/integrations/__init__.py`
- Create: `tests/integrations/conftest.py`

- [ ] **Step 1: Create the namespace files**

Create `finlit/integrations/__init__.py` with exactly:

```python
"""FinLit framework integrations (LangChain, LlamaIndex, etc.).

Each subpackage requires the corresponding optional extra. See pyproject.toml.
"""
```

Create `finlit/integrations/langchain/__init__.py` with exactly:

```python
"""LangChain integration for FinLit. Install with: pip install finlit[langchain]."""
```

Create `finlit/integrations/langchain/loader.py` with exactly:

```python
"""FinLitLoader — LangChain BaseLoader wrapper around DocumentPipeline."""
```

Create `tests/integrations/__init__.py` as an empty file.

- [ ] **Step 2: Create the integration test conftest**

Create `tests/integrations/conftest.py` with:

```python
"""Shared fixtures for finlit.integrations tests.

These tests run DocumentPipeline end-to-end against a stub extractor
and a monkeypatched Docling parser, so no network or LLM calls happen.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from finlit import schemas
from finlit.parsers.docling_parser import ParsedDocument
from finlit.pipeline import DocumentPipeline

from tests.conftest import StubExtractor  # noqa: F401  (re-exported for tests)


@pytest.fixture
def patch_docling_parser(monkeypatch, synthetic_parsed_document):
    """Replace DoclingParser.parse with a function that returns a canned
    ParsedDocument. Tests that want the default t4 text can just request
    this fixture; tests that need a different ParsedDocument can shadow
    the fixture locally."""
    def _fake_parse(self, path):  # noqa: ARG001
        return synthetic_parsed_document

    monkeypatch.setattr(
        "finlit.parsers.docling_parser.DoclingParser.parse", _fake_parse
    )
    return synthetic_parsed_document


@pytest.fixture
def t4_pipeline(high_confidence_t4_extractor) -> DocumentPipeline:
    """A DocumentPipeline pre-wired with a deterministic T4 stub extractor.

    Tests that need a pipeline but don't care about constructor branches
    in the loader should take this fixture + patch_docling_parser together."""
    return DocumentPipeline(
        schema=schemas.CRA_T4,
        extractor=high_confidence_t4_extractor,
    )


@pytest.fixture
def fake_t4_pdf(tmp_path) -> Path:
    """A temp file path that DoclingParser will never actually read (the
    parser is monkeypatched) but whose existence satisfies os.stat calls."""
    p = tmp_path / "t4.pdf"
    p.write_bytes(b"not a real pdf")
    return p
```

- [ ] **Step 3: Confirm package is importable**

Run: `python -c "import finlit.integrations.langchain; print('ok')"`
Expected: `ok`, no errors.

- [ ] **Step 4: Confirm test discovery is still clean**

Run: `pytest tests/ -v --collect-only | tail -5`
Expected: no collection errors, existing tests still listed.

- [ ] **Step 5: Commit**

```bash
git add finlit/integrations/ tests/integrations/
git commit -m "feat(integrations): scaffold finlit.integrations.langchain package

Add empty namespace modules and a shared integration-test conftest
that reuses the existing StubExtractor + monkeypatch-the-parser pattern
so downstream tests run with zero network or LLM activity."
```

---

## Task 3: Happy-path `FinLitLoader` — one file → one `Document`

**Files:**
- Modify: `finlit/integrations/langchain/loader.py`
- Modify: `finlit/integrations/langchain/__init__.py`
- Create: `tests/integrations/test_langchain_loader.py`

- [ ] **Step 1: Write the failing test**

Create `tests/integrations/test_langchain_loader.py` with:

```python
"""Tests for finlit.integrations.langchain.FinLitLoader."""
from __future__ import annotations


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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/integrations/test_langchain_loader.py::test_single_file_load_returns_one_document -v`
Expected: FAIL with `ImportError` on `FinLitLoader` (module has no such name yet).

- [ ] **Step 3: Implement the minimal loader**

Replace the contents of `finlit/integrations/langchain/loader.py` with:

```python
"""FinLitLoader — LangChain BaseLoader wrapper around DocumentPipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Union

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

from finlit.pipeline import DocumentPipeline
from finlit.result import ExtractionResult


PathLike = Union[str, Path]


class FinLitLoader(BaseLoader):
    """Load files through a FinLit DocumentPipeline and emit LangChain Documents.

    One Document per input file. `page_content` carries the raw parsed text;
    `metadata` carries the structured ExtractionResult under `finlit_*` keys.
    """

    def __init__(
        self,
        file_path: PathLike | list[PathLike],
        *,
        pipeline: DocumentPipeline,
    ) -> None:
        if isinstance(file_path, (str, Path)):
            self._paths: list[Path] = [Path(file_path)]
        else:
            self._paths = [Path(p) for p in file_path]
        self._pipeline = pipeline
        self.last_results: list[ExtractionResult | None] = []

    def lazy_load(self) -> Iterator[Document]:
        self.last_results = []
        for path in self._paths:
            result = self._pipeline.run(path)
            self.last_results.append(result)
            yield _build_document(path, result)


def _build_document(path: Path, result: ExtractionResult) -> Document:
    """Map an ExtractionResult into a LangChain Document.

    The raw parsed text is NOT stored on ExtractionResult today, so we read
    it back from the pipeline's `_parser` via a side channel. For now we
    surface what we have: the pipeline's validated fields as metadata, and
    a placeholder page_content we will enrich in the next task.
    """
    # NOTE: result does not carry the raw parsed text. Task 4 replaces this
    # placeholder with the real full_text once the pipeline passes it
    # through. For this task, we synthesise page_content from the fields
    # so the happy-path test can assert a known value is present.
    page_content = "\n".join(
        f"{k}: {v}" for k, v in result.fields.items() if v is not None
    )
    return Document(
        page_content=page_content,
        metadata={
            "source": str(path),
            "finlit_fields": dict(result.fields),
        },
    )
```

Replace `finlit/integrations/langchain/__init__.py` with:

```python
"""LangChain integration for FinLit. Install with: pip install finlit[langchain]."""
from finlit.integrations.langchain.loader import FinLitLoader

__all__ = ["FinLitLoader"]
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/integrations/test_langchain_loader.py::test_single_file_load_returns_one_document -v`
Expected: PASS. (The "Acme Corp" substring appears in the field-dump placeholder; `source` and `finlit_fields` are present.)

- [ ] **Step 5: Commit**

```bash
git add finlit/integrations/langchain/loader.py finlit/integrations/langchain/__init__.py tests/integrations/test_langchain_loader.py
git commit -m "feat(langchain): happy-path FinLitLoader

Minimal BaseLoader that wraps DocumentPipeline and emits one Document
per file with source + finlit_fields in metadata. page_content is a
placeholder field dump until Task 4 wires through the raw parsed text."
```

---

## Task 4: Wire raw parsed text into `page_content` + full metadata contract

**Files:**
- Modify: `finlit/integrations/langchain/loader.py`
- Modify: `tests/integrations/test_langchain_loader.py`

The spec (§5) requires `page_content = parsed.full_text` and a full set of `finlit_*` metadata keys. The pipeline today returns an `ExtractionResult` that does NOT expose the raw parsed text, so we extend the loader to call the parser itself (reusing the pipeline's configured parser) rather than modifying `ExtractionResult` and the public API.

- [ ] **Step 1: Write the failing contract-snapshot test**

Append to `tests/integrations/test_langchain_loader.py`:

```python
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

    assert doc.metadata["finlit_schema"] == "CRA_T4"
    assert doc.metadata["finlit_extraction_path"] == "text"
    assert doc.metadata["finlit_needs_review"] is False
    assert isinstance(doc.metadata["finlit_fields"], dict)
    assert isinstance(doc.metadata["finlit_confidence"], dict)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/integrations/test_langchain_loader.py::test_metadata_contract_snapshot -v`
Expected: FAIL. Current loader emits only `source` + `finlit_fields`; it is missing ten keys and page_content is a field dump, not the raw text.

- [ ] **Step 3: Implement page_content + full metadata**

Replace `finlit/integrations/langchain/loader.py` with:

```python
"""FinLitLoader — LangChain BaseLoader wrapper around DocumentPipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Union

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

from finlit.pipeline import DocumentPipeline
from finlit.result import ExtractionResult


PathLike = Union[str, Path]


class FinLitLoader(BaseLoader):
    """Load files through a FinLit DocumentPipeline and emit LangChain Documents.

    One Document per input file. `page_content` is the raw parsed text from
    Docling; `metadata` carries the structured ExtractionResult under
    `finlit_*` keys (plus `source` per LangChain convention).
    """

    def __init__(
        self,
        file_path: PathLike | list[PathLike],
        *,
        pipeline: DocumentPipeline,
    ) -> None:
        if isinstance(file_path, (str, Path)):
            self._paths: list[Path] = [Path(file_path)]
        else:
            self._paths = [Path(p) for p in file_path]
        self._pipeline = pipeline
        self.last_results: list[ExtractionResult | None] = []

    def lazy_load(self) -> Iterator[Document]:
        self.last_results = []
        for path in self._paths:
            # Parse once for page_content. This is the same parser the
            # pipeline will use internally; Docling caches nothing stateful
            # across parse() calls, so the double-parse cost is acceptable
            # for v0.1. A future optimisation can thread the parsed text
            # through ExtractionResult to avoid the second call.
            parsed = self._pipeline._parser.parse(path)
            result = self._pipeline.run(path)
            self.last_results.append(result)
            yield _build_document(path, parsed.full_text, result)


def _build_document(
    path: Path, full_text: str, result: ExtractionResult
) -> Document:
    return Document(
        page_content=full_text,
        metadata={
            "source": str(path),
            "finlit_schema": result.schema_name,
            "finlit_model": result.extractor_model,
            "finlit_extraction_path": result.extraction_path,
            "finlit_needs_review": result.needs_review,
            "finlit_extracted_field_count": result.extracted_field_count,
            "finlit_fields": dict(result.fields),
            "finlit_confidence": dict(result.confidence),
            "finlit_source_ref": dict(result.source_ref),
            "finlit_warnings": list(result.warnings),
            "finlit_review_fields": list(result.review_fields),
            "finlit_pii_entities": list(result.pii_entities),
        },
    )
```

- [ ] **Step 4: Run both tests to verify they pass**

Run: `pytest tests/integrations/test_langchain_loader.py -v`
Expected: both tests PASS.

- [ ] **Step 5: Commit**

```bash
git add finlit/integrations/langchain/loader.py tests/integrations/test_langchain_loader.py
git commit -m "feat(langchain): full metadata contract + raw parsed text page_content

Document.page_content now holds the raw Docling text (for splitters /
RAG). Metadata carries the 12-key finlit_* contract locked by a
snapshot test."
```

---

## Task 5: Batch support — list of paths, streaming generator

**Files:**
- Modify: `tests/integrations/test_langchain_loader.py`

No code changes needed — the loader already accepts `list[PathLike]` and `lazy_load` already iterates. We just prove the behavior with tests.

- [ ] **Step 1: Write the failing tests**

Add `import inspect` to the imports block at the top of `tests/integrations/test_langchain_loader.py`, then append the test functions below to the bottom of the file:

```python
def test_list_of_paths_preserves_order(
    t4_pipeline, patch_docling_parser, tmp_path
):
    from finlit.integrations.langchain import FinLitLoader

    p1 = tmp_path / "a.pdf"; p1.write_bytes(b"x")
    p2 = tmp_path / "b.pdf"; p2.write_bytes(b"x")
    p3 = tmp_path / "c.pdf"; p3.write_bytes(b"x")

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
```

- [ ] **Step 2: Run the tests**

Run: `pytest tests/integrations/test_langchain_loader.py -v`
Expected: all PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/integrations/test_langchain_loader.py
git commit -m "test(langchain): cover batch ordering and lazy_load streaming"
```

---

## Task 6: `schema=` kwarg support via shared `_resolve_schema`

**Files:**
- Create: `finlit/integrations/_schema_resolver.py`
- Create: `tests/integrations/test_schema_resolver.py`
- Modify: `finlit/integrations/langchain/loader.py`
- Modify: `tests/integrations/test_langchain_loader.py`

- [ ] **Step 1: Write failing tests for `_resolve_schema`**

Create `tests/integrations/test_schema_resolver.py`:

```python
"""Tests for the shared schema resolver used by integration loaders."""
from __future__ import annotations

import pytest

from finlit import schemas
from finlit.schema import Schema


def test_resolve_schema_accepts_schema_object():
    from finlit.integrations._schema_resolver import _resolve_schema
    assert _resolve_schema(schemas.CRA_T4) is schemas.CRA_T4


def test_resolve_schema_accepts_dotted_registry_key():
    from finlit.integrations._schema_resolver import _resolve_schema
    resolved = _resolve_schema("cra.t4")
    assert isinstance(resolved, Schema)
    assert resolved is schemas.CRA_T4


def test_resolve_schema_accepts_python_registry_name():
    from finlit.integrations._schema_resolver import _resolve_schema
    resolved = _resolve_schema("CRA_T4")
    assert resolved is schemas.CRA_T4


def test_resolve_schema_accepts_banking_dotted_key():
    from finlit.integrations._schema_resolver import _resolve_schema
    resolved = _resolve_schema("banking.bank_statement")
    assert resolved is schemas.BANK_STATEMENT


def test_resolve_schema_rejects_unknown_string():
    from finlit.integrations._schema_resolver import _resolve_schema
    with pytest.raises(ValueError, match="Unknown schema"):
        _resolve_schema("not.a.thing")


def test_resolve_schema_rejects_wrong_type():
    from finlit.integrations._schema_resolver import _resolve_schema
    with pytest.raises(TypeError):
        _resolve_schema(123)  # type: ignore[arg-type]
```

- [ ] **Step 2: Run and verify failure**

Run: `pytest tests/integrations/test_schema_resolver.py -v`
Expected: FAIL — `ImportError: No module named 'finlit.integrations._schema_resolver'`.

- [ ] **Step 3: Implement `_resolve_schema`**

Create `finlit/integrations/_schema_resolver.py`:

```python
"""Shared schema-input resolver used by integration loaders.

Accepts a Schema object, a dotted registry key ('cra.t4'), or the Python
registry name ('CRA_T4'). Returns the resolved Schema or raises.
"""
from __future__ import annotations

from finlit import schemas
from finlit.schema import Schema


# Dotted key → attribute name on `finlit.schemas`.
_DOTTED_TO_ATTR = {
    "cra.t4": "CRA_T4",
    "cra.t5": "CRA_T5",
    "cra.t4a": "CRA_T4A",
    "cra.nr4": "CRA_NR4",
    "banking.bank_statement": "BANK_STATEMENT",
}


def _resolve_schema(schema: Schema | str) -> Schema:
    """Coerce a schema input into a Schema instance.

    Acceptable forms:
        Schema instance       → returned as-is
        "cra.t4"              → schemas.CRA_T4
        "CRA_T4"              → schemas.CRA_T4
    """
    if isinstance(schema, Schema):
        return schema
    if isinstance(schema, str):
        attr = _DOTTED_TO_ATTR.get(schema, schema)
        resolved = getattr(schemas, attr, None)
        if isinstance(resolved, Schema):
            return resolved
        raise ValueError(
            f"Unknown schema {schema!r}. "
            f"Valid dotted keys: {sorted(_DOTTED_TO_ATTR)}. "
            f"Or pass a Schema instance directly."
        )
    raise TypeError(
        f"schema must be a Schema or a str, got {type(schema).__name__}"
    )
```

- [ ] **Step 4: Run resolver tests**

Run: `pytest tests/integrations/test_schema_resolver.py -v`
Expected: all six PASS.

- [ ] **Step 5: Write the failing loader test for `schema=` kwarg**

Append to `tests/integrations/test_langchain_loader.py`:

```python
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
    assert docs[0].metadata["finlit_schema"] == "CRA_T4"
    assert docs[0].metadata["finlit_fields"]["employer_name"] == "Acme Corp"
```

- [ ] **Step 6: Run and verify failure**

Run: `pytest tests/integrations/test_langchain_loader.py::test_loader_accepts_schema_kwarg -v`
Expected: FAIL — current signature has only `pipeline=` kwarg.

- [ ] **Step 7: Extend `FinLitLoader.__init__` with `schema` / `extractor` kwargs**

Replace `finlit/integrations/langchain/loader.py` with:

```python
"""FinLitLoader — LangChain BaseLoader wrapper around DocumentPipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Union

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

from finlit.extractors.base import BaseExtractor
from finlit.integrations._schema_resolver import _resolve_schema
from finlit.pipeline import DocumentPipeline
from finlit.result import ExtractionResult
from finlit.schema import Schema


PathLike = Union[str, Path]


class FinLitLoader(BaseLoader):
    """Load files through a FinLit DocumentPipeline and emit LangChain Documents.

    One Document per input file. `page_content` is the raw parsed text from
    Docling; `metadata` carries the structured ExtractionResult under
    `finlit_*` keys (plus `source` per LangChain convention).

    Construction:
        FinLitLoader(path, schema="cra.t4")                 # build pipeline
        FinLitLoader(path, schema=my_schema)                # Schema instance
        FinLitLoader(path, pipeline=my_pipeline)            # inject pipeline
        FinLitLoader(path, pipeline=p, schema="cra.t4")     # pipeline wins
    """

    def __init__(
        self,
        file_path: PathLike | list[PathLike],
        *,
        schema: Schema | str | None = None,
        extractor: str | BaseExtractor = "claude",
        pipeline: DocumentPipeline | None = None,
    ) -> None:
        if isinstance(file_path, (str, Path)):
            self._paths: list[Path] = [Path(file_path)]
        else:
            self._paths = [Path(p) for p in file_path]

        if pipeline is not None:
            self._pipeline = pipeline
        elif schema is not None:
            self._pipeline = DocumentPipeline(
                schema=_resolve_schema(schema),
                extractor=extractor,
            )
        else:
            raise ValueError(
                "FinLitLoader requires either schema=... or pipeline=..."
            )

        self.last_results: list[ExtractionResult | None] = []

    def lazy_load(self) -> Iterator[Document]:
        self.last_results = []
        for path in self._paths:
            parsed = self._pipeline._parser.parse(path)
            result = self._pipeline.run(path)
            self.last_results.append(result)
            yield _build_document(path, parsed.full_text, result)


def _build_document(
    path: Path, full_text: str, result: ExtractionResult
) -> Document:
    return Document(
        page_content=full_text,
        metadata={
            "source": str(path),
            "finlit_schema": result.schema_name,
            "finlit_model": result.extractor_model,
            "finlit_extraction_path": result.extraction_path,
            "finlit_needs_review": result.needs_review,
            "finlit_extracted_field_count": result.extracted_field_count,
            "finlit_fields": dict(result.fields),
            "finlit_confidence": dict(result.confidence),
            "finlit_source_ref": dict(result.source_ref),
            "finlit_warnings": list(result.warnings),
            "finlit_review_fields": list(result.review_fields),
            "finlit_pii_entities": list(result.pii_entities),
        },
    )
```

- [ ] **Step 8: Run all tests**

Run: `pytest tests/integrations/ -v`
Expected: all PASS.

- [ ] **Step 9: Commit**

```bash
git add finlit/integrations/_schema_resolver.py tests/integrations/test_schema_resolver.py finlit/integrations/langchain/loader.py tests/integrations/test_langchain_loader.py
git commit -m "feat(langchain): schema= kwarg + shared _resolve_schema helper

Loader now builds its own DocumentPipeline when schema=... is passed.
_resolve_schema lives at finlit/integrations/_schema_resolver.py so
the future LlamaIndex reader imports it without cross-framework
coupling. Accepts Schema instances, dotted keys ('cra.t4'), or Python
registry names ('CRA_T4')."
```

---

## Task 7: Precedence + fail-fast validation

**Files:**
- Modify: `tests/integrations/test_langchain_loader.py`

No code changes needed — the current `__init__` already does pipeline-wins and raises on missing-both. Tests lock both behaviors.

- [ ] **Step 1: Write the failing tests**

Add `import pytest` to the imports block at the top of `tests/integrations/test_langchain_loader.py`, then append the test functions below to the bottom of the file:

```python
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
```

- [ ] **Step 2: Run the tests**

Run: `pytest tests/integrations/test_langchain_loader.py -v`
Expected: all PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/integrations/test_langchain_loader.py
git commit -m "test(langchain): lock pipeline-wins precedence and fail-fast init"
```

---

## Task 8: `on_error="raise"` default + `on_error="skip"`

**Files:**
- Modify: `finlit/integrations/langchain/loader.py`
- Modify: `tests/integrations/test_langchain_loader.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/integrations/test_langchain_loader.py`:

```python
def test_on_error_raise_aborts_iteration(
    t4_pipeline, patch_docling_parser, tmp_path, monkeypatch
):
    """Default on_error='raise' re-raises and aborts the remaining files."""
    from finlit.integrations.langchain import FinLitLoader

    p1 = tmp_path / "ok1.pdf"; p1.write_bytes(b"x")
    p2 = tmp_path / "boom.pdf"; p2.write_bytes(b"x")
    p3 = tmp_path / "ok2.pdf"; p3.write_bytes(b"x")

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

    p1 = tmp_path / "ok1.pdf"; p1.write_bytes(b"x")
    p2 = tmp_path / "boom.pdf"; p2.write_bytes(b"x")
    p3 = tmp_path / "ok2.pdf"; p3.write_bytes(b"x")

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
```

Also add `from pathlib import Path` to the imports block at the top of the test file — the `_run` closure uses `Path(path).name`. Tasks 9 and 11 reuse this import.

- [ ] **Step 2: Run and verify failure**

Run: `pytest tests/integrations/test_langchain_loader.py -v -k on_error`
Expected: FAIL — `on_error` kwarg does not exist yet.

- [ ] **Step 3: Implement `on_error="raise" | "skip"` (leave `"include"` for Task 9)**

Replace the contents of `finlit/integrations/langchain/loader.py` with:

```python
"""FinLitLoader — LangChain BaseLoader wrapper around DocumentPipeline."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator, Literal, Union

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

from finlit.extractors.base import BaseExtractor
from finlit.integrations._schema_resolver import _resolve_schema
from finlit.pipeline import DocumentPipeline
from finlit.result import ExtractionResult
from finlit.schema import Schema


_log = logging.getLogger(__name__)

PathLike = Union[str, Path]
OnError = Literal["raise", "skip", "include"]


class FinLitLoader(BaseLoader):
    """Load files through a FinLit DocumentPipeline and emit LangChain Documents.

    See design doc: docs/superpowers/specs/2026-04-22-langchain-llamaindex-readers-design.md
    """

    def __init__(
        self,
        file_path: PathLike | list[PathLike],
        *,
        schema: Schema | str | None = None,
        extractor: str | BaseExtractor = "claude",
        pipeline: DocumentPipeline | None = None,
        on_error: OnError = "raise",
    ) -> None:
        if isinstance(file_path, (str, Path)):
            self._paths: list[Path] = [Path(file_path)]
        else:
            self._paths = [Path(p) for p in file_path]

        if pipeline is not None:
            self._pipeline = pipeline
        elif schema is not None:
            self._pipeline = DocumentPipeline(
                schema=_resolve_schema(schema),
                extractor=extractor,
            )
        else:
            raise ValueError(
                "FinLitLoader requires either schema=... or pipeline=..."
            )

        if on_error not in ("raise", "skip", "include"):
            raise ValueError(
                f"on_error must be 'raise', 'skip', or 'include', got {on_error!r}"
            )
        self._on_error = on_error

        self.last_results: list[ExtractionResult | None] = []

    def lazy_load(self) -> Iterator[Document]:
        self.last_results = []
        for path in self._paths:
            try:
                parsed = self._pipeline._parser.parse(path)
                result = self._pipeline.run(path)
            except Exception as exc:
                if self._on_error == "raise":
                    raise
                if self._on_error == "skip":
                    _log.warning(
                        "FinLit extraction failed for %s: %s", path, exc
                    )
                    continue
                # "include" handled in Task 9
                raise  # pragma: no cover
            self.last_results.append(result)
            yield _build_document(path, parsed.full_text, result)


def _build_document(
    path: Path, full_text: str, result: ExtractionResult
) -> Document:
    return Document(
        page_content=full_text,
        metadata={
            "source": str(path),
            "finlit_schema": result.schema_name,
            "finlit_model": result.extractor_model,
            "finlit_extraction_path": result.extraction_path,
            "finlit_needs_review": result.needs_review,
            "finlit_extracted_field_count": result.extracted_field_count,
            "finlit_fields": dict(result.fields),
            "finlit_confidence": dict(result.confidence),
            "finlit_source_ref": dict(result.source_ref),
            "finlit_warnings": list(result.warnings),
            "finlit_review_fields": list(result.review_fields),
            "finlit_pii_entities": list(result.pii_entities),
        },
    )
```

- [ ] **Step 4: Run all loader tests**

Run: `pytest tests/integrations/test_langchain_loader.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add finlit/integrations/langchain/loader.py tests/integrations/test_langchain_loader.py
git commit -m "feat(langchain): on_error raise|skip with structured logging"
```

---

## Task 9: `on_error="include"` — failure Document with `finlit_error` metadata

**Files:**
- Modify: `finlit/integrations/langchain/loader.py`
- Modify: `tests/integrations/test_langchain_loader.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/integrations/test_langchain_loader.py`:

```python
def test_on_error_include_emits_failure_document(
    t4_pipeline, patch_docling_parser, tmp_path, monkeypatch
):
    """on_error='include' yields a Document with page_content='' and
    finlit_error / finlit_error_type in metadata."""
    from finlit.integrations.langchain import FinLitLoader

    p1 = tmp_path / "ok1.pdf"; p1.write_bytes(b"x")
    p2 = tmp_path / "boom.pdf"; p2.write_bytes(b"x")
    p3 = tmp_path / "ok2.pdf"; p3.write_bytes(b"x")

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
```

- [ ] **Step 2: Run and verify failure**

Run: `pytest tests/integrations/test_langchain_loader.py::test_on_error_include_emits_failure_document -v`
Expected: FAIL — current "include" branch still re-raises.

- [ ] **Step 3: Implement `"include"` branch**

In `finlit/integrations/langchain/loader.py`, replace the `lazy_load` method with:

```python
    def lazy_load(self) -> Iterator[Document]:
        self.last_results = []
        for path in self._paths:
            try:
                parsed = self._pipeline._parser.parse(path)
                result = self._pipeline.run(path)
            except Exception as exc:
                if self._on_error == "raise":
                    raise
                if self._on_error == "skip":
                    _log.warning(
                        "FinLit extraction failed for %s: %s", path, exc
                    )
                    continue
                # on_error == "include"
                _log.warning(
                    "FinLit extraction failed for %s (emitted as error Document): %s",
                    path,
                    exc,
                )
                self.last_results.append(None)
                yield Document(
                    page_content="",
                    metadata={
                        "source": str(path),
                        "finlit_error": repr(exc),
                        "finlit_error_type": type(exc).__name__,
                    },
                )
                continue
            self.last_results.append(result)
            yield _build_document(path, parsed.full_text, result)
```

- [ ] **Step 4: Run all loader tests**

Run: `pytest tests/integrations/test_langchain_loader.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add finlit/integrations/langchain/loader.py tests/integrations/test_langchain_loader.py
git commit -m "feat(langchain): on_error=include emits failure Documents

Matches the compliance-friendly PIPEDA case: every input file is
surfaced, failures carry finlit_error + finlit_error_type metadata
and an empty page_content the caller must filter before embedding."
```

---

## Task 10: `include_audit_log` opt-in

**Files:**
- Modify: `finlit/integrations/langchain/loader.py`
- Modify: `tests/integrations/test_langchain_loader.py`

- [ ] **Step 1: Update the existing snapshot test + add opt-in test**

Append to `tests/integrations/test_langchain_loader.py`:

```python
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
```

- [ ] **Step 2: Run and verify failure**

Run: `pytest tests/integrations/test_langchain_loader.py -v -k audit_log`
Expected: FAIL — `include_audit_log` kwarg does not exist yet.

- [ ] **Step 3: Implement the flag**

In `finlit/integrations/langchain/loader.py`:

(a) Extend `__init__` to accept and store the flag — add this parameter and body line:

```python
        on_error: OnError = "raise",
        include_audit_log: bool = False,
    ) -> None:
```

Then inside `__init__`, near the other stored attributes:

```python
        self._include_audit_log = include_audit_log
```

(b) Rewrite `_build_document` so it receives the flag and adds the key conditionally. Change its signature and the success-path call site:

```python
def _build_document(
    path: Path,
    full_text: str,
    result: ExtractionResult,
    include_audit_log: bool,
) -> Document:
    metadata: dict = {
        "source": str(path),
        "finlit_schema": result.schema_name,
        "finlit_model": result.extractor_model,
        "finlit_extraction_path": result.extraction_path,
        "finlit_needs_review": result.needs_review,
        "finlit_extracted_field_count": result.extracted_field_count,
        "finlit_fields": dict(result.fields),
        "finlit_confidence": dict(result.confidence),
        "finlit_source_ref": dict(result.source_ref),
        "finlit_warnings": list(result.warnings),
        "finlit_review_fields": list(result.review_fields),
        "finlit_pii_entities": list(result.pii_entities),
    }
    if include_audit_log:
        metadata["finlit_audit_log"] = list(result.audit_log)
    return Document(page_content=full_text, metadata=metadata)
```

Update the `lazy_load` success-path call to pass the flag:

```python
            yield _build_document(
                path, parsed.full_text, result, self._include_audit_log
            )
```

- [ ] **Step 4: Run all loader tests**

Run: `pytest tests/integrations/test_langchain_loader.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add finlit/integrations/langchain/loader.py tests/integrations/test_langchain_loader.py
git commit -m "feat(langchain): include_audit_log opt-in for metadata

Audit log can be large; default-omit keeps vector store payloads lean.
Set include_audit_log=True to get the full event stream per Document."
```

---

## Task 11: `last_results` sidecar with None-alignment on failures

**Files:**
- Modify: `tests/integrations/test_langchain_loader.py`

The loader already populates `last_results` correctly (a `result` on success, a `None` on "include" failures, and `skip` mode `continue`s without appending — verify this matches §5.4 of the spec; if `skip` also needs a `None` slot for alignment, fix it).

- [ ] **Step 1: Re-read spec §5.4**

The spec says: _"Failed paths (in "skip" or "include" mode) append `None` at the matching index so that `zip(docs, loader.last_results)` remains aligned."_

In Task 8 the `skip` branch uses `continue` without appending. This contradicts the spec — fix it.

- [ ] **Step 2: Write the failing alignment test**

Append to `tests/integrations/test_langchain_loader.py`:

```python
def test_last_results_includes_none_for_skipped_failures(
    t4_pipeline, patch_docling_parser, tmp_path, monkeypatch
):
    """In skip mode, last_results gets a None placeholder so indices
    align with the input path list. Yielded docs are still only the
    successes — the user pairs them with paths via last_results."""
    from finlit.integrations.langchain import FinLitLoader
    from finlit.result import ExtractionResult

    p1 = tmp_path / "ok1.pdf"; p1.write_bytes(b"x")
    p2 = tmp_path / "boom.pdf"; p2.write_bytes(b"x")
    p3 = tmp_path / "ok2.pdf"; p3.write_bytes(b"x")

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
```

- [ ] **Step 3: Run and verify failure**

Run: `pytest tests/integrations/test_langchain_loader.py::test_last_results_includes_none_for_skipped_failures -v`
Expected: FAIL — `last_results` is length 2 (only successes), missing the middle `None`.

- [ ] **Step 4: Fix the `skip` branch to append `None` before continuing**

In `finlit/integrations/langchain/loader.py`, update the `skip` branch inside `lazy_load`:

```python
                if self._on_error == "skip":
                    _log.warning(
                        "FinLit extraction failed for %s: %s", path, exc
                    )
                    self.last_results.append(None)
                    continue
```

- [ ] **Step 5: Run all loader tests**

Run: `pytest tests/integrations/test_langchain_loader.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add finlit/integrations/langchain/loader.py tests/integrations/test_langchain_loader.py
git commit -m "fix(langchain): align last_results with input paths in skip mode

last_results now appends None for skipped failures so that
loader.last_results[i] corresponds to self._paths[i]."
```

---

## Task 12: Import guard in `finlit.integrations.langchain.__init__`

**Files:**
- Modify: `finlit/integrations/langchain/__init__.py`
- Modify: `tests/integrations/test_langchain_loader.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/integrations/test_langchain_loader.py`:

```python
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

    with pytest.raises(ImportError, match="pip install finlit\\[langchain\\]"):
        import finlit.integrations.langchain  # noqa: F401
```

- [ ] **Step 2: Run and verify failure**

Run: `pytest tests/integrations/test_langchain_loader.py::test_import_without_langchain_core_raises_helpful_error -v`
Expected: FAIL — current `__init__` has no guard, so the raised error is the bare `ImportError: import of langchain_core halted` without the install hint.

- [ ] **Step 3: Add the guard**

Replace `finlit/integrations/langchain/__init__.py` with:

```python
"""LangChain integration for FinLit. Install with: pip install finlit[langchain]."""
try:
    from finlit.integrations.langchain.loader import FinLitLoader
except ImportError as exc:  # pragma: no cover - exercised via sys.modules patching
    raise ImportError(
        "finlit[langchain] extras not installed. "
        "Run: pip install finlit[langchain]"
    ) from exc

__all__ = ["FinLitLoader"]
```

- [ ] **Step 4: Run all loader tests**

Run: `pytest tests/integrations/ -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add finlit/integrations/langchain/__init__.py tests/integrations/test_langchain_loader.py
git commit -m "feat(langchain): wrap import in install-hint ImportError

Users who import FinLitLoader without the [langchain] extra installed
now see a clear message pointing at 'pip install finlit[langchain]'
instead of a generic langchain_core ModuleNotFoundError."
```

---

## Task 13: Example script

**Files:**
- Create: `examples/langchain_rag.py`

- [ ] **Step 1: Write the example**

Create `examples/langchain_rag.py`:

```python
"""End-to-end: load a T4 PDF via FinLitLoader, split, embed, query.

Requires: pip install finlit[langchain] langchain-openai langchain-chroma
Env: ANTHROPIC_API_KEY (for FinLit's Claude extractor), OPENAI_API_KEY
     (for embeddings).
"""
from __future__ import annotations

import sys
from pathlib import Path

from finlit.integrations.langchain import FinLitLoader


def main(paths: list[str]) -> None:
    # Load in batch with on_error='include' so compliance teams see every
    # file that was submitted, even failed ones.
    loader = FinLitLoader(paths, schema="cra.t4", on_error="include")
    docs = loader.load()

    # Filter out failure Documents before feeding an embedder (empty
    # page_content will cause some embedders to error out).
    good_docs = [d for d in docs if not d.metadata.get("finlit_error")]
    print(f"Loaded {len(good_docs)} successful, {len(docs) - len(good_docs)} failed")

    # Structured field access — no embedding needed for this kind of query
    for d in good_docs:
        fields = d.metadata["finlit_fields"]
        needs_review = d.metadata["finlit_needs_review"]
        print(
            f"{Path(d.metadata['source']).name}: "
            f"{fields.get('employer_name')!r} "
            f"income={fields.get('box_14_employment_income')} "
            f"review={needs_review}"
        )

    # The same Documents are ready for a vector store. Uncomment to run
    # a real RAG pipeline (requires the extra deps above):
    #
    # from langchain_openai import OpenAIEmbeddings
    # from langchain_chroma import Chroma
    # from langchain_text_splitters import RecursiveCharacterTextSplitter
    #
    # splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    # chunks = splitter.split_documents(good_docs)
    # store = Chroma.from_documents(chunks, OpenAIEmbeddings())
    # hits = store.similarity_search("how much CPP did Acme withhold?")
    # for h in hits:
    #     print(h.page_content, h.metadata["finlit_fields"].get("box_16_cpp_contributions"))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python examples/langchain_rag.py <t4.pdf> [<t4.pdf> ...]")
        sys.exit(1)
    main(sys.argv[1:])
```

- [ ] **Step 2: Confirm the example imports cleanly**

Run: `python -c "import ast; ast.parse(open('examples/langchain_rag.py').read()); print('ok')"`
Expected: `ok`.

- [ ] **Step 3: Commit**

```bash
git add examples/langchain_rag.py
git commit -m "docs(examples): add langchain_rag.py end-to-end example

Shows the one-liner load pattern, on_error='include' for compliance,
structured metadata filtering, and a commented-out RAG flow over
Chroma + OpenAI embeddings."
```

---

## Task 14: README updates

**Files:**
- Modify: `README.md` (roadmap section and usage section)

- [ ] **Step 1: Update the roadmap line**

Open `README.md`, locate `- [ ] LangChain and LlamaIndex reader integrations` (currently `README.md:454`), and replace that single line with:

```
- [x] LangChain reader integration
- [ ] LlamaIndex reader integration
```

- [ ] **Step 2: Add a usage subsection**

Find the "Usage" (or "Getting started") section of `README.md`. After the existing pipeline usage example, insert a new subsection:

```markdown
### LangChain integration

FinLit ships a LangChain `BaseLoader` so you can drop extracted Canadian
financial documents straight into RAG pipelines, retrievers, and agents.

Install the extra:

```bash
pip install finlit[langchain]
```

Load one file:

```python
from finlit.integrations.langchain import FinLitLoader

docs = FinLitLoader("t4.pdf", schema="cra.t4").load()
doc = docs[0]
print(doc.metadata["finlit_fields"]["employer_name"])      # "Acme Corp"
print(doc.metadata["finlit_needs_review"])                  # False
```

Batch load with compliance-friendly error surfacing:

```python
loader = FinLitLoader(
    ["t4_001.pdf", "t4_002.pdf", "t4_003.pdf"],
    schema="cra.t4",
    on_error="include",  # failures become Documents with finlit_error
)
docs = loader.load()
# Filter out failures before embedding — empty page_content breaks most embedders.
good = [d for d in docs if not d.metadata.get("finlit_error")]
```

Access the underlying `ExtractionResult` objects via `loader.last_results`
(same order as the input paths, with `None` for skipped/included failures).
```

- [ ] **Step 3: Verify the file still renders**

Run: `python -c "import pathlib; content = pathlib.Path('README.md').read_text(); assert '[x] LangChain reader integration' in content; assert '### LangChain integration' in content; print('ok')"`
Expected: `ok`.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs(readme): add LangChain integration usage + flip roadmap

Splits the old combined roadmap bullet: LangChain reader ships now,
LlamaIndex reader remains on the roadmap. New usage subsection covers
one-liner load, batch with on_error='include', and last_results."
```

---

## Task 15: Final verification

**Files:** none modified — verification only.

- [ ] **Step 1: Run the full integration test suite**

Run: `pytest tests/integrations/ -v`
Expected: every test PASSes. Count should match:
- `test_schema_resolver.py`: 6 tests
- `test_langchain_loader.py`: at least 13 tests (single_file, list_of_paths, lazy_load_generator, metadata_contract_snapshot, schema_kwarg, pipeline_wins, missing_both_raises, on_error_raise, on_error_skip, on_error_include, audit_log_default, audit_log_flag, last_results_none_alignment, last_results_resets, import_guard)

Total: ≥19 tests.

- [ ] **Step 2: Run the full project test suite**

Run: `pytest tests/ -v`
Expected: every test PASSes, including the pre-existing suite (tests/test_pipeline.py, test_schema.py, test_pii.py, test_validator.py). Count should be roughly baseline + ≥19.

- [ ] **Step 3: Run ruff**

Run: `ruff check finlit/ tests/`
Expected: `All checks passed.`

If ruff flags anything in the new files, fix inline and re-run before committing.

- [ ] **Step 4: Run mypy**

Run: `mypy finlit/`
Expected: `Success: no issues found`. If mypy is not pinned tightly enough to catch the new annotations, at a minimum the new modules must not introduce new errors beyond the baseline.

- [ ] **Step 5: Confirm the bare install still works**

Run: `pip install -e . --no-deps --force-reinstall --quiet && python -c "import finlit; print(finlit.__version__)"`
Expected: prints version. Loader module must not have been pulled in by the bare install.

- [ ] **Step 6: Confirm the extra install works**

Run: `pip install -e ".[langchain]" --quiet && python -c "from finlit.integrations.langchain import FinLitLoader; print(FinLitLoader.__doc__.splitlines()[0])"`
Expected: prints the first line of the class docstring.

- [ ] **Step 7: Commit any lint-fix edits (if there were any)**

If Steps 3 or 4 required fixes, commit them:

```bash
git add -u
git commit -m "chore(langchain): fix ruff/mypy findings from final verification"
```

If no fixes were needed, skip this step.

- [ ] **Step 8: Final status check**

Run: `git log --oneline main...HEAD -- finlit/integrations finlit/integrations/langchain tests/integrations examples/langchain_rag.py README.md pyproject.toml | wc -l`
Expected: ≥ 14 (one per task that committed).

Run: `git status`
Expected: `nothing to commit, working tree clean`.

The feature is ready for review / PR.

---

## Summary

14 tasks, each bounded and test-driven. The plan follows the spec section-by-section:

| Spec § | Task(s) |
|---|---|
| 4 (public API) | 3, 4, 5, 6, 7 |
| 5 (Document shape) | 3, 4 |
| 5.4 (sidecar) | 11 |
| 6 (error handling) | 8, 9 |
| 7 (packaging + import guard) | 1, 12 |
| 8 (testing) | 3–12, 15 |
| 9 (docs) | 13, 14 |
| 12 (acceptance criteria) | 15 |
