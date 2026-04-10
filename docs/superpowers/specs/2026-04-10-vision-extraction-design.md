# FinLit v0.3.0 — Vision-Based Extraction Design

**Date:** 2026-04-10
**Status:** Design approved, awaiting implementation plan
**Target version:** v0.3.0

---

## 1. Goal and Non-Goals

### Goal

Add a model-agnostic vision extraction path that lets consumers plug in any multimodal LLM (commercial or open-source), triggered as an opt-in fallback from the existing text extraction pipeline.

This fixes the two silent-failure modes observed in v0.2.0:

1. **Image-only PDFs** (e.g., `T4.pdf`, `T4 Sybte 2020.pdf`) where Docling's OCR recovers fewer than 100 characters and the pipeline correctly flags `sparse_document` but has no path forward.
2. **Native-text forms with flattened layouts** (e.g., `T5_2024_Slip1_Srivatsa_Kasagar.pdf`) where Docling's `export_to_markdown()` destroys the 2D column alignment of a tax slip, and the LLM cannot reliably map box numbers to values — producing low-confidence, often-incorrect extractions.

### Non-Goals

- **Not replacing the text path.** Text-first remains the default because it is ~15× cheaper per page than vision. Vision is opt-in fallback only.
- **Not hardcoding any vendor.** FinLit ships sensible defaults (`claude-sonnet-4-6`); every default is overridable. Consumers can use Claude, OpenAI, Gemini, Ollama-hosted open-source models, or their own `BaseVisionExtractor` subclass.
- **Not building an arbitrary extraction strategy chain.** Exactly one fallback step: text → vision. YAGNI for multi-step chains.
- **Not bundling an OSS model.** FinLit stays a library. Consumers choose their runtime (Ollama, vLLM, HuggingFace) and bring their own weights. No `torch` or `transformers` added to the dependency tree.
- **Not doing image preprocessing** (deskew, denoise, contrast adjustment). Modern vision models handle that internally; consumers with exceptional preprocessing needs write their own extractor.
- **Not supporting audio, table-as-input, or other multimodal input types.** Images only in v0.3.
- **Not adding automatic retry logic.** One vision attempt per document. Retries are the consumer's responsibility (via pydantic-ai model settings or a custom `BaseVisionExtractor`).
- **Not merging text and vision results.** When vision runs, vision wins — full replacement semantics. See Section 3 for rationale.

---

## 2. Architecture

Five new/changed units, each with one clear responsibility.

### 2.1 New file: `finlit/parsers/image_renderer.py`

Render PDFs and image files to PNG bytes.

**Public API:**
```python
def render_pages(path: Path, dpi: int = 200) -> list[bytes]:
    """Render the pages of a document to PNG bytes.

    PDFs are rasterized via pypdfium2 at the requested DPI.
    Image files (.png, .jpg, .jpeg) are returned as a single-element
    list containing the original file bytes — no re-encoding.

    Raises:
        FileNotFoundError: path does not exist
        ValueError: file extension is not supported
    """
```

**Dependencies:** `pypdfium2` (already in the dependency tree via Docling), `PIL` (already a Docling transitive dep).

**Design notes:**
- Zero coupling to Docling, pipeline, or any extractor. Standalone module that could be used by a consumer directly.
- 200 DPI chosen as the default: 72 DPI is too low for small CRA box numbers, 300+ DPI blows up image token cost without measurable accuracy gains on forms.
- PNG chosen as the default format: lossless, universally supported by vision models, appropriate for forms with sharp text.

### 2.2 New file: `finlit/extractors/base_vision.py`

Abstract base class for vision extractors.

**Public API:**
```python
class BaseVisionExtractor(ABC):
    """Abstract base for extractors that consume page images.

    Parallel to BaseExtractor (which consumes text). A separate ABC is
    used instead of extending BaseExtractor to preserve type safety:
    DocumentPipeline's `vision_extractor` parameter is typed
    `BaseVisionExtractor | None`, preventing accidental misuse.
    """

    @abstractmethod
    def extract(
        self,
        images: list[bytes],
        schema: Schema,
        text: str = "",
    ) -> ExtractionOutput:
        """Extract structured fields from page images.

        Args:
            images: list of PNG-encoded page images, one per page
            schema: the FinLit Schema to extract against
            text: optional text hint (e.g., whatever the text path
                managed to recover, for use as additional context)
        """
```

**Design notes:**
- The `text=""` parameter is deliberately optional. Most implementations will ignore it, but it is available for vision extractors that want to use the text path's partial output as additional context (e.g., "the text extractor found 'Acme Corp' as the employer name with 99% confidence; verify from the image").
- Separate ABC over extending `BaseExtractor` chosen for three reasons: type safety (compiler catches "you passed a text extractor as `vision_extractor=`"), zero backwards-compat risk (existing `BaseExtractor` subclasses untouched), clarity (a consumer reading `BaseVisionExtractor` sees exactly what "vision" means in FinLit).

### 2.3 New file: `finlit/extractors/vision_extractor.py`

Default pydantic-ai–based implementation of `BaseVisionExtractor`.

**Public API:**
```python
class VisionExtractor(BaseVisionExtractor):
    def __init__(
        self,
        model: str = "anthropic:claude-sonnet-4-6",
        dpi: int = 200,
        image_format: str = "png",
        max_pages: int | None = None,
    ):
        """Vision extractor backed by pydantic-ai.

        Args:
            model: any pydantic-ai model string for a multimodal model.
                Examples:
                  - "anthropic:claude-sonnet-4-6"    (default)
                  - "openai:gpt-4o"
                  - "google-gla:gemini-2.0-flash"
                  - "ollama:llama3.2-vision"         (fully local OSS)
                  - "ollama:qwen2.5vl:7b"            (fully local OSS)
                  - "ollama:minicpm-v"               (fully local OSS)
                Non-multimodal models will raise at extraction time.
            dpi: DPI used when rendering PDFs to images.
            image_format: "png" (default). Reserved for future formats.
            max_pages: hard cap on page count per document. If a document
                has more pages than max_pages, extract() raises ValueError
                before any LLM call. None (default) disables the cap.
        """
```

**Implementation outline:**
- Wraps a pydantic-ai `Agent` with `output_type=ExtractionOutput` and a vision-specific system prompt.
- Builds the user message as a list of `[prompt_text, BinaryContent(data=img, media_type="image/png"), ...]` parts, one `BinaryContent` per page.
- System prompt explicitly frames the task: "You are looking at a scanned or rendered page of a Canadian financial document. Extract the fields defined in the schema by reading the visible layout. Pay attention to box numbers and their adjacent values — the spatial relationship between a box label and its value is the primary signal. Do not infer values from context if they are not visible in the image."

### 2.4 Modified file: `finlit/pipeline.py`

Adds vision fallback orchestration.

**New parameters on `DocumentPipeline.__init__`:**
```python
vision_extractor: BaseVisionExtractor | None = None,
vision_fallback_when: Callable[[ExtractionResult], bool] | None = None,
```

**Default callback:**
```python
# When vision_fallback_when is None but vision_extractor is not None,
# the pipeline uses this default policy:
lambda result: result.needs_review
```

**New flow in `run()`:**

The existing text extraction flow runs to completion exactly as in v0.2.0. After the provisional `ExtractionResult` is built, one new decision step is inserted before returning:

```
1. Run text path to completion (parse → OCR fallback → PII → extract → validate → review → warnings)
2. Build provisional ExtractionResult (extraction_path="text")
3. IF vision_extractor is None:
       return provisional result
4. Evaluate fallback callback:
       callback = vision_fallback_when or (lambda r: r.needs_review)
       try:
           should_fire = callback(provisional_result)
       except Exception:
           log "vision_fallback_callback_error"
           add warning "vision_fallback_failed" (reason="callback")
           return provisional result
5. IF should_fire is False:
       return provisional result
6. Log "vision_fallback_triggered"
7. Render page images:
       try:
           images = render_pages(path, dpi=vision_extractor.dpi)
       except Exception as e:
           log "vision_render_failed"
           add warning "vision_fallback_failed" (reason="render")
           return provisional result
8. Call vision extractor:
       try:
           log "vision_extraction_start"
           vision_output = vision_extractor.extract(images, schema, text=parsed.full_text)
           log "vision_extraction_complete"
       except Exception as e:
           log "vision_extraction_failed"
           add warning "vision_fallback_failed" (reason="extraction")
           return provisional result
9. Re-validate vision output with the same FieldValidator
10. Build new ExtractionResult with extraction_path="vision"
11. return vision result (replaces provisional text result)
```

**Design notes:**
- **Render-on-demand.** `render_pages()` is called lazily only when the fallback actually fires. Consumers who never trigger vision pay zero rendering cost.
- **Full-replacement semantics.** When vision runs and succeeds, the vision result completely replaces the text result. No field-level merging. Rationale: merging is speculative (which result do you trust per field?), bad vision results will naturally have `needs_review=True` from validation so the consumer still knows something is off, and a simple rule is easier to audit.
- **Fail-safe fallback.** Any failure in the vision path (render, extraction, callback) falls back to the text result with an added warning. The pipeline never crashes because vision is misconfigured.

### 2.5 Modified file: `finlit/result.py`

Add one field to `ExtractionResult`:

```python
extraction_path: str = "text"  # values: "text" or "vision"
```

**Design notes:**
- Additive with a default value — zero breaking change for existing consumers.
- Lets tests assert on which path ran and lets consumers programmatically detect vision usage (e.g., for cost tracking dashboards).

### 2.6 Data flow diagram

```
PDF or image file
    ↓
DoclingParser.parse() → ParsedDocument (text, metadata)
    ↓
sparse check → OCR retry → sparse warning         (unchanged from v0.2.0)
    ↓
PII scan on full_text                             (unchanged)
    ↓
text_extractor.extract(text, schema)              (unchanged)
    ↓
FieldValidator.validate()                         (unchanged)
    ↓
Build provisional ExtractionResult                (unchanged)
    ↓
vision_extractor provided?
    ├── NO → return provisional (extraction_path="text")
    └── YES ↓
        vision_fallback_when(provisional_result)?
            ├── False/error → return provisional + warning
            └── True ↓
                render_pages(path, dpi)
                    ↓ (on error → return provisional + warning)
                vision_extractor.extract(images, schema, text=...)
                    ↓ (on error → return provisional + warning)
                FieldValidator.validate()
                    ↓
                Build new ExtractionResult (extraction_path="vision")
                    ↓
                return vision result
```

### 2.7 What this deliberately does NOT change

- `BaseExtractor` ABC and `PydanticAIExtractor` — untouched. Zero breaking change to existing text extractors.
- `DoclingParser` — untouched.
- `FieldValidator` — untouched; reused for both paths.
- `AuditLog` — new event types (`vision_fallback_triggered`, etc.) but no API change.
- `ExtractionResult` — one additive field with a default.
- Public API additions only: `VisionExtractor` and `BaseVisionExtractor` exported from `finlit/__init__.py`.

---

## 3. Error Handling and Edge Cases

### 3.1 Failure modes in the vision path

**F1. `render_pages()` fails** (corrupted PDF, unsupported format, memory pressure on huge files)
- Catch at the pipeline level.
- Audit: `vision_render_failed` with exception type and message.
- Warning: `{"code": "vision_fallback_failed", "reason": "render", "message": "..."}` appended to the text result.
- Return the text result. `needs_review` stays True because the text result already flagged it.

**F2. `vision_extractor.extract()` raises** (missing API key, rate limit, network error, malformed output, non-multimodal model passed to a vision endpoint)
- Same pattern as F1.
- Audit: `vision_extraction_failed`.
- Warning: `{"code": "vision_fallback_failed", "reason": "extraction", "message": "..."}`.
- Return the text result. No automatic retry.

**F3. Vision returns garbage** (extractor ran successfully but produced nonsense — all nulls, invalid values, malformed fields)
- No special handling. The result goes through `FieldValidator` exactly like text output. If required fields are missing or regex fails, the vision result will have `required_fields_missing` warning and `needs_review=True`, and the consumer sees the usual compliance signals.
- We do not compare vision to text and pick the "better" one. Vision wins if it ran.

**F4. `vision_fallback_when(result)` callback raises**
- Audit: `vision_fallback_callback_error` with exception details.
- Warning: `{"code": "vision_fallback_failed", "reason": "callback", "message": "..."}`.
- Return the text result. The pipeline must not crash because of a consumer's buggy callback.

### 3.2 Edge cases with explicit tests

**E1. Consumer passes `vision_extractor=` but no `vision_fallback_when=`.**
Use the default callback (`result.needs_review`). Document clearly in the docstring.

**E2. Consumer passes `vision_fallback_when=` but no `vision_extractor=`.**
Silently ignored. The callback is never evaluated. Not an error.

**E3. Image file input (`.png`, `.jpg`) instead of PDF.**
`render_pages()` returns `[file_bytes]` as a single-element list without re-encoding. Works identically through the rest of the pipeline.

**E4. Schema has zero required fields.**
The default `needs_review` callback still fires correctly because `needs_review` triggers on `review_fields` OR `warnings`, not only on missing required fields.

**E5. Document has more pages than `max_pages`.**
`VisionExtractor.extract()` raises `ValueError` before any LLM call. Caught by F2's handler. Default `max_pages=None` disables the cap, so this only fires when the consumer explicitly sets a limit.

**E6. `sparse_document` warning triggered by text path + vision extractor provided.**
This is the ideal path for image-only PDFs. Text path adds `sparse_document` warning → `needs_review=True` → default callback fires → vision runs on rendered images → final result is the vision result (with `extraction_path="vision"`). The `sparse_document` warning from the text path is NOT carried over because the final returned result is a brand-new object built from the vision output. This is intentional: the final result reflects the path that actually ran.

**E7. Text path succeeds cleanly and vision extractor is provided.**
Default callback returns `False` (no review needed). Vision extractor is not called — consumers pay zero vision cost on healthy documents.

### 3.3 Audit event catalog

Every vision attempt produces this sequence of events in the audit log:

| Event | When logged | Payload |
|---|---|---|
| `vision_fallback_triggered` | Callback returned True | `{provisional_needs_review, provisional_warning_codes}` |
| `vision_render_start` | Before calling `render_pages()` | `{dpi, path}` |
| `vision_render_complete` | After successful render | `{page_count, total_bytes}` |
| `vision_render_failed` | Render exception | `{error_type, error_message}` |
| `vision_extraction_start` | Before calling `vision_extractor.extract()` | `{model, page_count}` |
| `vision_extraction_complete` | After successful extraction | `{fields_returned}` |
| `vision_extraction_failed` | Extraction exception | `{error_type, error_message}` |
| `vision_fallback_callback_error` | Callback raised | `{error_type, error_message}` |

This gives consumers a complete forensic trail of every decision the pipeline made, preserving FinLit's compliance story.

---

## 4. Testing Strategy

**Hard constraints:**
- Zero network access in the test suite.
- Zero API keys required.
- No real LLM calls, ever.
- Tests must pass in CI with no secrets configured.

### 4.1 New test file: `tests/test_image_renderer.py`

Pure unit tests on `render_pages()`. Uses tiny in-memory or `tmp_path` fixtures.

| # | Test | What it proves |
|---|---|---|
| R1 | `test_render_pdf_returns_png_bytes` | PDF → list of PNG byte strings, each starts with `\x89PNG` magic |
| R2 | `test_render_respects_dpi` | 200 DPI output is larger in bytes than 72 DPI output for same PDF (parameter is wired) |
| R3 | `test_render_multipage_pdf` | 3-page PDF → list of length 3 |
| R4 | `test_render_png_input_passthrough` | `.png` file → `[file_bytes]` unchanged |
| R5 | `test_render_jpg_input_passthrough` | `.jpg` file → `[file_bytes]` unchanged |
| R6 | `test_render_file_not_found_raises` | Missing path → `FileNotFoundError` |
| R7 | `test_render_unsupported_format_raises` | `.txt` → `ValueError` |

No mocking needed. `pypdfium2` and `PIL` are pure-Python, fast, and deterministic.

### 4.2 New test file: `tests/test_vision_extractor.py`

Tests `VisionExtractor` using pydantic-ai's `TestModel` (canned outputs, zero network).

| # | Test | What it proves |
|---|---|---|
| V1 | `test_vision_extractor_calls_agent_with_images` | Agent receives `BinaryContent` parts with `media_type="image/png"` |
| V2 | `test_vision_extractor_returns_extraction_output` | TestModel returns valid ExtractionOutput → extract() returns it unchanged |
| V3 | `test_vision_extractor_passes_text_hint` | `text=` parameter appears in the prompt sent to the model |
| V4 | `test_vision_extractor_default_model_is_claude` | Constructor default is `"anthropic:claude-sonnet-4-6"` |
| V5 | `test_vision_extractor_custom_dpi_stored` | `VisionExtractor(dpi=300).dpi == 300` |
| V6 | `test_vision_extractor_max_pages_enforced` | 3 images + `max_pages=2` → `ValueError` before any LLM call |

### 4.3 Additions to `tests/test_pipeline.py`

Integration tests for the fallback orchestration. All use `StubVisionExtractor` (added to `conftest.py`) which returns canned outputs with zero network.

| # | Test | What it proves |
|---|---|---|
| P1 | `test_vision_fallback_fires_when_callback_returns_true` | Text all-None + vision valid → final result is vision, `extraction_path=="vision"`, audit has `vision_fallback_triggered` + `vision_extraction_complete` |
| P2 | `test_vision_fallback_skipped_when_callback_returns_false` | Text valid + default callback → vision NOT called (stub call counter), `extraction_path=="text"` |
| P3 | `test_vision_fallback_skipped_when_vision_extractor_not_provided` | Text all-None + `vision_extractor=None` → text result unchanged, no vision events |
| P4 | `test_vision_fallback_custom_callback` | `vision_fallback_when=lambda r: True` overrides default and forces vision |
| P5 | `test_vision_render_failure_falls_back_to_text_result` | `render_pages` patched to raise → text result returned + warning `vision_fallback_failed` reason=`render` |
| P6 | `test_vision_extraction_failure_falls_back_to_text_result` | Stub raises RuntimeError → text result + warning reason=`extraction` |
| P7 | `test_vision_callback_exception_falls_back_to_text_result` | Callback raises → pipeline does not crash, text result + warning reason=`callback` |
| P8 | `test_vision_result_replaces_text_result_fully` | Text `{employer_name: "WRONG"}` + vision `{employer_name: "CORRECT"}` → final is `"CORRECT"` |
| P9 | `test_vision_audit_trail_complete` | Successful vision run produces the full audit event sequence in order |

### 4.4 New conftest fixtures

```python
class StubVisionExtractor(BaseVisionExtractor):
    """Deterministic vision extractor for tests — zero network."""
    def __init__(self, fields=None, confidence=None, raises=None):
        self.fields = fields or {}
        self.confidence = confidence or {}
        self.raises = raises
        self.call_count = 0
        self.last_images = None
        self.last_text = None

    def extract(self, images, schema, text=""):
        self.call_count += 1
        self.last_images = images
        self.last_text = text
        if self.raises:
            raise self.raises
        return ExtractionOutput(fields=self.fields, confidence=self.confidence)
```

Plus a `high_confidence_vision_t4_extractor` fixture parallel to the existing `high_confidence_t4_extractor`.

### 4.5 Manual verification plan (outside the automated suite)

After automated tests pass, run these four real-world documents using actual LLM calls, document results in a PR comment (NOT committed to the repo because they contain real PII):

| Document | Expected result |
|---|---|
| `T4.pdf` (image-only PDF) | Text path fails with `sparse_document`, vision fallback fires, vision extracts T4 fields correctly |
| `T5_2024_Slip1_Srivatsa_Kasagar.pdf` (native text, flattened layout) | Text path extracts wrong boxes with low confidence, vision fallback fires, vision correctly maps values to box numbers |
| `cra_example_t5_albert_chang.jpg` (CRA ground-truth example) | Both paths reach 12/12; confirms vision does not regress on documents that already worked |
| `T5_2024_Slip1` with `vision_extractor=VisionExtractor(model="ollama:qwen2.5vl:7b")` | Fully local OSS run succeeds, proves the open-source story works end-to-end |

The fourth test is critical: if OSS does not work with the shipped design, we have not actually delivered "model-agnostic."

### 4.6 Test count target

- v0.2.0: 45 tests
- v0.3.0 target: ~67 tests (7 renderer + 6 vision extractor + 9 pipeline integration = 22 new)

---

## 5. Public API and Consumer Ergonomics

### 5.1 New exports from `finlit/__init__.py`

```python
from finlit.extractors.base_vision import BaseVisionExtractor
from finlit.extractors.vision_extractor import VisionExtractor

__all__ = [
    "DocumentPipeline",
    "BatchPipeline",
    "Schema",
    "Field",
    "ExtractionResult",
    "schemas",
    # new in v0.3:
    "VisionExtractor",
    "BaseVisionExtractor",
]
```

Nothing removed. Nothing renamed.

### 5.2 Consumer usage patterns

**Pattern 1: Default — Claude for both text and vision**
```python
from finlit import DocumentPipeline, VisionExtractor, schemas

pipeline = DocumentPipeline(
    schema=schemas.CRA_T5,
    extractor="claude",
    vision_extractor=VisionExtractor(),  # defaults to claude-sonnet-4-6
)
result = pipeline.run("slip.pdf")
```

**Pattern 2: Mixed providers — cheap text, premium vision**
```python
pipeline = DocumentPipeline(
    schema=schemas.CRA_T5,
    extractor="openai:gpt-4o-mini",
    vision_extractor=VisionExtractor(
        model="anthropic:claude-sonnet-4-6",
        dpi=250,
    ),
)
```

**Pattern 3: Custom policy — only use vision when confidence is low**
```python
def fallback_policy(result):
    if not result.confidence:
        return True
    avg_conf = sum(result.confidence.values()) / len(result.confidence)
    return avg_conf < 0.80

pipeline = DocumentPipeline(
    schema=schemas.CRA_T5,
    extractor="claude",
    vision_extractor=VisionExtractor(model="google-gla:gemini-2.0-flash"),
    vision_fallback_when=fallback_policy,
)
```

**Pattern 4: Fully local with OSS — zero API keys, zero network**
```python
pipeline = DocumentPipeline(
    schema=schemas.CRA_T5,
    extractor="ollama:llama3.2",
    vision_extractor=VisionExtractor(model="ollama:qwen2.5vl:7b"),
)
```

**Pattern 5: Consumer's own vision extractor (on-prem, fine-tuned, internal compliance gateway)**
```python
from finlit import BaseVisionExtractor, DocumentPipeline, schemas

class MyBankVisionExtractor(BaseVisionExtractor):
    def extract(self, images, schema, text=""):
        # call internal multimodal model
        return ExtractionOutput(fields=..., confidence=...)

pipeline = DocumentPipeline(
    schema=schemas.CRA_T5,
    extractor="ollama:llama3.2",
    vision_extractor=MyBankVisionExtractor(),
)
```

### 5.3 CLI support

New flag on `finlit extract`:

```bash
finlit extract slip.pdf \
    --schema cra.t5 \
    --extractor claude \
    --vision-extractor claude-sonnet-4-6
```

When `--vision-extractor` is set, the CLI constructs a `VisionExtractor` with the specified model string and wires it into the pipeline with the default callback. No CLI flag for custom callbacks — that is a Python API feature.

### 5.4 Documentation deliverables (shipped with v0.3 implementation)

1. **New README section: "Running fully locally with OSS models"** — lists tested Ollama model strings (`llama3.2-vision`, `qwen2.5vl:7b`, `minicpm-v`), states that any pydantic-ai–compatible multimodal model works, links to Ollama installation docs.
2. **New README section: "When to use vision extraction"** — includes the cost/accuracy trade-off table and the five usage patterns above.
3. **Updated docstring on `DocumentPipeline.__init__`** — documents `vision_extractor` and `vision_fallback_when` parameters.
4. **Loud docstring on `VisionExtractor`** — states the multimodal-model requirement explicitly, lists tested models.
5. **New example file: `examples/extract_with_vision.py`** — minimal runnable example with Claude vision.
6. **New example file: `examples/extract_with_local_vision.py`** — minimal runnable example with Ollama + Qwen2.5-VL. Proves the zero-API-keys story.
7. **Updated Roadmap in README** — mark v0.3 items shipped (✅) and move "Vision-based extraction" from unchecked to checked.

All README changes land in the same commit as the code, as the final task in the implementation plan. No documentation ships ahead of working code.

### 5.5 Backwards compatibility guarantee

Zero breaking changes. A consumer on v0.2.0 upgrading to v0.3.0 with no code changes sees **identical behavior**:

- Both new parameters on `DocumentPipeline.__init__` default to `None`.
- Both new exports in `finlit/__init__.py` are additive.
- The `extraction_path` field on `ExtractionResult` defaults to `"text"`.
- No method signatures changed, no existing field semantics changed, no existing class renamed.

---

## 6. Dependency Changes

**No new dependencies required.**

- `pypdfium2` — already in the dep tree via Docling.
- `PIL` — already in the dep tree via Docling.
- `pydantic-ai` — already the text extractor backend; `BinaryContent` is an existing feature.
- Ollama / vLLM / HuggingFace providers — not FinLit deps; consumers install their own runtime.

---

## 7. Open Questions (resolved during brainstorming)

| Question | Resolution |
|---|---|
| Scope — minimal, fallback, or vision-first? | Fallback (option B from brainstorm Q1) |
| API shape — second parameter, auto-detect, or chain? | Second parameter `vision_extractor` + callback (options A + D) |
| Fallback trigger policy? | Consumer-controlled callback, defaults to `result.needs_review` |
| Page rendering approach? | `pypdfium2` directly, independent of Docling |
| Image format / DPI defaults? | PNG, 200 DPI, both overridable |
| `BaseExtractor` ABC — extend or separate? | Separate `BaseVisionExtractor` ABC |
| Merge text + vision results? | No — full replacement if vision runs |
| Automatic retry on vision failure? | No — consumer's responsibility |
| Ship an OSS model bundle? | No — consumer brings their own runtime |
| README update before v0.3 ships? | No — README only documents shipped features |

---

## 8. Success Criteria

The v0.3.0 release is complete when:

1. All ~67 automated tests pass in CI with no API keys.
2. `T4.pdf` (image-only) is correctly extracted via the vision fallback path.
3. `T5_2024_Slip1_Srivatsa_Kasagar.pdf` is correctly extracted with Box 13 interest properly assigned (fixing the v0.2.0 box-swap failure).
4. `cra_example_t5_albert_chang.jpg` still reaches 12/12 extraction (no regression).
5. `Ollama + Qwen2.5-VL` produces a working extraction on at least one test document — proves the OSS story.
6. Zero breaking changes: all v0.2.0 tests still pass unchanged.
7. README updated with vision extraction section, OSS callout, and updated roadmap — in the same commit as the code.

---

*Design brainstormed 2026-04-10. Implementation plan to follow via the writing-plans skill.*
