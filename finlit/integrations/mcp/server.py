"""FinLit MCP server - FastMCP app + tool registrations + serve() entry point.

The module exposes:

  - build_app(...)  - build a FastMCP app with the given server-startup config.
                       Pure construction; no I/O. Used by tests.
  - serve(...)      - build_app + run stdio. The CLI and __main__ launchers
                       both call this.
"""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

from mcp.server.fastmcp import FastMCP

from finlit.audit.pii import CanadianPIIDetector
from finlit.integrations._schema_resolver import _DOTTED_TO_ATTR, _resolve_schema
from finlit.integrations.mcp.pipeline_cache import get_pipeline
from finlit.integrations.mcp.responses import build_extraction_response

PIIMode = Literal["redact", "raw"]


def build_app(
    *,
    extractor: str,
    vision_extractor: str | None,
    review_threshold: float,
    pii_mode: PIIMode,
) -> FastMCP:
    """Construct a FastMCP app with the given server-startup configuration."""
    app = FastMCP("finlit")

    # Server-startup config is captured in the closures below.
    server_default_redact = pii_mode == "redact"

    @app.tool()
    def list_schemas() -> list[dict]:
        """List all built-in FinLit schemas with field counts and required fields."""
        out = []
        for dotted_key in sorted(_DOTTED_TO_ATTR):
            schema = _resolve_schema(dotted_key)
            out.append({
                "key": dotted_key,
                "name": schema.document_type or schema.name,
                "version": schema.version,
                "field_count": len(schema.fields),
                "required_fields": [f.name for f in schema.fields if f.required],
                "description": schema.description,
            })
        return out

    @app.tool()
    async def extract_document(
        path: str,
        schema: str,
        extractor_override: str | None = None,
        vision_extractor_override: str | None = None,
        redact_pii: bool | None = None,
        include_audit_log: bool = False,
        include_source_ref: bool = False,
        include_pii_entities: bool = False,
    ) -> dict:
        """Extract structured fields from a single Canadian financial document."""
        doc_path = Path(path)
        if not doc_path.exists():
            raise ValueError(f"path does not exist: {path}")

        chosen_extractor = extractor_override or extractor
        chosen_vision = (
            vision_extractor_override
            if vision_extractor_override is not None
            else vision_extractor
        )
        effective_redact = (
            redact_pii if redact_pii is not None else server_default_redact
        )

        pipeline = get_pipeline(
            chosen_extractor, chosen_vision, schema, review_threshold,
        )

        # Run the sync pipeline in a thread so the event loop stays responsive.
        result = await asyncio.to_thread(pipeline.run, doc_path)

        return build_extraction_response(
            result=result,
            schema=pipeline.schema,
            schema_key=schema,
            document_path=str(doc_path.resolve()),
            redact=effective_redact,
            include_audit_log=include_audit_log,
            include_source_ref=include_source_ref,
            include_pii_entities=include_pii_entities,
        )

    @app.tool()
    async def batch_extract(
        paths: list[str],
        schema: str,
        extractor_override: str | None = None,
        vision_extractor_override: str | None = None,
        redact_pii: bool | None = None,
        on_error: Literal["raise", "skip", "include"] = "raise",
        max_workers: int | None = None,
        include_audit_log: bool = False,
        include_source_ref: bool = False,
        include_pii_entities: bool = False,
    ) -> dict:
        """Extract from many documents in parallel; returns aligned results + errors."""
        if on_error not in ("raise", "skip", "include"):
            raise ValueError(
                f"on_error must be 'raise', 'skip', or 'include', got {on_error!r}"
            )

        doc_paths = [Path(p) for p in paths]
        for i, p in enumerate(doc_paths):
            if not p.exists():
                raise ValueError(f"paths[{i}] does not exist: {p}")

        chosen_extractor = extractor_override or extractor
        chosen_vision = (
            vision_extractor_override
            if vision_extractor_override is not None
            else vision_extractor
        )
        effective_redact = (
            redact_pii if redact_pii is not None else server_default_redact
        )

        pipeline = get_pipeline(
            chosen_extractor, chosen_vision, schema, review_threshold,
        )
        workers = max_workers if max_workers is not None else 4

        def _run(path: Path):
            return pipeline.run(path)

        results: list[dict | None] = [None] * len(doc_paths)
        errors: list[dict] = []

        def _do_batch():
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(_run, p): (i, p) for i, p in enumerate(doc_paths)}
                for fut in futures:
                    i, p = futures[fut]
                    try:
                        result = fut.result()
                    except Exception as e:
                        if on_error == "raise":
                            raise
                        errors.append({"path": str(p), "error": str(e), "stage": "extract"})
                        # leave results[i] as None for both skip and include
                        continue
                    results[i] = build_extraction_response(
                        result=result, schema=pipeline.schema, schema_key=schema,
                        document_path=str(p.resolve()), redact=effective_redact,
                        include_audit_log=include_audit_log,
                        include_source_ref=include_source_ref,
                        include_pii_entities=include_pii_entities,
                    )

        await asyncio.to_thread(_do_batch)

        if on_error == "skip":
            results = [r for r in results if r is not None]

        return {"results": results, "errors": errors}

    # Built once per server, reused across tool calls. Presidio is heavy.
    pii_detector = CanadianPIIDetector()

    @app.tool()
    def detect_pii(text: str, return_redacted: bool = False) -> dict:
        """Detect Canadian + standard PII in arbitrary text. No LLM, no pipeline."""
        if return_redacted:
            redacted = pii_detector.redact(text)
            return {
                "entities": redacted.detected_entities,
                "redacted_text": redacted.redacted_text,
            }
        entities = pii_detector.analyze(text)
        return {"entities": entities, "redacted_text": None}

    # Stash config on the app for downstream tools added in later tasks.
    app._finlit_extractor = extractor                # type: ignore[attr-defined]
    app._finlit_vision = vision_extractor            # type: ignore[attr-defined]
    app._finlit_threshold = review_threshold         # type: ignore[attr-defined]
    app._finlit_default_redact = server_default_redact  # type: ignore[attr-defined]

    return app


def serve(
    *,
    extractor: str = "claude",
    vision_extractor: str | None = None,
    review_threshold: float = 0.85,
    pii_mode: PIIMode = "redact",
) -> None:
    """Build the app and run it over stdio. Blocks until the host disconnects."""
    app = build_app(
        extractor=extractor,
        vision_extractor=vision_extractor,
        review_threshold=review_threshold,
        pii_mode=pii_mode,
    )
    app.run()  # FastMCP defaults to stdio transport.
