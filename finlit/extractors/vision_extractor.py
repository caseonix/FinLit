"""
Vision-based extractor built on pydantic-ai.

Sends a list of PNG page images to any multimodal LLM that pydantic-ai
supports — Claude, OpenAI, Gemini, Ollama-hosted open-source models, or
anything behind an OpenAI-compatible endpoint. Consumers pick the model
by passing a pydantic-ai model string.

Tested OSS model strings (via Ollama):
    - "ollama:llama3.2-vision"      Meta, 11B, general-purpose
    - "ollama:qwen2.5vl:7b"         Alibaba, strongest for forms
    - "ollama:minicpm-v"            OpenBMB, fast 8B

Non-multimodal models will fail at extraction time with a provider error.
"""
from __future__ import annotations

import threading
from typing import Any

from pydantic_ai import BinaryContent

from finlit.extractors.base_vision import BaseVisionExtractor
from finlit.extractors.pydantic_ai_extractor import ExtractionOutput
from finlit.schema import Schema


class VisionExtractor(BaseVisionExtractor):
    """Vision extractor backed by pydantic-ai.

    Parameters
    ----------
    model:
        A pydantic-ai model string for a multimodal model. Defaults to
        ``"anthropic:claude-sonnet-4-6"``. Examples:

            VisionExtractor(model="openai:gpt-4o")
            VisionExtractor(model="google-gla:gemini-2.0-flash")
            VisionExtractor(model="ollama:qwen2.5vl:7b")   # fully local

        The model MUST be multimodal. Non-vision models will raise at
        extraction time.
    dpi:
        DPI used when the pipeline renders PDFs via ``render_pages()``.
        Default 200. Stored on the extractor so the pipeline can read it
        when deciding how to render.
    image_format:
        Reserved. Only ``"png"`` is supported today.
    max_pages:
        Hard cap on page count per document. If a document has more
        pages than ``max_pages``, ``extract()`` raises ``ValueError``
        *before* any LLM call. ``None`` (default) disables the cap.
    """

    # Maximum characters of the text-path hint forwarded to the LLM in
    # the prompt. Keeps token usage bounded when the text extractor
    # produced a long but low-quality output.
    _TEXT_HINT_MAX_CHARS: int = 4000

    def __init__(
        self,
        model: str = "anthropic:claude-sonnet-4-6",
        dpi: int = 200,
        image_format: str = "png",
        max_pages: int | None = None,
    ) -> None:
        self.model = model
        self.dpi = dpi
        self.image_format = image_format
        self.max_pages = max_pages
        # Agent is built lazily in _get_agent() so that constructing
        # VisionExtractor() does not require an API key to be present.
        # Tests replace _agent with a fake before calling extract().
        self._agent: Any = None
        # BatchPipeline submits pipeline.run() to a ThreadPoolExecutor
        # with a shared VisionExtractor instance. Double-checked locking
        # in _get_agent() prevents parallel Agent construction.
        self._agent_lock = threading.Lock()

    def _get_agent(self) -> Any:
        """Return the pydantic-ai Agent, building it on first use."""
        if self._agent is None:
            with self._agent_lock:
                if self._agent is None:
                    from pydantic_ai import Agent

                    self._agent = Agent(
                        self.model,
                        output_type=ExtractionOutput,
                        system_prompt=self._system_prompt(),
                    )
        return self._agent

    def _system_prompt(self) -> str:
        return """You are a precise document field extractor for Canadian financial documents.

You are looking at scanned or rendered page images of a document (not plain text).
Read the visible layout the way a human would: find each labelled box or field,
match it to its adjacent value, and extract the value. The spatial relationship
between a label and its value is the primary signal — do not infer a value from
context if it is not visible in the image.

Return:
1. A JSON object 'fields' mapping field_name -> extracted value (use null if not found)
2. A JSON object 'confidence' mapping field_name -> float 0.0-1.0
3. A 'notes' string for any extraction warnings

Rules:
- Monetary values must be floats (e.g. 87500.00, not "$87,500.00")
- Social Insurance Numbers in format "XXX-XXX-XXX"
- Province codes as 2-letter uppercase (ON, BC, QC, etc.)
- Tax year as 4-digit integer
- If a box is present but illegible, return null with confidence 0.0
- Do not hallucinate. If a field is not visible, return null.
"""

    def extract(
        self,
        images: list[bytes],
        schema: Schema,
        text: str = "",
    ) -> ExtractionOutput:
        if self.max_pages is not None and len(images) > self.max_pages:
            raise ValueError(
                f"VisionExtractor: document has {len(images)} pages but "
                f"max_pages={self.max_pages}"
            )

        prompt_text = self._build_prompt(schema, text)
        # pydantic-ai accepts a list of mixed string + BinaryContent parts
        parts: list[Any] = [prompt_text]
        for img in images:
            parts.append(BinaryContent(data=img, media_type="image/png"))

        result = self._get_agent().run_sync(parts)
        return result.output

    def _build_prompt(self, schema: Schema, text_hint: str) -> str:
        field_descriptions = "\n".join(
            f"  - {f.name} ({f.dtype.__name__}): {f.description}"
            + (" [REQUIRED]" if f.required else "")
            for f in schema.fields
        )
        hint_section = ""
        if text_hint:
            # Truncate to keep token usage bounded
            hint_section = (
                "\n\nText path partial output (may be incomplete or wrong — "
                "prefer what you can see in the images):\n"
                f"---\n{text_hint[: self._TEXT_HINT_MAX_CHARS]}\n---\n"
            )
        return f"""Document type: {schema.document_type}

Fields to extract:
{field_descriptions}
{hint_section}
Extract all fields listed above from the page image(s) below."""
