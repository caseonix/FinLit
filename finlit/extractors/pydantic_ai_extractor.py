"""
pydantic-ai backed extractor.

Supports:
  - anthropic:claude-sonnet-4-6  (default)
  - openai:gpt-4o
  - ollama:llama3.2              (fully local)

Note: targets pydantic-ai >= 1.x, which uses ``output_type=`` on the
Agent constructor and ``result.output`` on the run result (the older
``result_type=`` / ``result.data`` names from 0.x are not used here).
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel
from pydantic_ai import Agent

from finlit.extractors.base import BaseExtractor
from finlit.schema import Schema


class ExtractionOutput(BaseModel):
    fields: dict[str, Any]
    confidence: dict[str, float]
    notes: str = ""


class PydanticAIExtractor(BaseExtractor):
    """LLM-backed field extractor built on pydantic-ai."""

    def __init__(self, model: str = "anthropic:claude-sonnet-4-6") -> None:
        self.model = model
        self._agent = Agent(
            model,
            output_type=ExtractionOutput,
            system_prompt=self._system_prompt(),
        )

    def _system_prompt(self) -> str:
        return """You are a precise document field extractor for Canadian financial documents.

Given document text and a list of fields to extract, return:
1. A JSON object 'fields' mapping field_name -> extracted value (use null if not found)
2. A JSON object 'confidence' mapping field_name -> float 0.0-1.0
3. A 'notes' string for any extraction warnings

Rules:
- Monetary values must be returned as float (e.g. 87500.00, not "$87,500.00")
- Social Insurance Numbers must be returned in format "XXX-XXX-XXX"
- Province codes must be 2-letter uppercase (ON, BC, QC, etc.)
- If a field is present but illegible, return null with confidence 0.0
- Do not hallucinate values. If you cannot find a field, return null.
- Tax year must be a 4-digit integer
"""

    def extract(self, text: str, schema: Schema) -> ExtractionOutput:
        prompt = self._build_prompt(text, schema)
        result = self._agent.run_sync(prompt)
        return result.output

    async def extract_async(self, text: str, schema: Schema) -> ExtractionOutput:
        prompt = self._build_prompt(text, schema)
        result = await self._agent.run(prompt)
        return result.output

    def _build_prompt(self, text: str, schema: Schema) -> str:
        field_descriptions = "\n".join(
            f"  - {f.name} ({f.dtype.__name__}): {f.description}"
            + (" [REQUIRED]" if f.required else "")
            for f in schema.fields
        )
        return f"""Document type: {schema.document_type}

Fields to extract:
{field_descriptions}

Document text:
---
{text[:8000]}
---

Extract all fields listed above from the document text."""
