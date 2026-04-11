"""
FinLit CLI.

Commands:
  finlit extract document.pdf --schema cra.t4 --extractor claude
  finlit schema-list
"""
from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="finlit", help="Canadian Financial Document Intelligence Framework"
)
console = Console()


def _schema_map() -> dict:
    from finlit import schemas as sr
    return {
        "cra.t4": sr.CRA_T4,
        "cra.t5": sr.CRA_T5,
        "cra.t4a": sr.CRA_T4A,
        "cra.nr4": sr.CRA_NR4,
        "banking.statement": sr.BANK_STATEMENT,
    }


@app.command()
def extract(
    document: Path = typer.Argument(..., help="Path to document (PDF, DOCX, image)"),
    schema: str = typer.Option("cra.t4", help="Schema to use (e.g. cra.t4, cra.t5)"),
    extractor: str = typer.Option("claude", help="Text extractor: claude | openai | ollama | <pydantic-ai model string>"),
    vision_extractor: str = typer.Option(
        None,
        "--vision-extractor",
        help="Optional vision fallback model (e.g. 'claude', 'openai:gpt-4o', 'ollama:qwen2.5vl:7b')",
    ),
    output: str = typer.Option("table", help="Output format: table | json | jsonl"),
    review_threshold: float = typer.Option(0.85, help="Confidence threshold"),
):
    """Extract structured data from a Canadian financial document."""
    from finlit import DocumentPipeline, VisionExtractor

    schema_map = _schema_map()
    if schema not in schema_map:
        console.print(
            f"[red]Unknown schema: {schema}. "
            f"Available: {', '.join(schema_map.keys())}[/red]"
        )
        raise typer.Exit(1)

    ve = None
    if vision_extractor:
        # Accept shorthand aliases the same way the text extractor does.
        model_str = {
            "claude": "anthropic:claude-sonnet-4-6",
            "openai": "openai:gpt-4o",
            "ollama": "ollama:llama3.2-vision",
        }.get(vision_extractor, vision_extractor)
        ve = VisionExtractor(model=model_str)

    console.print(f"[dim]Parsing {document}...[/dim]")
    pipeline = DocumentPipeline(
        schema=schema_map[schema],
        extractor=extractor,
        review_threshold=review_threshold,
        vision_extractor=ve,
    )
    result = pipeline.run(document)

    if output == "json":
        console.print(json.dumps(result.fields, indent=2, default=str))
        return
    if output == "jsonl":
        console.print(
            json.dumps(
                {"fields": result.fields, "confidence": result.confidence},
                default=str,
            )
        )
        return

    table = Table(title=f"Extraction: {document.name}")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="white")
    table.add_column("Confidence", style="green")
    table.add_column("Review?", style="yellow")

    review_set = {r["field"] for r in result.review_fields}
    for field_name, value in result.fields.items():
        conf = result.confidence.get(field_name, 0.0)
        needs_review = "!" if field_name in review_set else ""
        table.add_row(
            field_name,
            str(value) if value is not None else "[dim]-[/dim]",
            f"{conf:.0%}",
            needs_review,
        )
    console.print(table)

    if result.extraction_path == "vision":
        console.print("[cyan]ℹ Result produced by vision fallback[/cyan]")

    if result.needs_review:
        console.print(
            f"\n[yellow]{len(result.review_fields)} field(s) flagged for review[/yellow]"
        )


@app.command("schema-list")
def schema_list():
    """List all built-in schemas."""
    schema_map = _schema_map()
    table = Table(title="Built-in Schemas")
    table.add_column("Key", style="cyan")
    table.add_column("Name")
    table.add_column("Fields", justify="right")
    for key, s in schema_map.items():
        table.add_row(key, s.document_type, str(len(s.fields)))
    console.print(table)


if __name__ == "__main__":
    app()
