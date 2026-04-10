"""
Convenience imports for built-in schemas.

Usage:
    from finlit import schemas
    pipeline = DocumentPipeline(schema=schemas.CRA_T4)
"""
from pathlib import Path
from finlit.schema import Schema

_SCHEMAS_DIR = Path(__file__).parent


def _load(rel_path: str) -> Schema:
    return Schema.from_yaml(_SCHEMAS_DIR / rel_path)


# CRA tax slips
CRA_T4 = _load("cra/t4.yaml")
CRA_T5 = _load("cra/t5.yaml")
CRA_T4A = _load("cra/t4a.yaml")
CRA_NR4 = _load("cra/nr4.yaml")

# Banking
BANK_STATEMENT = _load("banking/bank_statement.yaml")
