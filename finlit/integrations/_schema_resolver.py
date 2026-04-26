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
