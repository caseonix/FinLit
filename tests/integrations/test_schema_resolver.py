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
