"""
Validates extracted field values against their Schema Field definitions.
Checks dtype, regex pattern, and required fields.
Returns (validated_dict, list_of_errors).
"""
from __future__ import annotations

import re
from typing import Any

from finlit.schema import Schema


class FieldValidator:
    def validate(
        self, raw_fields: dict[str, Any], schema: Schema
    ) -> tuple[dict[str, Any], list[str]]:
        validated: dict[str, Any] = {}
        errors: list[str] = []

        for schema_field in schema.fields:
            raw = raw_fields.get(schema_field.name)

            if raw is None:
                if schema_field.required:
                    errors.append(
                        f"Required field missing: {schema_field.name}"
                    )
                validated[schema_field.name] = None
                continue

            # dtype coercion
            try:
                coerced = schema_field.dtype(raw)
            except (ValueError, TypeError):
                errors.append(
                    f"Type error on {schema_field.name}: "
                    f"cannot cast {raw!r} to {schema_field.dtype.__name__}"
                )
                validated[schema_field.name] = raw
                continue

            # regex validation (only for strings)
            if schema_field.regex and isinstance(coerced, str):
                if not re.fullmatch(schema_field.regex, coerced):
                    errors.append(
                        f"Regex validation failed on {schema_field.name}: "
                        f"{coerced!r} does not match {schema_field.regex}"
                    )

            validated[schema_field.name] = coerced

        return validated, errors
