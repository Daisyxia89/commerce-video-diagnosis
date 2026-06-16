from __future__ import annotations

from typing import Mapping

import pytest


def format_case_context(case_id: str, case_meta: Mapping[str, Mapping[str, object]] | None) -> str:
    meta = (case_meta or {}).get(case_id) or {}
    if not meta:
        return f"case_id={case_id}"
    return " | ".join(
        [
            f"case_id={case_id}",
            f"section={meta.get('source_section', '-')}",
            f"title={meta.get('title', '-')}",
            f"acceptance_focus={meta.get('acceptance_focus', '-')}",
        ]
    )



def assert_equal_with_case_context(
    actual: object,
    expected: object,
    *,
    case_id: str,
    field_name: str,
    case_meta: Mapping[str, Mapping[str, object]] | None,
) -> None:
    if actual == expected:
        return
    pytest.fail(
        f"{format_case_context(case_id, case_meta)} | field={field_name} | expected={expected!r} | actual={actual!r}"
    )



def assert_true_with_case_context(
    condition: bool,
    *,
    case_id: str,
    field_name: str,
    detail: str,
    case_meta: Mapping[str, Mapping[str, object]] | None,
) -> None:
    if condition:
        return
    pytest.fail(f"{format_case_context(case_id, case_meta)} | field={field_name} | detail={detail}")



def assert_contains_with_case_context(
    actual_text: str,
    expected_fragment: str,
    *,
    case_id: str,
    field_name: str,
    case_meta: Mapping[str, Mapping[str, object]] | None,
) -> None:
    if expected_fragment in actual_text:
        return
    pytest.fail(
        f"{format_case_context(case_id, case_meta)} | field={field_name} | expected_fragment={expected_fragment!r} | actual={actual_text!r}"
    )
