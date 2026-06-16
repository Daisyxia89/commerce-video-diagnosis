from __future__ import annotations

from functools import partial
import json
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

SKILL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_PATH = SKILL_ROOT / "tests/fixtures/replay_pipeline_cases.json"

for candidate in (str(SKILL_ROOT), str(REPO_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from extractor.errors import FactPackViolation  # noqa: E402
from extractor.validators.factpack_assertions import assert_factpack_schema  # noqa: E402
from tests.case_meta_helpers import (  # noqa: E402
    assert_equal_with_case_context,
    assert_true_with_case_context,
)
from tests.factpack_protocol_negative_cases import (  # noqa: E402
    apply_mutation,
    build_main_skeleton_assertion_negative_cases,
    build_main_skeleton_parse_negative_cases,
)
from commerce_video_diagnosis.understanding.core import FactPack  # noqa: E402


FIXTURE_DATA = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
CASES = FIXTURE_DATA["cases"]
CASE_META = FIXTURE_DATA.get("case_meta") or {}
_assert_true_with_case_context = partial(assert_true_with_case_context, case_meta=CASE_META)
_assert_equal_with_case_context = partial(assert_equal_with_case_context, case_meta=CASE_META)


SECOND_FILTER_TRACE_REPLAY_TYPE_BOUNDARY_CASES = [
    {
        "label": "candidate_high_ocr_scene_string",
        "path": ("second_filter_trace", "candidates", 0, "high_ocr_scene"),
        "invalid_value": "true",
        "expected_path": "second_filter_trace -> candidates -> 0 -> high_ocr_scene",
    },
    {
        "label": "decision_cta_int",
        "path": ("second_filter_trace", "decisions", 0, "cta"),
        "invalid_value": 1,
        "expected_path": "second_filter_trace -> decisions -> 0 -> cta",
    },
]


@pytest.mark.integration
@pytest.mark.parametrize("case", CASES, ids=[item["case_id"] for item in CASES])
def test_replay_pipeline_cases(case: dict) -> None:
    case_id = case["case_id"]

    if case["type"] == "factpack_artifact":
        factpack_path = REPO_ROOT / case["path"]
        _assert_true_with_case_context(
            factpack_path.is_file(),
            case_id=case_id,
            field_name="artifact_path",
            detail=f"missing artifact: {factpack_path}",
        )
        factpack = json.loads(factpack_path.read_text(encoding="utf-8"))
        assert_factpack_schema(factpack)
        parsed_factpack = FactPack.parse_obj(factpack)
        _assert_true_with_case_context(
            bool(parsed_factpack.second_filter_trace.candidates),
            case_id=case_id,
            field_name="second_filter_trace.candidates",
            detail="parsed second_filter_trace candidates must exist",
        )
        _assert_true_with_case_context(
            bool(parsed_factpack.second_filter_trace.decisions),
            case_id=case_id,
            field_name="second_filter_trace.decisions",
            detail="parsed second_filter_trace decisions must exist",
        )
        for schema_case in SECOND_FILTER_TRACE_REPLAY_TYPE_BOUNDARY_CASES:
            mutated_factpack = apply_mutation(
                factpack,
                {"mode": "set", "path": schema_case["path"], "value": schema_case["invalid_value"]},
            )
            with pytest.raises(ValidationError) as exc_info:
                FactPack.parse_obj(mutated_factpack)
            _assert_true_with_case_context(
                schema_case["expected_path"] in str(exc_info.value),
                case_id=case_id,
                field_name=schema_case["label"],
                detail=str(exc_info.value),
            )
        for schema_case in build_main_skeleton_parse_negative_cases(factpack):
            mutated_factpack = apply_mutation(factpack, schema_case["mutation"])
            with pytest.raises(ValidationError) as exc_info:
                FactPack.parse_obj(mutated_factpack)
            _assert_true_with_case_context(
                schema_case["expected_path"] in str(exc_info.value),
                case_id=case_id,
                field_name=schema_case["label"],
                detail=str(exc_info.value),
            )
        for assertion_case in build_main_skeleton_assertion_negative_cases(factpack):
            mutated_factpack = apply_mutation(factpack, assertion_case["mutation"])
            with pytest.raises(FactPackViolation) as exc_info:
                assert_factpack_schema(mutated_factpack)
            _assert_true_with_case_context(
                assertion_case["expected_error"] in str(exc_info.value),
                case_id=case_id,
                field_name=assertion_case["label"],
                detail=str(exc_info.value),
            )
        _assert_true_with_case_context(
            len(factpack.get("semantic_bundles") or []) >= case["expected"]["semantic_bundle_min"],
            case_id=case_id,
            field_name="semantic_bundles",
            detail=(
                f"expected at least {case['expected']['semantic_bundle_min']} semantic bundles, "
                f"actual={len(factpack.get('semantic_bundles') or [])}"
            ),
        )
        if case["expected"].get("second_filter_trace_required"):
            _assert_true_with_case_context(
                bool(factpack.get("second_filter_trace")),
                case_id=case_id,
                field_name="second_filter_trace",
                detail="second_filter_trace must exist",
            )
        return

    if case["type"] == "commerce_video_diagnosis_request":
        payload_path = REPO_ROOT / case["payload"]
        output_path = REPO_ROOT / case["output"]
        _assert_true_with_case_context(
            payload_path.is_file(),
            case_id=case_id,
            field_name="payload_path",
            detail=f"missing payload: {payload_path}",
        )
        command = [
            sys.executable,
            str(REPO_ROOT / "user_skills/commerce-video-diagnosis/scripts/run_v2.py"),
            "--payload",
            str(payload_path),
            "--output",
            str(output_path),
        ]
        subprocess.run(command, check=True)
        result = json.loads(output_path.read_text(encoding="utf-8"))
        blueprint = result.get("blueprint") or {}
        _assert_equal_with_case_context(
            blueprint.get("storyboard_source"),
            case["expected"]["storyboard_source"],
            case_id=case_id,
            field_name="blueprint.storyboard_source",
        )
        _assert_true_with_case_context(
            len(blueprint.get("semantic_bundles") or []) >= case["expected"]["semantic_bundle_min"],
            case_id=case_id,
            field_name="blueprint.semantic_bundles",
            detail=(
                f"expected at least {case['expected']['semantic_bundle_min']} semantic bundles, "
                f"actual={len(blueprint.get('semantic_bundles') or [])}"
            ),
        )
        return

    raise AssertionError(f"unsupported replay case type: {case['type']}")
