from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest

SKILL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
for candidate in (str(SKILL_ROOT), str(REPO_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from extractor.errors import ContaminationViolation
from extractor.validators.contamination_assertions import assert_no_contamination
from scripts.build_request import load_factpack


@pytest.mark.unit
@pytest.mark.parametrize("forbidden_key", [
    "category_strategy_intent",
    "product_strategy_intent",
    "intent_coordinates",
    "modifiers",
])
def test_assert_no_contamination_crashes_early_on_forbidden_protocol_fields(forbidden_key: str) -> None:
    payload = {
        "video_meta": {"source_platform": "抖音", "duration_sec": 1.0, "fps": 30.0, "resolution": "720x1280"},
        "segments": [
            {
                "segment_id": "SEG01",
                forbidden_key: {"leak": True},
            }
        ],
    }

    with pytest.raises(ContaminationViolation) as exc_info:
        assert_no_contamination(payload)

    assert forbidden_key in str(exc_info.value)


@pytest.mark.unit
@pytest.mark.parametrize("forbidden_key", [
    "category_strategy_intent",
    "product_strategy_intent",
    "intent_coordinates",
    "modifiers",
])
def test_build_request_factpack_loader_rejects_forbidden_protocol_fields(tmp_path: Path, forbidden_key: str) -> None:
    factpack_path = tmp_path / "factpack.json"
    factpack_path.write_text(
        json.dumps(
            {
                "video_meta": {"source_platform": "抖音", "duration_sec": 1.0, "fps": 30.0, "resolution": "720x1280"},
                "segments": [{"segment_id": "SEG01", forbidden_key: {"leak": True}}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as exc_info:
        load_factpack(factpack_path)

    assert forbidden_key in str(exc_info.value)
