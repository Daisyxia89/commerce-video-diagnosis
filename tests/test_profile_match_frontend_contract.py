import pytest

from commerce_video_diagnosis.understanding.engines.video_diagnoser import (
    VideoDiagnosisEngine,
    VideoDiagnosisInputError,
)


def test_profile_match_rejects_available_for_frontend_mapping():
    with pytest.raises(VideoDiagnosisInputError, match="available_for_frontend_mapping"):
        VideoDiagnosisEngine._assert_frontend_profile_match(
            {
                "status": "completed",
                "product_audience": {"primary": "商品人群", "scene": "场景", "core_need": "需求"},
                "video_audience": {"primary": "视频人群", "scene": "场景", "core_need": "需求"},
                "gap": {"level": "low", "description": "匹配"},
                "match_result": "high_match",
                "evidence": [
                    {"source": "product_factpack", "field": "a", "value": "b"},
                    {"source": "video_factpack", "field": "c", "value": "d"},
                ],
                "summary": "匹配",
                "available_for_frontend_mapping": True,
            }
        )


def test_profile_match_completed_requires_two_sided_evidence():
    with pytest.raises(VideoDiagnosisInputError, match="product_factpack"):
        VideoDiagnosisEngine._assert_frontend_profile_match(
            {
                "status": "completed",
                "product_audience": {"primary": "商品人群", "scene": "场景", "core_need": "需求"},
                "video_audience": {"primary": "视频人群", "scene": "场景", "core_need": "需求"},
                "gap": {"level": "low", "description": "匹配"},
                "match_result": "high_match",
                "evidence": [{"source": "video_factpack", "field": "c", "value": "d"}],
                "summary": "匹配",
            }
        )
