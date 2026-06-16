from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import yaml

ROOT = Path(__file__).resolve().parent
KEYWORD_RULES_PATH = ROOT / "config" / "keyword_rules.yaml"


@dataclass(frozen=True)
class KeywordRuleTrace:
    rule_path: str
    matched_keyword: str
    source_rule: str
    source_evidence: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@lru_cache(maxsize=1)
def load_keyword_rules() -> dict[str, Any]:
    if not KEYWORD_RULES_PATH.exists():
        raise FileNotFoundError(f"keyword rule config 不存在: {KEYWORD_RULES_PATH}")
    with KEYWORD_RULES_PATH.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError("keyword_rules.yaml 顶层必须是 object/dict。")
    return payload


def get_rule(rule_path: str) -> Any:
    current: Any = load_keyword_rules()
    for segment in rule_path.split("."):
        if not isinstance(current, Mapping) or segment not in current:
            raise KeyError(f"keyword rule 未配置: {rule_path}")
        current = current[segment]
    return current


def get_string_list(rule_path: str) -> list[str]:
    value = get_rule(rule_path)
    if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"keyword rule {rule_path} 必须是非空字符串列表。")
    return [item.strip() for item in value]


def get_mapping_of_string_lists(rule_path: str) -> dict[str, list[str]]:
    value = get_rule(rule_path)
    if not isinstance(value, Mapping):
        raise ValueError(f"keyword rule {rule_path} 必须是字符串 -> 字符串列表的映射。")
    normalized: dict[str, list[str]] = {}
    for key, items in value.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"keyword rule {rule_path} 存在非法 key。")
        if not isinstance(items, list) or any(not isinstance(item, str) or not item.strip() for item in items):
            raise ValueError(f"keyword rule {rule_path}.{key} 必须是非空字符串列表。")
        normalized[key.strip()] = [item.strip() for item in items]
    return normalized


def build_rule_trace(rule_path: str, matched_keyword: str, source_rule: str | None = None) -> KeywordRuleTrace:
    matched_value = str(matched_keyword or "").strip()
    source_value = str(source_rule or matched_keyword or "").strip()
    if not matched_value or not source_value:
        raise ValueError(f"keyword rule {rule_path} 缺少 matched_keyword/source_rule，无法构建 source_evidence。")
    return KeywordRuleTrace(
        rule_path=rule_path,
        matched_keyword=matched_value,
        source_rule=source_value,
        source_evidence=f"config/keyword_rules.yaml::{rule_path} 命中规则={source_value}；命中值={matched_value}",
    )


def assert_rule_trace(trace: KeywordRuleTrace | None, rule_path: str) -> KeywordRuleTrace:
    if trace is None:
        raise AssertionError(f"keyword rule {rule_path} 命中后缺少 source_evidence。")
    if trace.rule_path != rule_path:
        raise AssertionError(f"keyword rule trace 路径不一致: expected={rule_path}, actual={trace.rule_path}")
    if not trace.source_evidence.strip():
        raise AssertionError(f"keyword rule {rule_path} source_evidence 为空。")
    return trace
