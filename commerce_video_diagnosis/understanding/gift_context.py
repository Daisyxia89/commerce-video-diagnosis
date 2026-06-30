"""通用 gift_context 送礼场景识别（第五批）。

确定性查表识别，**不依赖 LLM 语义推断**。信号来源：商品标题 / 卖点 /
source_evidence / 模块 1 目标人群原始线索（target_people_raw）。

字典 SSOT：``config/keyword_rules.yaml::product_diagnoser.gift_context``
（三类：intent_keywords / scenes / recipients）。字典缺失或结构非法时
Crash Early，禁止静默兜底。

该模块同时供：
- persuasion_requirement_engine（profile 礼赠类 requirement 激活）；
- product_diagnoser._derive_product_target_audience（audience 双重角色拆分）。
共用同一识别口径，保证 profile 与 audience 对 gift_context 的判定一致。
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any, Iterable, Mapping

from .keyword_rules import get_rule

_GIFT_CONTEXT_RULE_PATH = "product_diagnoser.gift_context"


@lru_cache(maxsize=1)
def _load_gift_dictionary() -> dict[str, Any]:
    """加载并校验 gift_context 字典（Crash Early）。"""
    raw = get_rule(_GIFT_CONTEXT_RULE_PATH)
    if not isinstance(raw, Mapping):
        raise ValueError(f"gift_context 字典必须是 object：{_GIFT_CONTEXT_RULE_PATH}")

    intent = raw.get("intent_keywords")
    if not isinstance(intent, list) or not intent or any(not str(x).strip() for x in intent):
        raise ValueError("gift_context.intent_keywords 必须是非空字符串列表。")

    scenes = raw.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        raise ValueError("gift_context.scenes 必须是非空列表。")
    for s in scenes:
        if not isinstance(s, Mapping) or not str(s.get("scene", "")).strip():
            raise ValueError("gift_context.scenes 每项必须含非空 scene。")
        kws = s.get("keywords")
        if not isinstance(kws, list) or not kws or any(not str(x).strip() for x in kws):
            raise ValueError(f"gift_context.scenes[{s.get('scene')}].keywords 必须是非空字符串列表。")

    recipients = raw.get("recipients")
    if not isinstance(recipients, list) or not recipients:
        raise ValueError("gift_context.recipients 必须是非空列表。")
    for r in recipients:
        if not isinstance(r, Mapping):
            raise ValueError("gift_context.recipients 每项必须是 object。")
        for key in ("recipient", "demographic", "purchase_decider", "relationship"):
            if not str(r.get(key, "")).strip():
                raise ValueError(f"gift_context.recipients 每项必须含非空 {key}。")
        kws = r.get("keywords")
        if not isinstance(kws, list) or not kws or any(not str(x).strip() for x in kws):
            raise ValueError(f"gift_context.recipients[{r.get('recipient')}].keywords 必须是非空字符串列表。")

    default_recipient = raw.get("default_recipient")
    if not isinstance(default_recipient, Mapping):
        raise ValueError("gift_context.default_recipient 必须是 object。")
    for key in ("recipient", "demographic", "purchase_decider", "relationship"):
        if not str(default_recipient.get(key, "")).strip():
            raise ValueError(f"gift_context.default_recipient 必须含非空 {key}。")

    return {
        "intent_keywords": [str(x).strip() for x in intent],
        "scenes": scenes,
        "recipients": recipients,
        "default_recipient": default_recipient,
    }


def _recipient_profile_by_label(recipients: list[Mapping[str, Any]], label: str) -> Mapping[str, Any] | None:
    for r in recipients:
        if str(r.get("recipient", "")).strip() == label:
            return r
    return None


def detect_gift_context(text_segments: Iterable[Any]) -> dict[str, Any] | None:
    """从文本片段集合中识别通用 gift_context。

    返回 ``None`` 表示非送礼场景（信号未命中）；命中时返回结构：
    ``{is_gift, gift_scene, gift_recipient, recipient_demographic,
       purchase_decider, relationship, evidence}``。

    判定口径（见 PRD §10.3）：
    - ``is_gift`` = 命中送礼意图 或 送礼时机（仅命中纯人群词不判为送礼）；
    - ``gift_recipient`` 优先级：显式送礼对象 > 时机隐含受礼者 > 通用受礼者。
    """
    gd = _load_gift_dictionary()

    # 统一文本化（保留原始信号，不做截断/剥离）。
    chunks: list[str] = []

    def _walk(value: Any) -> None:
        if value is None:
            return
        if isinstance(value, str):
            if value.strip():
                chunks.append(value)
        elif isinstance(value, Mapping):
            for v in value.values():
                _walk(v)
        elif isinstance(value, (list, tuple, set)):
            for v in value:
                _walk(v)
        else:
            chunks.append(str(value))

    for seg in text_segments:
        _walk(seg)
    blob = "\n".join(chunks)
    if not blob.strip():
        return None

    evidence: list[str] = []

    def _add_evidence(kw: str) -> None:
        if kw and kw not in evidence:
            evidence.append(kw)

    # 1) 送礼意图
    intent_hits = [kw for kw in gd["intent_keywords"] if kw in blob]
    for kw in intent_hits:
        _add_evidence(kw)

    # 2) 送礼时机（保留 yaml 顺序，取首个命中 scene；记录隐含受礼者）
    matched_scene_label: str | None = None
    scene_implied_recipient: str | None = None
    scene_hit = False
    for s in gd["scenes"]:
        hit_kw = next((kw for kw in s["keywords"] if str(kw) in blob), None)
        if hit_kw is not None:
            scene_hit = True
            _add_evidence(str(hit_kw))
            if matched_scene_label is None:
                matched_scene_label = str(s["scene"]).strip()
                implied = str(s.get("implies_recipient", "") or "").strip()
                if implied:
                    scene_implied_recipient = implied

    # is_gift 仅由送礼意图或时机决定（纯人群词不判为送礼）
    is_gift = bool(intent_hits) or scene_hit
    if not is_gift:
        return None

    # 3) 送礼对象：显式命中（yaml 顺序首个）> 时机隐含 > 默认
    explicit_recipient: Mapping[str, Any] | None = None
    for r in gd["recipients"]:
        hit_kw = next((kw for kw in r["keywords"] if str(kw) in blob), None)
        if hit_kw is not None:
            _add_evidence(str(hit_kw))
            if explicit_recipient is None:
                explicit_recipient = r

    if explicit_recipient is not None:
        recipient_profile: Mapping[str, Any] = explicit_recipient
    elif scene_implied_recipient:
        recipient_profile = (
            _recipient_profile_by_label(gd["recipients"], scene_implied_recipient)
            or gd["default_recipient"]
        )
    else:
        recipient_profile = gd["default_recipient"]

    return {
        "is_gift": True,
        "gift_scene": matched_scene_label or "通用送礼",
        "gift_recipient": str(recipient_profile["recipient"]).strip(),
        "recipient_demographic": str(recipient_profile["demographic"]).strip(),
        "purchase_decider": str(recipient_profile["purchase_decider"]).strip(),
        "relationship": str(recipient_profile["relationship"]).strip(),
        "evidence": evidence,
    }
