from __future__ import annotations

from collections import Counter
import re

from ..errors import FactPackViolation

ALLOWED_TASK_STAGES = {
    "problem_statement",
    "test_setup",
    "test_execution",
    "result_explanation",
    "social_proof",
    "ingredient_backing",
    "cta_close",
}
ALLOWED_ACTION_TYPES = {
    "talking_head_explanation",
    "broll_supporting",
    "closeup_demo",
    "instrument_test",
    "result_showcase",
    "social_proof_insert",
    "cta_delivery",
}
ALLOWED_VISUAL_CARRIER_TYPES = {
    "talking_head",
    "closeup_face",
    "closeup_hand",
    "broll",
    "comment_page",
    "comparison_card",
    "report_card",
    "product_demo",
}
ALLOWED_EVIDENCE_REFS = {"ASR", "OCR", "VISUAL", "ACTION"}
LAYOUT_OCR_SIGNALS = {"layout_migration", "ocr_structure_jump"}
ALLOWED_STAGE_TRANSITIONS = {
    ("problem_statement", "test_setup"),
    ("problem_statement", "result_explanation"),
    ("test_setup", "test_execution"),
    ("test_setup", "result_explanation"),
    ("test_execution", "result_explanation"),
    ("result_explanation", "social_proof"),
    ("result_explanation", "ingredient_backing"),
    ("result_explanation", "cta_close"),
    ("ingredient_backing", "social_proof"),
    ("ingredient_backing", "cta_close"),
    ("social_proof", "cta_close"),
}
CTA_TOKENS = {
    "下单",
    "购买",
    "立即购买",
    "马上买",
    "点链接",
    "点击链接",
    "购物车",
    "橱窗",
    "领券",
    "直播间",
    "拍",
    "抢",
    "私信",
    "客服",
    "关注",
}
SOCIAL_PROOF_TOKENS = {"评论", "评论区", "反馈", "回购", "用户说", "大家都说", "测评", "口碑", "见证"}
INGREDIENT_TOKENS = {"成分", "配方", "专利", "修护", "屏障", "神经酰胺", "玻尿酸", "二裂酵母", "烟酰胺"}
RESULT_TOKENS = {"舒缓", "缓解", "降温", "不卡粉", "服帖", "前后", "结果", "有效", "即时", "即刻", "退红", "舒服"}
TEST_START_TOKENS = {"开始测", "来测", "测试", "上脸", "上妆", "抹开", "开测", "测一下", "试一下", "温度"}
PROBLEM_TOKENS = {"痛", "疼", "刺", "红", "烫", "干", "晒", "敏感", "卡粉", "搓泥", "鸡皮"}


class SecondFilterContractViolation(FactPackViolation):
    """协议层二筛断言异常。"""



def _normalize_text(value: object) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"\s+", "", text)
    return text



def _extract_tokens(*parts: object) -> set[str]:
    tokens: set[str] = set()
    for part in parts:
        text = _normalize_text(part)
        if not text:
            continue
        tokens.update(re.findall(r"[a-z0-9_]+", text))
        tokens.update(re.findall(r"[\u4e00-\u9fff]{2,}", text))
    return {token for token in tokens if token}



def _segment_text_payload(segment: dict) -> str:
    visual = segment.get("visual_facts") or {}
    audio = segment.get("audio_facts") or {}
    ocr = segment.get("ocr_facts") or []
    actions = visual.get("actions") or []
    return " ".join(
        str(part).strip()
        for part in (
            audio.get("asr_text"),
            visual.get("visual_subject"),
            " ".join(str(item.get("text") or "") for item in ocr),
            " ".join(str(item.get("action_name") or "") for item in actions),
            " ".join(str(item) for item in visual.get("key_objects") or []),
        )
        if str(part).strip()
    )



def _contains_any(text: str, words: set[str]) -> bool:
    normalized = _normalize_text(text)
    return any(_normalize_text(word) in normalized for word in words)



def _guess_visual_carrier_type(segment: dict) -> str:
    visual = segment.get("visual_facts") or {}
    shot_size = _normalize_text(visual.get("shot_size"))
    subject = _normalize_text(visual.get("visual_subject"))
    ocr_text = _normalize_text(" ".join(str(item.get("text") or "") for item in segment.get("ocr_facts") or []))
    text = " ".join(part for part in (subject, ocr_text) if part)

    if any(token in text for token in ("评论", "comment", "review", "feedback")):
        return "comment_page"
    if any(token in text for token in ("报告", "report", "数据", "score", "排行")):
        return "report_card"
    if any(token in text for token in ("对比", "前后", "vs", "beforeafter", "comparison")):
        return "comparison_card"
    if any(token in text for token in ("product", "bottle", "jar", "面霜", "凝露", "瓶", "罐")) and "hand" in text:
        return "product_demo"
    if any(token in text for token in ("hand", "手", "掌心", "手臂")) or shot_size in {"hand_closeup", "手部特写"}:
        return "closeup_hand"
    if any(token in text for token in ("face", "脸", "面部", "脸颊")) or shot_size in {"close_up", "特写", "closeup"}:
        return "closeup_face"
    if shot_size in {"medium_shot", "wide_shot", "full_shot", "中景", "远景"}:
        return "talking_head"
    return "broll"



def _guess_action_type(segment: dict, carrier_type: str, task_stage: str) -> str:
    visual = segment.get("visual_facts") or {}
    action_text = _normalize_text(" ".join(str(item.get("action_name") or "") for item in visual.get("actions") or []))
    payload = _normalize_text(_segment_text_payload(segment))

    if task_stage == "cta_close":
        return "cta_delivery"
    if task_stage == "social_proof":
        return "social_proof_insert"
    if any(token in action_text or token in payload for token in ("meter", "测", "温度", "仪器", "test")):
        return "instrument_test"
    if carrier_type in {"closeup_face", "closeup_hand", "product_demo"}:
        if any(token in action_text or token in payload for token in ("show", "展示", "抹开", "涂", "holding bottle", "holdingdevice")):
            return "closeup_demo"
        return "result_showcase"
    if carrier_type in {"comment_page", "report_card", "comparison_card", "broll"}:
        return "broll_supporting"
    return "talking_head_explanation"



def _infer_concern_label(segment: dict) -> str:
    payload = _normalize_text(_segment_text_payload(segment))
    if any(token in payload for token in ("晒", "烫", "红", "刺痛", "舒缓", "降温", "火焰山")):
        return "soothing_relief"
    if any(token in payload for token in ("妆前", "上妆", "不卡粉", "服帖", "底妆", "搓泥")):
        return "makeup_prep"
    if any(token in payload for token in ("成分", "配方", "修护", "屏障")):
        return "ingredient_backing"
    if any(token in payload for token in ("温度", "测温", "仪器", "数据")):
        return "instrument_result"
    if any(token in payload for token in ("质地", "抹开", "化水", "肤感")):
        return "texture_demo"
    return "general_claim"



def _infer_task_stage(segment: dict, carrier_type: str) -> str:
    payload = _segment_text_payload(segment)
    normalized = _normalize_text(payload)
    if _contains_any(payload, CTA_TOKENS):
        return "cta_close"
    if carrier_type == "comment_page" or _contains_any(payload, SOCIAL_PROOF_TOKENS):
        return "social_proof"
    if _contains_any(payload, INGREDIENT_TOKENS):
        return "ingredient_backing"
    if _contains_any(payload, TEST_START_TOKENS) or any(token in normalized for token in ("meter", "测温", "开始测", "来测")):
        return "test_setup"
    if any(token in normalized for token in ("抹开", "涂开", "上脸", "上妆", "继续测", "apply")):
        return "test_execution"
    if _contains_any(payload, RESULT_TOKENS):
        return "result_explanation"
    if _contains_any(payload, PROBLEM_TOKENS):
        return "problem_statement"
    return "result_explanation" if carrier_type == "talking_head" else "test_execution"



def _infer_subject_entity(segment: dict, carrier_type: str) -> str:
    visual = segment.get("visual_facts") or {}
    subject = str(visual.get("visual_subject") or "").strip()
    if subject:
        return subject
    if carrier_type == "comment_page":
        return "评论页"
    if carrier_type == "report_card":
        return "报告卡"
    return "当前主视觉主体"



def _infer_target_object(segment: dict) -> str:
    payload = _segment_text_payload(segment)
    for token in ("婴适孩童面霜", "面霜", "凝露", "妆前", "粉底", "温度计"):
        if token in str(payload):
            return token
    return _infer_concern_label(segment)



def derive_segment_semantics(segment: dict) -> dict:
    carrier_type = _guess_visual_carrier_type(segment)
    task_stage = _infer_task_stage(segment, carrier_type)
    action_type = _guess_action_type(segment, carrier_type, task_stage)
    proof_goal = {
        "cta_close": "促进成交",
        "social_proof": "建立社会证明",
        "ingredient_backing": "建立成分背书",
        "problem_statement": f"建立问题场景:{_infer_concern_label(segment)}",
        "test_setup": f"验证准备:{_infer_concern_label(segment)}",
        "test_execution": f"验证执行:{_infer_concern_label(segment)}",
        "result_explanation": f"验证结果:{_infer_concern_label(segment)}",
    }[task_stage]
    target_object = _infer_target_object(segment)
    evidence_refs: list[str] = []
    if str((segment.get("audio_facts") or {}).get("asr_text") or "").strip():
        evidence_refs.append("ASR")
    if segment.get("ocr_facts"):
        evidence_refs.append("OCR")
    if str((segment.get("visual_facts") or {}).get("visual_subject") or "").strip():
        evidence_refs.append("VISUAL")
    if (segment.get("visual_facts") or {}).get("actions"):
        evidence_refs.append("ACTION")
    return {
        "argument_chain_id": f"CHAIN_{proof_goal}_{target_object}",
        "task_stage": task_stage,
        "proof_goal": proof_goal,
        "subject_entity": _infer_subject_entity(segment, carrier_type),
        "target_object": target_object,
        "action_type": action_type,
        "visual_carrier_type": carrier_type,
        "contains_new_test_start": False,
        "contains_new_subject_switch": False,
        "contains_new_goal_switch": False,
        "contains_cta_transition": False,
        "semantic_summary": str(segment.get("audio_facts", {}).get("asr_text") or segment.get("visual_facts", {}).get("visual_subject") or "当前语义片段")[:120],
        "evidence_refs": evidence_refs,
    }



def _same_subject(prev: dict, nxt: dict) -> bool:
    return _normalize_text(prev.get("subject_entity")) == _normalize_text(nxt.get("subject_entity"))



def _same_goal(prev: dict, nxt: dict) -> bool:
    return _normalize_text(prev.get("proof_goal")) == _normalize_text(nxt.get("proof_goal"))



def _decorate_transition_flags(prev: dict, nxt: dict) -> tuple[dict, dict]:
    prev_copy = dict(prev)
    next_copy = dict(nxt)
    next_copy["contains_cta_transition"] = nxt.get("task_stage") == "cta_close" and prev.get("task_stage") != "cta_close"
    next_copy["contains_new_subject_switch"] = not _same_subject(prev, nxt)
    next_copy["contains_new_goal_switch"] = not _same_goal(prev, nxt) and not next_copy["contains_cta_transition"]
    next_copy["contains_new_test_start"] = nxt.get("task_stage") in {"test_setup", "test_execution"} and (
        prev.get("task_stage") not in {"test_setup", "test_execution"}
        or prev.get("action_type") != nxt.get("action_type")
    )
    return prev_copy, next_copy



def is_stage_shift(stage_a: str, stage_b: str) -> bool:
    normalized_a = str(stage_a or "").strip()
    normalized_b = str(stage_b or "").strip()
    if not normalized_a or not normalized_b or normalized_a == normalized_b:
        return False
    return (normalized_a, normalized_b) in ALLOWED_STAGE_TRANSITIONS



def is_carrier_only_shift(carrier_a: str, carrier_b: str) -> bool:
    normalized = {str(carrier_a or "").strip(), str(carrier_b or "").strip()}
    if len(normalized) == 1:
        return True
    material_carrier_cluster = {
        "talking_head",
        "broll",
        "comment_page",
        "comparison_card",
        "report_card",
        "closeup_face",
        "closeup_hand",
        "product_demo",
    }
    return normalized.issubset(material_carrier_cluster)



def same_task_stage(prev: dict, nxt: dict) -> bool:
    return str(prev.get("task_stage") or "").strip() == str(nxt.get("task_stage") or "").strip()



def is_same_step_micro_cut(prev: dict, nxt: dict) -> bool:
    if not same_task_stage(prev, nxt):
        return False
    if _normalize_text(prev.get("target_object")) != _normalize_text(nxt.get("target_object")):
        return False
    closeup_family = {"closeup_face", "closeup_hand", "product_demo", "broll", "talking_head"}
    if {str(prev.get("visual_carrier_type") or ""), str(nxt.get("visual_carrier_type") or "")}.issubset(closeup_family):
        return True
    action_pair = {str(prev.get("action_type") or ""), str(nxt.get("action_type") or "")}
    return action_pair.issubset({"closeup_demo", "result_showcase", "broll_supporting", "talking_head_explanation"})



def mainly_triggered_by_layout_or_ocr(signals: list[str]) -> bool:
    normalized = [str(item or "").strip() for item in signals if str(item or "").strip()]
    if not normalized:
        return False
    counts = Counter(normalized)
    layout_ocr_count = sum(count for signal, count in counts.items() if signal in LAYOUT_OCR_SIGNALS)
    return layout_ocr_count >= max(1, len(normalized) - 1)



def validate_contract(candidate: dict, prev: dict, nxt: dict) -> None:
    top_required = (
        "boundary_id",
        "protected_sec",
        "prev_segment_id",
        "next_segment_id",
        "trigger_signals",
        "high_ocr_scene",
        "prev_segment_semantics",
        "next_segment_semantics",
        "decision_context",
    )
    for field in top_required:
        if field not in candidate:
            raise SecondFilterContractViolation(f"SecondFilterBoundaryCandidate 缺少字段: {field}")

    decision_context = candidate.get("decision_context")
    if not isinstance(decision_context, dict):
        raise SecondFilterContractViolation("decision_context 必须为对象")
    required_context_fields = {
        "candidate_score": (int, float, type(None)),
        "adjacent_protected_count_10s": (int,),
        "same_bundle_relation": (str,),
        "ocr_jump_strength": (int, float),
        "layout_migration_strength": (int, float),
    }
    for field, allowed_types in required_context_fields.items():
        value = decision_context.get(field)
        if not isinstance(value, allowed_types):
            raise SecondFilterContractViolation(f"decision_context.{field} 非法")

    required_semantic_fields = ("argument_chain_id", "task_stage", "proof_goal", "subject_entity", "visual_carrier_type")
    for side_name, payload in (("prev", prev), ("next", nxt)):
        for field in required_semantic_fields:
            if not str(payload.get(field) or "").strip():
                raise SecondFilterContractViolation(f"{side_name}.{field} 缺失")
        for bool_field in (
            "contains_new_test_start",
            "contains_new_subject_switch",
            "contains_new_goal_switch",
            "contains_cta_transition",
        ):
            if not isinstance(payload.get(bool_field), bool):
                raise SecondFilterContractViolation(f"{side_name}.{bool_field} 必须显式为 bool")
        if payload.get("task_stage") not in ALLOWED_TASK_STAGES:
            raise SecondFilterContractViolation(f"{side_name}.task_stage 非法: {payload.get('task_stage')}")
        if str(payload.get("action_type") or "").strip() and payload.get("action_type") not in ALLOWED_ACTION_TYPES:
            raise SecondFilterContractViolation(f"{side_name}.action_type 非法: {payload.get('action_type')}")
        if payload.get("visual_carrier_type") not in ALLOWED_VISUAL_CARRIER_TYPES:
            raise SecondFilterContractViolation(f"{side_name}.visual_carrier_type 非法: {payload.get('visual_carrier_type')}")
        evidence_refs = payload.get("evidence_refs") or []
        if not isinstance(evidence_refs, list) or not all(item in ALLOWED_EVIDENCE_REFS for item in evidence_refs):
            raise SecondFilterContractViolation(f"{side_name}.evidence_refs 非法")
        if candidate.get("high_ocr_scene"):
            has_ocr = "OCR" in evidence_refs
            has_cross_modal = bool({"ASR", "VISUAL"} & set(evidence_refs))
            if not (has_ocr and has_cross_modal):
                raise SecondFilterContractViolation(f"{side_name}.evidence_refs 不满足高 OCR 交叉证据要求")

    same_chain = _normalize_text(prev.get("argument_chain_id")) == _normalize_text(nxt.get("argument_chain_id"))
    stage_shift = prev.get("task_stage") != nxt.get("task_stage")
    if same_chain and stage_shift and not (nxt.get("contains_new_test_start") or nxt.get("contains_new_goal_switch")):
        raise SecondFilterContractViolation("同论证链 task_stage 变化但缺少新任务/新目标支撑")



def second_filter(candidate: dict) -> dict:
    prev = candidate["prev_segment_semantics"]
    nxt = candidate["next_segment_semantics"]
    validate_contract(candidate, prev, nxt)

    same_chain = _normalize_text(prev.get("argument_chain_id")) == _normalize_text(nxt.get("argument_chain_id"))
    same_goal_flag = _same_goal(prev, nxt)
    no_new_task = not nxt.get("contains_new_test_start")
    no_new_subject = not nxt.get("contains_new_subject_switch")
    no_new_goal = not nxt.get("contains_new_goal_switch")
    no_cta = not nxt.get("contains_cta_transition")
    carrier_only_shift = is_carrier_only_shift(prev.get("visual_carrier_type", ""), nxt.get("visual_carrier_type", ""))
    stage_shift_supported = bool(
        not same_chain
        or nxt.get("contains_new_test_start")
        or nxt.get("contains_new_goal_switch")
        or nxt.get("contains_new_subject_switch")
        or nxt.get("contains_cta_transition")
    )

    if nxt.get("contains_new_test_start"):
        decision = "keep"
        reason_code = "new_test_start"
    elif nxt.get("contains_new_subject_switch"):
        decision = "keep"
        reason_code = "new_subject_switch"
    elif nxt.get("contains_new_goal_switch"):
        decision = "keep"
        reason_code = "new_goal_switch"
    elif nxt.get("contains_cta_transition"):
        decision = "keep"
        reason_code = "cta_transition"
    elif is_stage_shift(prev.get("task_stage", ""), nxt.get("task_stage", "")) and stage_shift_supported:
        decision = "keep"
        reason_code = "task_stage_shift"
    elif same_chain and same_goal_flag and same_task_stage(prev, nxt) and is_same_step_micro_cut(prev, nxt):
        decision = "drop"
        reason_code = "same_step_micro_cut"
    elif same_chain and same_goal_flag and no_new_task and no_new_subject and no_new_goal and no_cta and carrier_only_shift:
        decision = "drop"
        reason_code = "same_argument_chain_continuous_shot"
    elif candidate.get("high_ocr_scene") and mainly_triggered_by_layout_or_ocr(candidate.get("trigger_signals") or []):
        if same_chain and no_new_task and no_new_subject and no_new_goal and no_cta:
            decision = "drop"
            reason_code = "high_ocr_structure_jump_without_semantic_shift"
        else:
            decision = "keep"
            reason_code = "fallback_keep"
    else:
        decision = "keep"
        reason_code = "fallback_keep"

    return {
        "boundary_id": candidate["boundary_id"],
        "protected_sec": candidate["protected_sec"],
        "decision": decision,
        "reason_code": reason_code,
        "same_chain": same_chain,
        "same_goal": same_goal_flag,
        "new_test": bool(nxt.get("contains_new_test_start")),
        "new_subject": bool(nxt.get("contains_new_subject_switch")),
        "new_goal": bool(nxt.get("contains_new_goal_switch")),
        "cta": bool(nxt.get("contains_cta_transition")),
        "prev_carrier": prev.get("visual_carrier_type"),
        "next_carrier": nxt.get("visual_carrier_type"),
        "decision_context": dict(candidate.get("decision_context") or {}),
    }



def build_second_filter_candidate(
    left_segment: dict,
    right_segment: dict,
    *,
    boundary_info: dict,
    adjacent_protected_count_10s: int,
) -> dict:
    prev_semantics = derive_segment_semantics(left_segment)
    next_semantics = derive_segment_semantics(right_segment)
    prev_semantics, next_semantics = _decorate_transition_flags(prev_semantics, next_semantics)
    trigger_signals = list(boundary_info.get("trigger_signals") or [])
    high_ocr_scene = any(signal in LAYOUT_OCR_SIGNALS for signal in trigger_signals)
    rep_metrics = boundary_info.get("representative_metrics") or {}
    same_bundle_relation = (
        "same_bundle"
        if _normalize_text(prev_semantics.get("argument_chain_id")) == _normalize_text(next_semantics.get("argument_chain_id"))
        and _normalize_text(prev_semantics.get("proof_goal")) == _normalize_text(next_semantics.get("proof_goal"))
        else "cross_bundle_candidate"
    )
    return {
        "boundary_id": str(boundary_info.get("boundary_id") or "").strip(),
        "protected_sec": round(float(boundary_info.get("protected_sec") or 0.0), 3),
        "prev_segment_id": left_segment["segment_id"],
        "next_segment_id": right_segment["segment_id"],
        "trigger_signals": trigger_signals,
        "high_ocr_scene": high_ocr_scene,
        "prev_segment_semantics": prev_semantics,
        "next_segment_semantics": next_semantics,
        "decision_context": {
            "candidate_score": rep_metrics.get("score"),
            "adjacent_protected_count_10s": adjacent_protected_count_10s,
            "same_bundle_relation": same_bundle_relation,
            "ocr_jump_strength": 1.0 if "ocr_structure_jump" in trigger_signals else 0.0,
            "layout_migration_strength": 1.0 if "layout_migration" in trigger_signals else 0.0,
        },
    }
