"""
Trace script for B20_N02 sample: 卸妆膏 misrouted to 生存/运转维系.
Instruments key methods to capture the full execution path.
"""
import sys
import os
import json

sys.path.insert(0, "/workspace/iris_72fb6608-39b2-4dd3-9702-0a5f8a377abb/commerce-video-diagnosis")
os.chdir("/workspace/iris_72fb6608-39b2-4dd3-9702-0a5f8a377abb/commerce-video-diagnosis")

from commerce_video_diagnosis.understanding.engines.product_diagnoser import (
    ProductDiagnosisEngine,
    MAINTENANCE_SUPPLY_TOKENS,
    ORDINARY_DAILY_TOKENS,
    EFFICIENCY_TOKENS,
    OPERATION_EASE_TOKENS,
    STRONG_DEFECT_TOKENS,
    DEFECT_STATE_TOKENS,
    DEFECT_REMEDIATION_TOKENS,
    HOUSEHOLD_STAGEB_SUBCATEGORY_PACKS,
    PERSONAL_CARE_STAGEB_SUBCATEGORY_CONTEXTS,
    PERSONAL_CARE_COMMON_OBJECT_TOKENS,
    PERSONAL_CARE_COMMON_STATE_TOKENS,
    PERSONAL_CARE_COMMON_ACTION_TOKENS,
)

# ======== Trace storage ========
trace_log = []

def log(section, msg):
    entry = f"[{section}] {msg}"
    trace_log.append(entry)
    print(entry)


# ======== Payload ========
payload = dict(
    leaf_category="卸妆膏", shop_name="YOUFE旗舰店",
    product_name="YOUFE净透卸妆膏小金砖快乳化眼唇脸清洁油乳",
    price="49.9",
    core_selling_point="快乳化卸妆，省时一步到位清洁眼唇脸，日常基础卸妆",
    core_selling_point_source="caller_provided.core_selling_points",
    target_people="用过老款的粉丝及有高效卸妆需求的美妆爱好者",
    differentiator="",
    bridge_comparison_object="", bridge_comparison_object_evidence_type="null",
    bridge_difference_domain="functional", bridge_difference_type="自身卖点陈述",
    bridge_evidence_source="商品信息",
    bridge_source_evidence=["YOUFE净透卸妆膏小金砖快乳化眼唇脸清洁油乳","快乳化卸妆，省时一步到位清洁眼唇脸，日常基础卸妆"],
    engine_node={"relative_price_level":"低水位"},
)

# ======== Monkeypatch key methods ========
engine = ProductDiagnosisEngine()

# 1. Trace _build_rule_tree_context
original_build_rule_tree_context = engine._build_rule_tree_context
def traced_build_rule_tree_context(payload_arg, module1_output):
    result = original_build_rule_tree_context(payload_arg, module1_output)
    log("_build_rule_tree_context", f"candidate_tasks = {result['candidate_tasks']}")
    log("_build_rule_tree_context", f"candidate_reasons = {json.dumps(result.get('candidate_reasons', {}), ensure_ascii=False)}")
    log("_build_rule_tree_context", f"excluded_tasks = {json.dumps(result.get('excluded_tasks', {}), ensure_ascii=False)}")
    log("_build_rule_tree_context", f"triggered_rule = {result.get('triggered_rule')}")
    log("_build_rule_tree_context", f"subcategory_context = {result.get('subcategory_context')}")
    log("_build_rule_tree_context", f"trace_tokens = {result.get('trace_tokens')}")
    log("_build_rule_tree_context", f"functional_facts = {json.dumps(result.get('functional_facts', []), ensure_ascii=False, indent=2)}")
    log("_build_rule_tree_context", f"candidate_pool = {json.dumps(result.get('candidate_pool', []), ensure_ascii=False, indent=2)}")
    log("_build_rule_tree_context", f"veto_trace = {result.get('veto_trace')}")
    log("_build_rule_tree_context", f"reasoning_path = {result.get('reasoning_path')}")
    return result
engine._build_rule_tree_context = traced_build_rule_tree_context

# 2. Trace _resolve_jtbd
original_resolve_jtbd = engine._resolve_jtbd
def traced_resolve_jtbd(payload_arg, module1_output):
    result = original_resolve_jtbd(payload_arg, module1_output)
    proposal = result[0]
    log("_resolve_jtbd", f"proposal.primary_task = {proposal.primary_task}")
    log("_resolve_jtbd", f"proposal.domain = {proposal.domain}")
    log("_resolve_jtbd", f"proposal.candidate_tasks = {proposal.candidate_tasks}")
    log("_resolve_jtbd", f"len(candidate_tasks) = {len(proposal.candidate_tasks)}")
    log("_resolve_jtbd", f"proposal.reasoning = {proposal.reasoning}")
    log("_resolve_jtbd", f"单候选直接定 or LLM选? = {'单候选直接定' if len(proposal.candidate_tasks) == 1 else 'LLM分类器'}")
    return result
engine._resolve_jtbd = traced_resolve_jtbd

# 3. Trace _infer_functional_candidates
original_infer_functional = engine._infer_functional_candidates
def traced_infer_functional(module1_output, text):
    result = original_infer_functional(module1_output, text)
    log("_infer_functional_candidates", f"candidate_tasks = {result['candidate_tasks']}")
    log("_infer_functional_candidates", f"subcategory_context = {result.get('subcategory_context')}")
    log("_infer_functional_candidates", f"functional_facts count = {len(result.get('functional_facts', []))}")
    log("_infer_functional_candidates", f"candidate_pool = {json.dumps(result.get('candidate_pool', []), ensure_ascii=False, indent=2)}")
    return result
engine._infer_functional_candidates = traced_infer_functional

# 4. Trace _resolve_household_subcategory_context
original_resolve_subcategory = engine._resolve_household_subcategory_context
def traced_resolve_subcategory(module1_output, text):
    result = original_resolve_subcategory(module1_output, text)
    log("_resolve_household_subcategory_context", f"result = {result}")
    # Show what category_text was
    category_text = "｜".join(
        value for value in [
            module1_output.second_level_category,
            module1_output.third_level_category,
            module1_output.leaf_category,
        ] if value
    )
    log("_resolve_household_subcategory_context", f"category_text = '{category_text}'")
    pack = HOUSEHOLD_STAGEB_SUBCATEGORY_PACKS.get("cleanse_protection", {})
    cat_terms = pack.get("category_terms", set())
    log("_resolve_household_subcategory_context", f"cleanse_protection category_terms = {cat_terms}")
    matches = [t for t in cat_terms if t in category_text]
    log("_resolve_household_subcategory_context", f"cleanse_protection matches in category_text = {matches}")
    return result
engine._resolve_household_subcategory_context = traced_resolve_subcategory

# 5. Trace _build_cleanse_protection_candidate_pool
original_build_cleanse = engine._build_cleanse_protection_candidate_pool
def traced_build_cleanse(sub_pack_facts):
    log("_build_cleanse_protection_candidate_pool", f"sub_pack_facts = {json.dumps(sub_pack_facts, ensure_ascii=False, indent=2)}")
    result = original_build_cleanse(sub_pack_facts)
    log("_build_cleanse_protection_candidate_pool", f"candidate_pool = {json.dumps(result[0], ensure_ascii=False, indent=2)}")
    log("_build_cleanse_protection_candidate_pool", f"veto_trace = {result[1]}")
    # Show maintenance vs defect split
    for fact in sub_pack_facts:
        log("_build_cleanse_protection_candidate_pool",
            f"  fact '{fact.get('fact_id')}': problem_state='{fact.get('problem_state')}', "
            f"action='{fact.get('action_mechanism')}', object='{fact.get('problem_object')}' "
            f"=> {'维持' if fact.get('problem_state') == '维持正常' else '缺陷'}"
        )
    return result
engine._build_cleanse_protection_candidate_pool = traced_build_cleanse

# 6. Trace _supports_maintenance_task
original_supports_maintenance = engine._supports_maintenance_task
def traced_supports_maintenance(module1_output, text):
    result = original_supports_maintenance(module1_output, text)
    log("_supports_maintenance_task", f"result = {result}")
    # Show sub-checks
    has_strong_defect = engine._has_strong_defect_signal(module1_output, text)
    log("_supports_maintenance_task", f"  _has_strong_defect_signal = {has_strong_defect}")
    category_text = f"{module1_output.leaf_category} {module1_output.product_name}"
    is_food = engine._is_food_like_category(category_text)
    log("_supports_maintenance_task", f"  _is_food_like_category = {is_food}")
    maint_tokens_hit = [t for t in MAINTENANCE_SUPPLY_TOKENS if t in text]
    log("_supports_maintenance_task", f"  MAINTENANCE_SUPPLY_TOKENS hits in text = {maint_tokens_hit}")
    is_ordinary = engine._is_ordinary_daily_category(module1_output)
    log("_supports_maintenance_task", f"  _is_ordinary_daily_category = {is_ordinary}")
    diff_type = module1_output.differentiator.difference_type
    log("_supports_maintenance_task", f"  difference_type = {diff_type}")
    return result
engine._supports_maintenance_task = traced_supports_maintenance

# 7. Trace _is_ordinary_daily_category
original_is_ordinary = engine._is_ordinary_daily_category
def traced_is_ordinary(module1_output):
    result = original_is_ordinary(module1_output)
    text = f"{module1_output.leaf_category} {module1_output.product_name} {module1_output.core_selling_point}"
    hits = [t for t in ORDINARY_DAILY_TOKENS if t in text]
    log("_is_ordinary_daily_category", f"result = {result}, text snippet = '{text[:80]}...', hits = {hits}")
    return result
engine._is_ordinary_daily_category = traced_is_ordinary

# 8. Trace _extract_subcategory_facts (for cleanse_protection)
original_extract_subcategory_facts = engine._extract_subcategory_facts
def traced_extract_subcategory_facts(subcategory_context, clause, module1_output):
    result = original_extract_subcategory_facts(subcategory_context, clause, module1_output)
    if result:
        log("_extract_subcategory_facts", f"subcategory_context={subcategory_context}, clause='{clause[:60]}...', facts={json.dumps(result, ensure_ascii=False, indent=2)}")
    return result
engine._extract_subcategory_facts = traced_extract_subcategory_facts

# 9. Trace _build_subcategory_fact_from_group
original_build_fact_from_group = engine._build_subcategory_fact_from_group
def traced_build_fact_from_group(subcategory_context, clause, module1_output, pack, group):
    result = original_build_fact_from_group(subcategory_context, clause, module1_output, pack, group)
    if subcategory_context == "cleanse_protection":
        # Show what matched
        object_matches = engine._extract_keyword_matches(clause, group.get("problem_object_terms") or pack.get("problem_object_terms", {}))
        state_matches = engine._extract_keyword_matches(clause, group.get("problem_state_terms") or pack.get("problem_state_terms", {}))
        action_matches = engine._extract_keyword_matches(clause, group.get("action_mechanism_terms") or pack.get("action_mechanism_terms", {}))
        log("_build_subcategory_fact_from_group",
            f"clause='{clause[:60]}...'\n"
            f"    object_matches={object_matches}\n"
            f"    state_matches={state_matches}\n"
            f"    action_matches={action_matches}\n"
            f"    allow_action_only_without_object={pack.get('allow_action_only_without_object')}\n"
            f"    default_state_when_action_only='{pack.get('default_state_when_action_only')}'\n"
            f"    result={'FORMED' if result else 'NONE'}")
        if result:
            log("_build_subcategory_fact_from_group", f"    => problem_state='{result.get('problem_state')}', action='{result.get('action_mechanism')}'")
    return result
engine._build_subcategory_fact_from_group = traced_build_fact_from_group

# 10. Trace _apply_hard_gates
original_apply_hard_gates = engine._apply_hard_gates
def traced_apply_hard_gates(payload_arg, module1_output, proposal):
    log("_apply_hard_gates", f"INPUT proposal.primary_task = {proposal.primary_task}")
    result = original_apply_hard_gates(payload_arg, module1_output, proposal)
    gated = result[0]
    log("_apply_hard_gates", f"OUTPUT gated.primary_task = {gated.primary_task}")
    log("_apply_hard_gates", f"gate_notes = {result[1]}")
    return result
engine._apply_hard_gates = traced_apply_hard_gates

# ======== Also show the joined text ========
original_module1_joined_text = engine._module1_joined_text

# ======== Run ========
log("MAIN", "Starting B20_N02 trace run...")

# Show key token sets content for reference
log("TOKEN_SETS", f"ORDINARY_DAILY_TOKENS = {sorted(ORDINARY_DAILY_TOKENS)}")
log("TOKEN_SETS", f"MAINTENANCE_SUPPLY_TOKENS = {sorted(MAINTENANCE_SUPPLY_TOKENS)}")
log("TOKEN_SETS", f"cleanse_protection pack action_only_terms = {HOUSEHOLD_STAGEB_SUBCATEGORY_PACKS['cleanse_protection'].get('action_only_terms')}")
log("TOKEN_SETS", f"cleanse_protection default_state_when_action_only = {HOUSEHOLD_STAGEB_SUBCATEGORY_PACKS['cleanse_protection'].get('default_state_when_action_only')}")

out = engine.diagnose(payload)
log("MAIN", f"\n{'='*60}")
log("MAIN", f"FINAL jtbd = {out.jtbd}")
out_dict = out.dict()
log("MAIN", f"FINAL jtbd field in output = {out_dict.get('jtbd')}")
log("MAIN", f"FINAL RESULT (truncated):\n{json.dumps(out_dict, ensure_ascii=False, indent=2)[:3000]}")

# Write output
output_path = "/workspace/iris_72fb6608-39b2-4dd3-9702-0a5f8a377abb/commerce-video-diagnosis/jg_jtbd_expansion/trace_b20n02_output.txt"
with open(output_path, "w", encoding="utf-8") as f:
    f.write("\n".join(trace_log))

print(f"\n\nTrace saved to: {output_path}")
