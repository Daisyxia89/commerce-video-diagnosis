"""/JG 独立验收脚本（依据独立测试用例文档，不参考开发实现逆向调整）。

校验对象：
- outputs/runben_diagnosis/runben_full_diagnosis.json  (product 侧)
- outputs/runben_diagnosis/runben_video_diagnosis.json  (video 侧)

输出：逐项 Pass/Fail + 全量枚举合规扫描。
"""
import json
import sys

# ---- 独立合法枚举集（来源：/JG 测试用例文档 第1节裁决 + 枚举速查） ----
EIGHT_GROUPS = {
    "年长中高消费力女性", "年长低消费力女性", "年轻中高消费力女性", "年轻低消费力女性",
    "年长中高消费力男性", "年长低消费力男性", "年轻中高消费力男性", "年轻低消费力男性",
}
LEGAL = {
    "product.fit_level": {"primary", "secondary", "weak_fit"},
    "video.fit_level": {"primary", "secondary", "mismatch_risk"},
    "axis.age": {"young", "mature", "mixed"},
    "axis.gender": {"female", "male", "mixed"},
    "axis.consumption": {"low", "mid_high", "mixed"},
    "audience_match.match_status": {"high_match", "partial_match", "low_match", "too_broad"},
    "profile.match_status": {"completed", "partial", "weak", "missing", "not_applicable"},
    "hec.match_status": {"good", "acceptable_deviation", "risky_deviation", "mismatch"},
    "slider.match_status": {"fit", "mixed_deviation", "too_strong", "too_weak", "wrong_direction", "mismatch"},
    "slider.axis_fit": {"fit", "too_strong", "too_weak", "wrong_direction"},
    "summary.overall_status": {"good", "needs_minor_repair", "needs_major_repair", "mismatch"},
    "repair.priority": {"P0", "P1", "P2"},
    "repair.issue_type": {"audience", "profile", "hec", "slider"},
}

results = []
def rec(case, ok, detail):
    results.append((case, ok, detail))

full = json.load(open("outputs/runben_diagnosis/runben_full_diagnosis.json"))
vid = json.load(open("outputs/runben_diagnosis/runben_video_diagnosis.json"))
vr = vid["video_persuasion_diagnosis_result"]

def groups(items, with_level=False):
    out = []
    for it in items or []:
        if with_level:
            out.append((it.get("audience_group"), it.get("fit_level")))
        else:
            out.append(it.get("audience_group"))
    return out

# ============ P1-TC-RB: product_target_audience（润本，必测）============
pta = full["product_target_audience"]
prim = groups(pta.get("primary_audiences"), True)
sec = groups(pta.get("secondary_audiences"), True)
weak = groups(pta.get("weak_fit_audiences"), True)
rc = pta.get("reasoning_chain", {})
ok = (prim == [("年长中高消费力女性", "primary")]
      and sec == [("年长低消费力女性", "secondary")]
      and not weak
      and all(k in rc for k in ("task_to_role", "role_category_to_age_gender", "brand_price_to_consumption_power")))
rec("P1-TC-RB product_target_audience", ok,
    f"primary={prim}; secondary={sec}; weak={weak}; reasoning_chain_keys={list(rc.keys())}")

# ============ P2-TC-RB: video_target_audience（润本，必测）============
vta = vr["video_target_audience"]
vp = set(groups(vta.get("primary_audiences")))
vs = set(groups(vta.get("secondary_audiences")))
vm = set(groups(vta.get("mismatch_risk_audiences")))
aj = vta.get("axis_judgment", {})
cons = aj.get("consumption_power_axis", {}).get("value")
exp_p = {"年长低消费力女性", "年长中高消费力女性"}
exp_s = {"年轻低消费力女性", "年轻中高消费力女性"}
exp_m = {"年轻低消费力男性"}
ok = (vp == exp_p and vs == exp_s and vm == exp_m and cons == "mixed"
      and all(k in aj for k in ("age_axis", "gender_axis", "consumption_power_axis")))
rec("P2-TC-RB video_target_audience", ok,
    f"primary={sorted(vp)}(期望{sorted(exp_p)}); secondary={sorted(vs)}(期望{sorted(exp_s)}); "
    f"mismatch_risk={sorted(vm)}(期望{sorted(exp_m)}); consumption_axis={cons}(期望mixed)")

# axis_judgment 存在性（P2-AC2/D4）
rec("P2-axis_judgment 三轴存在", all(k in aj for k in ("age_axis", "gender_axis", "consumption_power_axis")),
    f"axis_judgment keys={list(aj.keys())}")

# ============ P3-TC-RB: 四步 Match（润本，必测）============
am = vr["audience_match_diagnosis"]["match_status"]
rec("P3-TC-RB Audience Match = high_match", am == "high_match", f"actual={am}")

pm = vr["profile_match_diagnosis"]
# D1: 字段必须更名为 match_status，旧字段名 overall_status 判 Fail
has_match = "match_status" in pm
has_old = "overall_status" in pm
pm_val = pm.get("match_status")
rec("P3-TC-RB Profile 字段名=match_status(D1)", has_match and not has_old,
    f"keys={list(pm.keys())}; 旧字段名overall_status存在={has_old}")
rec("P3-TC-RB Profile match_status=partial", pm_val == "partial",
    f"match_status={pm_val}; (兼容读 overall_status={pm.get('overall_status')}); 期望 partial")

hec = vr["hec_match_diagnosis"]["match_status"]
rec("P3-TC-RB HEC = acceptable_deviation", hec == "acceptable_deviation", f"actual={hec}; 期望 acceptable_deviation")

sm = vr["slider_match_diagnosis"]
sm_status = sm["match_status"]
axis_map = {a["axis"]: a["fit_status"] for a in sm["axis_results"]}
ok = (sm_status == "mixed_deviation"
      and axis_map.get("visual") == "too_strong"
      and axis_map.get("cta") == "too_weak"
      and axis_map.get("proof") == "fit"
      and axis_map.get("audio") == "fit")
rec("P3-TC-RB Slider = mixed_deviation(visual too_strong + cta too_weak)", ok,
    f"match_status={sm_status}; axis={axis_map}; sig={sm.get('actual_slider_signature')}")

# ============ P4-TC-RB: 总输出（润本，必测）============
ds = vr["diagnosis_summary"]
rec("P4-TC-RB overall_status = needs_minor_repair", ds["overall_status"] == "needs_minor_repair",
    f"actual={ds['overall_status']}; priority_issues={ds.get('priority_issues')}")
# repair_suggestions 结构 AC7
rs_ok = all(all(k in s for k in ("priority", "issue_type", "issue_summary", "repair_direction", "related_evidence_spans"))
            for s in ds.get("repair_suggestions", []))
rec("P4 repair_suggestions 字段齐全(AC7)", rs_ok, f"count={len(ds.get('repair_suggestions',[]))}")

# ============ 全量枚举合规扫描 ============
enum_fail = []
def chk(path, value, key):
    if value is not None and value not in LEGAL[key]:
        enum_fail.append(f"{path}={value!r} 不在 {key} 合法集")

# product
for a in pta.get("primary_audiences", []) + pta.get("secondary_audiences", []) + pta.get("weak_fit_audiences", []):
    chk("product.audience_group", a.get("audience_group"), None) if False else None
    if a.get("audience_group") not in EIGHT_GROUPS:
        enum_fail.append(f"product.audience_group={a.get('audience_group')!r} 越八大人群枚举")
    chk("product.fit_level", a.get("fit_level"), "product.fit_level")
# video audience
for a in vta.get("primary_audiences", []) + vta.get("secondary_audiences", []) + vta.get("mismatch_risk_audiences", []):
    if a.get("audience_group") not in EIGHT_GROUPS:
        enum_fail.append(f"video.audience_group={a.get('audience_group')!r} 越八大人群枚举")
    chk("video.fit_level", a.get("fit_level"), "video.fit_level")
chk("axis.age", aj.get("age_axis", {}).get("value"), "axis.age")
chk("axis.gender", aj.get("gender_axis", {}).get("value"), "axis.gender")
chk("axis.consumption", aj.get("consumption_power_axis", {}).get("value"), "axis.consumption")
chk("audience_match.match_status", am, "audience_match.match_status")
# profile：字段名校验 + 值校验
if has_old and not has_match:
    enum_fail.append("profile 使用旧字段名 overall_status（D1 要求 match_status）")
chk("profile.match_status", pm.get("match_status") or pm.get("overall_status"), "profile.match_status")
for r in pm.get("requirement_results", []):
    chk("profile.completion_status", r.get("completion_status"), "profile.match_status")
chk("hec.match_status", hec, "hec.match_status")
chk("slider.match_status", sm_status, "slider.match_status")
for a in sm["axis_results"]:
    chk("slider.axis_fit_status", a.get("fit_status"), "slider.axis_fit")
chk("summary.overall_status", ds["overall_status"], "summary.overall_status")
for s in ds.get("repair_suggestions", []):
    chk("repair.priority", s.get("priority"), "repair.priority")
    chk("repair.issue_type", s.get("issue_type"), "repair.issue_type")

# segments 禁用字段扫描
blob = json.dumps({"full": full, "vid": vid}, ensure_ascii=False)
seg_hit = '"segments"' in blob
if seg_hit:
    enum_fail.append("输出中出现禁用字段名 segments")

rec("全量枚举合规扫描（含字段名/八大人群/segments）", not enum_fail,
    ("无越界" if not enum_fail else " | ".join(enum_fail)))

# ============ 汇总 ============
print("=" * 80)
passed = sum(1 for _, ok, _ in results if ok)
total = len(results)
for case, ok, detail in results:
    print(f"[{'PASS' if ok else 'FAIL'}] {case}")
    print(f"        {detail}")
print("=" * 80)
print(f"独立验收（润本必测 + 枚举扫描）: {passed}/{total} 通过")
sys.exit(0 if passed == total else 1)
