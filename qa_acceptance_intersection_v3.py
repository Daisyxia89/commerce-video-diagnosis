"""独立 QA 验收脚本（/JG 出题）
针对 module3_intent_derivation._ordered_intersection 与
persuasion_requirement_engine._build_main_route 改造的 PRD 8.5 合规校验。

不依赖研发自测，直接基于 PRD 第 8.5 节字面语义独立编写。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from commerce_video_diagnosis.understanding.module3_intent_derivation import (  # noqa: E402
    _ordered_intersection,
)

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, cond: bool, observed: str) -> None:
    RESULTS.append((name, cond, observed))


# -------- A1: 严格两轴交集 ---------------------------------------------------
pool = ["E1", "E2", "E3", "E4", "E5", "E6"]
R = ("E3", "E1", "E6", "E7")           # R 轴偏好（含池外 E7）
P = ("E6", "E1", "E3", "E9")            # P 轴偏好（含池外 E9）
out = _ordered_intersection(pool, R, P)
expected = ["E3", "E1", "E6"]  # 顺序按 R 优先
check("A1.严格两轴交集仅保留三方共有", set(out) == set(expected), f"out={out}")
check("A1.无单轴覆盖条目入选", all(c in set(R) & set(P) & set(pool) for c in out), f"out={out}")

# -------- A2: 排序为 R 优先，P 次之 ------------------------------------------
check("A2.排序按 R 轴顺序优先", out == expected, f"out={out} expected={expected}")

# 进一步：相同 R 排名时按 P 轴次序
pool2 = ["X", "Y", "Z"]
R2 = ("X", "Y", "Z")
P2 = ("Z", "Y", "X")
out2 = _ordered_intersection(pool2, R2, P2)
check("A2.R 优先于 P", out2 == ["X", "Y", "Z"], f"out2={out2}")

# -------- A6: 修饰符强插之后再次执行幂等 -------------------------------------
# _ordered_intersection 是纯函数：同输入必同输出，且修饰符强插发生在交集之后
# （见 module3_intent_derivation._cross_map_weapon_pool 调用顺序）。
try:
    pool_idem = ["E1", "E3", "E6"]
    R_idem = ("E3", "E1")
    P_idem = ("E1", "E3")
    r1 = _ordered_intersection(pool_idem, R_idem, P_idem)
    r2 = _ordered_intersection(pool_idem, R_idem, P_idem)
    check("A6.同输入两次调用结果一致（幂等）", r1 == r2, f"r1={r1} r2={r2}")
    # 修饰符强插不污染交集函数本身：交集结果中不会出现池外条目（如 E6 不在 R/P 偏好里）
    raw = _ordered_intersection(["E1", "E3", "E6"], ("E1", "E3"), ("E3", "E1"))
    check("A6.修饰符不污染纯交集（E6 未在两轴偏好中故不入选）", "E6" not in raw, f"raw={raw}")
except Exception as e:  # noqa: BLE001
    check("A6.修饰符强插幂等", False, f"exception: {e!r}")

# -------- A7: 空 tuple 视为该轴不过滤 ---------------------------------------
out_empty_R = _ordered_intersection(["E1", "E2"], (), ("E2",))
check("A7.R 轴空 tuple 不约束 → 仅 P 轴过滤", out_empty_R == ["E2"], f"out={out_empty_R}")

out_empty_P = _ordered_intersection(["E1", "E2"], ("E1",), ())
check("A7.P 轴空 tuple 不约束 → 仅 R 轴过滤", out_empty_P == ["E1"], f"out={out_empty_P}")

out_empty_both = _ordered_intersection(["E1", "E2"], (), ())
check("A7.两轴均空 → 不过滤", set(out_empty_both) == {"E1", "E2"}, f"out={out_empty_both}")

# -------- A3/A4/A5: 润本诊断成品文件断言 -------------------------------------
diag_path = ROOT / "outputs/runben_diagnosis/runben_full_diagnosis.json"
with diag_path.open(encoding="utf-8") as fh:
    d = json.load(fh)
ci = d["core_intent"]
core_e = [x["code"] for x in ci["core_e"]]
core_c = [x["code"] for x in ci["core_c"]]
check("A3.润本 Core E-list = [E3,E1,E6]", core_e == ["E3", "E1", "E6"], f"observed={core_e}")
check("A3.润本 Core C-list = [C1,C4]", core_c == ["C1", "C4"], f"observed={core_c}")

product_hecs = d["product_hecs"]
ec_keys = {(h["effect_tag"], h["cta_tag"]) for h in product_hecs}
check("A4.润本 EC 主链数量 = 6", len(ec_keys) == 6, f"observed={sorted(ec_keys)} (count={len(ec_keys)})")

cat_res = d["persuasion_requirement_profile"]["main_persuasion_route"]["category_resistance"]
check(
    "A5.润本 category_resistance.rule = '红海-核心 × 快消'",
    cat_res.get("rule") == "红海-核心 × 快消",
    f"observed={cat_res.get('rule')!r}",
)
check(
    "A5.不含'认知门槛'字样",
    "认知门槛" not in json.dumps(cat_res, ensure_ascii=False),
    f"cat_res={cat_res}",
)

# 同步检查 slim 文件存在性 + 报告存在性
slim_path = ROOT / "outputs/runben_diagnosis/runben_slim_diagnosis.json"
report_path = ROOT / "outputs/runben_diagnosis/runben_persuasion_profile_report.md"
check("产物.slim 文件存在", slim_path.exists(), str(slim_path))
check("产物.report 文件存在", report_path.exists(), str(report_path))

# ---------------------- 汇总 -------------------------------------------------
print("\n=== /JG 独立验收结果 ===")
ok = 0
for name, cond, obs in RESULTS:
    tag = "PASS" if cond else "FAIL"
    if cond:
        ok += 1
    print(f"[{tag}] {name}  | {obs}")
total = len(RESULTS)
print(f"\n汇总: {ok}/{total} PASS")
print("最终结论:", "PASS" if ok == total else "FAIL")
sys.exit(0 if ok == total else 1)
