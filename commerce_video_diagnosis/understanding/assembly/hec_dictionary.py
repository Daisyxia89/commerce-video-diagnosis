"""HEC 标签权威字典加载器（code → {code, name, definition}）。

唯一权威源（后端供给，前端不得自维护）：
    ``understanding/memory/topics/taxonomy_dictionary_v2.md``
    （§1.2 Hook / §2.2 Effect / §3.2 CTA「定义与核心边界」）。

设计原则（与 memory 铁律 / Crash Early 一致）：
- 启动期一次性加载并 ``lru_cache``；不做运行期热更新。
- 文件缺失 / 解析为空 / 某条目缺定义文本 → Crash Early（``HECDictionaryError``）。
- 查表时 code 未命中 → Crash Early；**禁止编造 definition、禁止空字符串占位**。

注：本加载器只读取「定义」正文（不含「核心边界 / 绝对边界 / 主体边界铁律」），
``name``/``definition`` 逐字来源于字典，不做语义改写。
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

# assembly/ -> understanding/ ，与 module3_intent_derivation / video_understanding 口径一致
_TAXONOMY_PATH = (
    Path(__file__).resolve().parents[1] / "memory" / "topics" / "taxonomy_dictionary_v2.md"
)

# 标签条目头，例如：``*   **H1 痛点/焦虑直击**``
_HEADER_RE = re.compile(r"^\s*\*\s+\*\*([HEC]\d)\s+(.+?)\*\*\s*$")
# 「定义」起始行，例如：``    *   **定义**：诉诸避害。...``
_DEF_RE = re.compile(r"^\s*\*\s+\*\*定义\*\*\s*[:：]\s*(.*)$")
# 终止标记（同级 bullet）：核心边界 / 绝对边界 / 主体边界铁律 —— 定义正文到此为止
_STOP_RE = re.compile(r"^\s*\*\s+\*\*(核心边界|绝对边界|主体边界)")


class HECDictionaryError(RuntimeError):
    """HEC 权威字典加载 / 查表失败（Crash Early，禁止编造）。"""


@lru_cache(maxsize=1)
def _load_hec_dictionary() -> dict[str, dict[str, str]]:
    """解析 taxonomy_dictionary_v2.md，返回 ``{code: {code, name, definition}}``。"""
    if not _TAXONOMY_PATH.exists():
        raise HECDictionaryError(f"HEC 权威字典缺失：{_TAXONOMY_PATH}")

    lines = _TAXONOMY_PATH.read_text(encoding="utf-8").splitlines()
    mapping: dict[str, dict[str, str]] = {}

    cur_code: str | None = None
    cur_name = ""
    def_parts: list[str] = []
    collecting = False

    def _flush() -> None:
        nonlocal cur_code, cur_name, def_parts, collecting
        if cur_code is not None:
            definition = " ".join(p.strip() for p in def_parts if p.strip()).strip()
            if not cur_name:
                raise HECDictionaryError(f"HEC 字典标签 {cur_code} 缺少业务名称（解析为空）。")
            if not definition:
                raise HECDictionaryError(f"HEC 字典标签 {cur_code} 缺少定义文本（解析为空）。")
            if cur_code in mapping:
                raise HECDictionaryError(f"HEC 字典标签 {cur_code} 重复定义（字典源异常）。")
            mapping[cur_code] = {
                "code": cur_code,
                "name": cur_name,
                "definition": definition,
            }
        cur_code = None
        cur_name = ""
        def_parts = []
        collecting = False

    for line in lines:
        header = _HEADER_RE.match(line)
        if header:
            _flush()
            cur_code = header.group(1).strip()
            cur_name = header.group(2).strip()
            def_parts = []
            collecting = False
            continue
        if cur_code is None:
            continue
        def_match = _DEF_RE.match(line)
        if def_match:
            collecting = True
            def_parts = [def_match.group(1).strip()]
            continue
        if _STOP_RE.match(line):
            collecting = False
            continue
        if collecting:
            # 定义下的嵌套子标签行（如 H5-1 / H6-1）逐字并入定义正文，去掉 markdown 标记
            cleaned = line.strip().lstrip("*").strip().replace("*", "")
            if cleaned:
                def_parts.append(cleaned)
    _flush()

    if not mapping:
        raise HECDictionaryError(f"HEC 权威字典解析为空（格式非法）：{_TAXONOMY_PATH}")
    return mapping


def lookup_hec(code: str) -> dict[str, str]:
    """按标签 code（如 ``H1`` / ``E6`` / ``C4``）回查 ``{code, name, definition}``。

    code 为空或未命中字典一律 Crash Early（禁止编造 definition）。返回副本，调用方可安全修改。
    """
    code_norm = str(code or "").strip().upper()
    if not code_norm:
        raise HECDictionaryError("HEC code 为空，无法回查权威字典。")
    table = _load_hec_dictionary()
    entry = table.get(code_norm)
    if entry is None:
        raise HECDictionaryError(
            f"HEC code {code_norm!r} 未命中权威字典（{_TAXONOMY_PATH.name}）；"
            f"禁止编造 definition / 占位，已 Crash Early。"
        )
    return dict(entry)
