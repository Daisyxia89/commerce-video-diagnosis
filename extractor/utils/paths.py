"""统一的 skill 资源路径解析。

解决问题：fixture / SSOT 等资源路径过去被硬编码为
`user_skills/commerce-video-diagnosis/...`，导致 skill 一旦不在该目录下
（例如直接 clone 到任意位置）就会因 "path 不存在" 而无法运行。

解析规则（按优先级）：
1. 绝对路径：原样返回；
2. 含 `${SKILL_ROOT}` 占位符：替换为实际 skill 根目录；
3. 相对路径：
   a. 先按"相对 skill 根目录"解析；
   b. 若历史写法带有 `user_skills/<skill-name>/` 前缀，自动剥离该前缀后再相对 skill 根解析（向后兼容）；
   c. 最后回退到"相对当前工作目录"（兼容旧行为）。
返回第一个真实存在的路径；都不存在时返回"相对 skill 根目录"的结果，
以便上层给出清晰、可定位的报错。
"""

from __future__ import annotations

from pathlib import Path

# 本文件位于 <skill_root>/extractor/utils/paths.py，向上三级即 skill 根目录。
SKILL_ROOT = Path(__file__).resolve().parents[2]
_SKILL_NAME = SKILL_ROOT.name


def resolve_resource_path(raw_path: str) -> Path:
    """把 config 中声明的资源路径解析为可用的绝对/相对路径。"""
    if not raw_path:
        return Path(raw_path)

    text = str(raw_path).replace("${SKILL_ROOT}", str(SKILL_ROOT))
    candidate = Path(text)

    if candidate.is_absolute():
        return candidate

    candidates: list[Path] = []

    # a. 相对 skill 根目录
    candidates.append(SKILL_ROOT / candidate)

    # b. 历史前缀 user_skills/<skill-name>/ 自动剥离
    parts = candidate.parts
    if len(parts) >= 2 and parts[0] == "user_skills" and parts[1] == _SKILL_NAME:
        candidates.append(SKILL_ROOT / Path(*parts[2:]))
    # 更宽松：任意 user_skills/*/ 前缀
    elif len(parts) >= 2 and parts[0] == "user_skills":
        candidates.append(SKILL_ROOT / Path(*parts[2:]))

    # c. 相对当前工作目录（旧行为兜底）
    candidates.append(candidate)

    for c in candidates:
        if c.exists():
            return c

    # 都不存在：返回相对 skill 根目录的解析，报错信息更可定位
    return candidates[0]
