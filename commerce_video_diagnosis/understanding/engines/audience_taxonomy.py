"""抖音电商八大人群坐标共享分类（taxonomy）。

仅承载枚举与坐标组合这种纯结构能力，供商品侧 `product_target_audience`
与视频侧 `video_target_audience` 共用。两侧的业务判定逻辑各自独立实现，
不在此处共享，符合 PRD「video_target_audience 不得继承 product_target_audience」约束。
"""
from __future__ import annotations

from typing import Literal

# 八大人群枚举（顺序与 PRD §9.5 / video PRD 一致）
EIGHT_AUDIENCE_GROUPS: tuple[str, ...] = (
    "年轻中高消费力女性",
    "年轻中高消费力男性",
    "年轻低消费力女性",
    "年轻低消费力男性",
    "年长中高消费力女性",
    "年长中高消费力男性",
    "年长低消费力女性",
    "年长低消费力男性",
)

AGE_LABELS = {"young": "年轻", "mature": "年长"}
GENDER_LABELS = {"female": "女性", "male": "男性"}
CONSUMPTION_LABELS = {"mid_high": "中高消费力", "low": "低消费力"}

AgeAxis = Literal["young", "mature", "mixed"]
GenderAxis = Literal["female", "male", "mixed"]
ConsumptionAxis = Literal["mid_high", "low", "mixed"]


class AudienceTaxonomyError(ValueError):
    """坐标组合非法时抛出，用于 Crash Early。"""


def compose_audience_group(age: str, gender: str, consumption: str) -> str:
    """把三轴确定值组合成八大人群枚举字符串。

    任一轴为 mixed / 空 / 非法时直接抛错（Crash Early），调用方需先把 mixed
    展开成确定的多个坐标后再逐一组合。
    """
    if age not in AGE_LABELS:
        raise AudienceTaxonomyError(f"年龄轴必须是 young/mature，收到: {age!r}")
    if gender not in GENDER_LABELS:
        raise AudienceTaxonomyError(f"性别轴必须是 female/male，收到: {gender!r}")
    if consumption not in CONSUMPTION_LABELS:
        raise AudienceTaxonomyError(f"消费力轴必须是 mid_high/low，收到: {consumption!r}")
    group = f"{AGE_LABELS[age]}{CONSUMPTION_LABELS[consumption]}{GENDER_LABELS[gender]}"
    if group not in EIGHT_AUDIENCE_GROUPS:
        raise AudienceTaxonomyError(f"组合得到非法八大人群: {group!r}")
    return group


def expand_axis(value: str, *, axis: str) -> list[str]:
    """把单轴取值展开成确定取值列表。mixed -> 两个确定值。"""
    if axis == "age":
        if value == "mixed":
            return ["young", "mature"]
        if value in AGE_LABELS:
            return [value]
    elif axis == "gender":
        if value == "mixed":
            return ["female", "male"]
        if value in GENDER_LABELS:
            return [value]
    elif axis == "consumption":
        if value == "mixed":
            return ["mid_high", "low"]
        if value in CONSUMPTION_LABELS:
            return [value]
    raise AudienceTaxonomyError(f"非法轴值 axis={axis} value={value!r}")
