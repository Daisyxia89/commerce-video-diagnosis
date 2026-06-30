from __future__ import annotations

import csv
import json
import os
import re
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Optional

import requests
from commerce_video_diagnosis.understanding.llm_provider import build_chat_headers, require_llm_config, resolve_llm_config
from pydantic import BaseModel, Field, root_validator, validator

from commerce_video_diagnosis.understanding.engines.product_variant_assembler import ProductVariantAssembler
from commerce_video_diagnosis.understanding.engines.persuasion_requirement_engine import PersuasionRequirementEngine
from commerce_video_diagnosis.understanding.engines.audience_taxonomy import (
    AudienceTaxonomyError,
    CONSUMPTION_LABELS,
    compose_audience_group,
    expand_axis,
)
from commerce_video_diagnosis.understanding.module3_intent_derivation import (
    Module3IntentInput,
    derive_candidate_set,
    derive_category_strategy_intent,
    derive_product_strategy_intent,
)
from commerce_video_diagnosis.understanding.validators.schema_assertions import VALID_JTBD, assert_product_diagnosis
from commerce_video_diagnosis.understanding.keyword_rules import assert_rule_trace, build_rule_trace, get_mapping_of_string_lists, get_string_list
from commerce_video_diagnosis.understanding.gift_context import detect_gift_context

ROOT = Path(__file__).resolve().parents[1]

FUNCTIONAL_DOMAIN = "功能域"
EMOTIONAL_DOMAIN = "情绪域"
SOCIAL_DOMAIN = "社会域"
TITLE_CORE_SELLING_POINT_ALLOWED_SOURCES = {
    "title_llm_extracted",
    "caller_provided.core_selling_points",
}
FIELD_PROVENANCE_ALLOWED_VALUES = {
    "caller_product_info",
    "product_detail",
    "video_extracted_candidate",
}
BLOCKED_CORE_SELLING_POINT_PROVENANCE = "video_extracted_candidate"


def _normalize_compact_text(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"[\s\-_,，。；;：:【】\[\]\(\)（）/]+", "", text)
    return text.lower()


TASK_DOMAIN_MAP: dict[str, str] = {
    "生存/运转维系": FUNCTIONAL_DOMAIN,
    "缺陷修复/冲突消除": FUNCTIONAL_DOMAIN,
    "降本增效/懒人替代": FUNCTIONAL_DOMAIN,
    "物理安全与风险规避": FUNCTIONAL_DOMAIN,
    "情绪安心/主观降险": EMOTIONAL_DOMAIN,
    "新奇探索/瞬时刺激": EMOTIONAL_DOMAIN,
    "自我犒赏与秩序掌控": EMOTIONAL_DOMAIN,
    "照护与责任履行": SOCIAL_DOMAIN,
    "礼赠与关系表达": SOCIAL_DOMAIN,
    "圈层认同（圈层归属/身份锚定）": SOCIAL_DOMAIN,
    "阶层与审美发信": SOCIAL_DOMAIN,
}

ALLOWED_SUB_TASKS = {"身份跃迁", "审美/阶层标榜"}

# =============================================================================
# 商品目标人群（product_target_audience）确定性派生所需常量（Block 1.1）
# 全部为代码内显式表，禁止依赖 LLM；任务/角色为精确表，关键词为人群轴信号。
# =============================================================================
# Step1：唯一任务 -> 购买角色（精确表，键为代码内 11 个合法任务名）
TASK_TO_PURCHASE_ROLE: dict[str, str] = {
    "生存/运转维系": "运转责任人",
    "缺陷修复/冲突消除": "问题承受者",
    "降本增效/懒人替代": "成本承担者",
    "物理安全与风险规避": "风险责任人",
    "情绪安心/主观降险": "焦虑承担者",
    "新奇探索/瞬时刺激": "体验追求者",
    "自我犒赏与秩序掌控": "自我奖励者",
    "照护与责任履行": "照护责任人",
    "礼赠与关系表达": "关系表达者",
    "圈层认同（圈层归属/身份锚定）": "圈层成员",
    "阶层与审美发信": "信号发出者",
}

# Step2：年龄/性别关键词（作用于 leaf_category+product_name+core_selling_point+target_people 拼接文本）
AUDIENCE_FEMALE_KEYWORDS: tuple[str, ...] = (
    "母婴", "婴", "宝宝", "儿童", "孕", "妈妈", "家庭", "驱蚊", "洗护", "护肤",
    "面膜", "美妆", "彩妆", "香氛", "女", "内衣", "卫生巾", "厨房", "家清", "辅食",
)
AUDIENCE_MALE_KEYWORDS: tuple[str, ...] = (
    "汽车", "车载", "机油", "工具", "五金", "户外", "钓鱼", "男士", "剃须",
    "电竞", "装备", "摩托",
    # 第五批：性别信号辅助扩展（仅作非送礼样本的八大人群派生辅助，不替代 gift_context 分支）。
    "男性", "男装",
)
AUDIENCE_MATURE_KEYWORDS: tuple[str, ...] = (
    "母婴", "宝宝", "儿童", "家庭", "辅食", "老人", "养生", "保健", "厨房",
    "家清", "驱蚊", "照护",
    # 第五批：年龄信号辅助扩展（中老年），同样仅作非送礼样本的辅助。
    "中年", "中老年",
)
AUDIENCE_YOUNG_KEYWORDS: tuple[str, ...] = (
    "潮玩", "盲盒", "零食", "球鞋", "电竞", "新奇", "学生", "ins", "网红", "二次元",
)
# 性别角色默认（都命中/都未命中时）：以下角色默认 female
AUDIENCE_FEMALE_DEFAULT_ROLES = {"照护责任人", "自我奖励者", "关系表达者"}
# 以下角色在无品类信号时默认 mixed（不强行归性别）
AUDIENCE_MIXED_GENDER_ROLES = {"风险责任人", "体验追求者"}
# 年龄角色默认（无年龄关键词时）
AUDIENCE_AGE_DEFAULT_ROLES = {"照护责任人": "mature", "体验追求者": "young"}

SAFETY_REASONING_REGEX = re.compile(r"预防|避险|防受伤|安全|防晒|防漏|防摔|防刮|防烫|隔离|保护")
HIGH_RISK_SINGLE_CHAR_TOKENS = {"味", "黄", "控", "祛", "污", "脏"}
DEFECT_TOKENS = {"黄", "脏", "痘", "塌", "秃", "卡", "裂", "斑", "味", "污", "去渍", "除味", "修护", "修复", "祛", "控"}
DEFECT_STATE_TOKENS = set(get_string_list("product_diagnoser.jtbd.defect_state_tokens"))
DEFECT_REMEDIATION_TOKENS = set(get_string_list("product_diagnoser.jtbd.defect_remediation_tokens"))
STRONG_DEFECT_TOKENS = DEFECT_STATE_TOKENS | DEFECT_REMEDIATION_TOKENS
PREFERENCE_ONLY_TOKENS = set(get_string_list("product_diagnoser.jtbd.preference_only_tokens"))
MAINTENANCE_SUPPLY_TOKENS = set(get_string_list("product_diagnoser.jtbd.maintenance_supply_tokens"))
FOOD_CATEGORY_TOKENS = set(get_string_list("product_diagnoser.jtbd.food_category_tokens"))
EFFICIENCY_TOKENS = set(get_string_list("product_diagnoser.jtbd.efficiency_tokens"))
OPERATION_EASE_TOKENS = set(get_string_list("product_diagnoser.jtbd.operation_ease_tokens"))
BLUE_OCEAN_TOKENS = set(get_string_list("product_diagnoser.jtbd.blue_ocean_tokens"))
FAST_MOVING_TOKENS = set(get_string_list("product_diagnoser.frequency.fast_moving_tokens"))
DURABLE_TOKENS = set(get_string_list("product_diagnoser.frequency.durable_tokens"))
ORDINARY_DAILY_TOKENS = set(get_string_list("product_diagnoser.jtbd.ordinary_daily_tokens"))
PHYSICAL_SAFETY_TOKENS = set(get_string_list("product_diagnoser.jtbd.physical_safety_tokens"))
PROTECTION_SEMANTIC_TOKENS = set(get_string_list("product_diagnoser.jtbd.protection_semantic_tokens"))
PROTECTION_RISK_CATEGORY_TOKENS = set(get_string_list("product_diagnoser.jtbd.protection_risk_category_tokens"))
CARE_TOKENS = {"宝宝", "婴儿", "儿童", "家人", "父母", "老人", "宠物"}
GIFT_TOKENS = {"礼盒", "礼物", "送礼", "伴手礼", "谢礼", "回礼", "节日礼物"}
EMOTIONAL_PREMIUM_TOKENS = set(get_string_list("product_diagnoser.jtbd.emotional_premium_tokens"))
FUNCTIONAL_BREAKOUT_TOKENS = set(get_string_list("product_diagnoser.jtbd.functional_breakout_tokens"))
# PRD-2.1：GR-02（天然≠安全）排除判定用词表，仅命中此集合而无真实伤害 token 时不得触发物理安全。
NATURAL_MILD_PREFERENCE_TOKENS = set(get_string_list("product_diagnoser.jtbd.natural_mild_preference_tokens"))
FRAGRANCE_SENSORY_VALUE_TOKENS = set(get_string_list("product_diagnoser.jtbd.fragrance_sensory_value_tokens"))
FRAGRANCE_CATEGORY_TOKENS = set(get_string_list("product_diagnoser.jtbd.fragrance_category_tokens"))
GIFT_RELATIONSHIP_SCENE_TOKENS = set(get_string_list("product_diagnoser.jtbd.gift_relationship_scene_tokens"))
HAIRCARE_DEFECT_SIGNAL_TOKENS = set(get_string_list("product_diagnoser.jtbd.haircare_defect_signal_tokens"))
# PRD-1.2：Stage B 空候选中止态。needs_review 为协议合法值，但不是 12 标签业务任务。
NEEDS_REVIEW = "needs_review"
# 商品理解应然侧的 content_goal 默认口径（P0 假设：转化目标）。集中此处，禁止散落写死；
# 后续如需支持非转化目标，由调用侧显式透传 content_goal，不在 diagnose() 内写死分支。
DEFAULT_PRODUCT_UNDERSTANDING_CONTENT_GOAL = "purchase"
FUNCTIONAL_CORE_TOKENS = {"升级", "更薄", "更厚", "更稳", "更持久", "更服帖", "更贴合", "更耐用", "更温和", "参数", "成分"}
EMOTIONAL_BREAKOUT_TOKENS = {"替代", "场景", "惊喜", "盲盒", "解闷", "尝鲜", "陪伴", "玩法", "跨界"}
EMOTIONAL_CORE_TOKENS = {"香氛", "助眠", "疗愈", "悦己", "犒赏", "松弛", "仪式感", "氛围感"}
GR03_STRONG_ANXIETY_MAIN_CHAIN_TOKENS = {
    "怕烂脸",
    "烂脸",
    "过敏焦虑",
    "刺激恐惧",
    "成分焦虑",
    "反复试错恐惧",
    "敏感肌恐惧",
    "不敢用",
    "担心毁脸",
    "红肿刺痛担忧",
}
SOCIAL_BREAKOUT_TOKENS = {"礼物", "送礼", "通行证", "身份", "圈层", "玩家", "社群", "替代", "跨界", "节日"}
SOCIAL_CORE_CARRIER_TOKENS = {"礼盒", "香水", "鲜花", "首饰", "箱包", "穿搭", "配饰", "玩具礼盒", "文具礼盒"}

# =============================================================================
# PRD-0：MVP 范围控制（scope gate）— 早停，不进入模块3/4
# =============================================================================
MVP_UNSUPPORTED_CATEGORY_TOKENS = {
    "香水",
    "香氛",
    "香薰",
    "香精",
    "淡香",
    "礼盒",
    "送礼",
    "伴手礼",
}
MVP_UNSUPPORTED_EMOTIONAL_TOKENS = {
    "自我犒赏",
    "犒赏",
    "悦己",
    "关系表达",
    "高级感",
    "氛围感",
    "仪式感",
}
MVP_UNSUPPORTED_JTBD = {
    "自我犒赏与秩序掌控",
    "礼赠与关系表达",
    "圈层认同（圈层归属/身份锚定）",
    "阶层与审美发信",
}

# === PRD §5.1.5 社会任务硬门槛证据词表 ===
# 关系对象 / 礼赠对象（必须命中明确实体，而非泛化营销词）
SOCIAL_RELATIONSHIP_OBJECT_TOKENS = {
    "长辈", "父母", "爸妈", "妈妈", "爸爸", "爷爷", "奶奶", "外婆", "外公", "姥姥", "姥爷",
    "孩子", "宝宝", "婴儿", "幼儿", "儿童", "孕妇",
    "妻子", "丈夫", "老公", "老婆", "伴侣", "男友", "女友", "对象",
    "朋友", "闺蜜", "同事", "客户", "领导", "老师",
    "宠物", "猫咪", "狗狗", "毛孩子", "家人", "家庭",
}
# 照护 / 责任履行动作
SOCIAL_CAREGIVING_VERB_TOKENS = {
    "照护", "护理", "看护", "照顾", "照料", "陪伴", "守护",
    "喂养", "哺乳", "辅食", "换尿布", "洗澡护理",
    "责任", "履行",
}
# 礼赠场景
SOCIAL_GIFT_SCENE_TOKENS = {
    "送礼", "礼赠", "伴手礼", "回礼", "礼盒", "节日礼物", "谢礼",
    "送父母", "送爸妈", "送长辈", "送朋友", "送同事", "送客户", "送女友", "送男友", "送老婆", "送老公",
    "中秋", "春节", "端午", "新年", "生日礼物", "节日送礼",
}
# 可追溯的稳定圈层（必须是具名群体，泛化潮流词不计）
SOCIAL_CIRCLE_IDENTITY_TOKENS = {
    "露营圈", "骑行圈", "钓鱼圈", "登山圈", "户外圈",
    "二次元", "lo娘", "lolita", "lo圈", "jk圈", "汉服圈", "汉服娘",
    "谷子", "吧唧", "原神", "崩坏", "盲盒圈", "潮玩圈",
    "母婴圈", "宠物圈", "猫奴", "狗奴",
    "球鞋圈", "潮鞋圈", "高尔夫圈", "瑜伽圈", "健身圈", "格斗圈",
    "职场通勤", "通勤穿搭", "商务精英", "极客圈", "电竞圈", "玩家身份",
}
# 阶层与审美发信 - 高外显性证据
SOCIAL_STATUS_VISIBILITY_TOKENS = {
    "穿搭", "佩戴", "随身", "出街", "见客户", "宴请", "商务场合", "出席",
    "腕表", "手表", "包包", "首饰", "项链", "戒指", "耳饰", "珠宝", "皮带",
    "外套", "西装", "礼服", "高跟鞋", "墨镜", "丝巾",
}
# 阶层与审美发信 - 获取门槛 / 稀缺性证据
SOCIAL_STATUS_BARRIER_TOKENS = {
    "限量", "稀缺", "限定", "孤品", "联名", "藏品", "收藏级",
    "奢侈", "高端定制", "高定", "顶级", "高净值",
    "手工", "匠造", "工艺大师", "原产地直供",
}
# 阶层与审美发信 - 圈层共识 / 审美共识证据
SOCIAL_STATUS_CONSENSUS_TOKENS = {
    "老钱风", "新中式高奢", "审美在线", "圈内认", "懂的人都懂", "行家",
    "经典款", "传家", "icon", "信仰",
}
# 排除：基础功能型品类典型标记
SOCIAL_BASIC_FUNCTIONAL_EXCLUDE_TOKENS = {
    "清洁", "去污", "去渍", "除味", "收纳", "保暖", "防晒", "补水", "保湿",
    "续航", "提效", "替换", "替芯", "维稳", "止汗", "防滑", "去屑", "驱蚊",
    "防蚊", "杀菌", "消毒",
}
# 排除：低外显 / 私密自用品
SOCIAL_LOW_VISIBILITY_EXCLUDE_TOKENS = {
    "卫生巾", "护垫", "私护", "情趣", "成人尿不湿", "纸尿裤",
    "卫生棉条", "内裤", "袜子", "保暖内衣",
}
# 仅营销修辞，不构成社会任务证据
SOCIAL_MARKETING_RHETORIC_ONLY_TOKENS = {
    "网红", "爆款", "同款", "高级感", "有面子", "潮流",
}
ENDORSEMENT_TOKENS = set(get_string_list("product_diagnoser.jtbd.endorsement_tokens"))

FACT_OBJECT_TOKENS = {
    "油污": "油污",
    "重油污": "油污",
    "污渍": "污渍",
    "顽固污渍": "污渍",
    "牙渍": "牙渍",
    "牙黄": "牙黄",
    "口气": "口气",
    "异味": "异味",
    "头皮屑": "头皮屑",
    "痘痘": "痘痘",
    "黑头": "黑头",
    "营养": "营养",
    "能量": "能量",
    "饱腹": "饱腹",
    "水分": "水分",
    "电解质": "电解质",
    "厨房": "厨房",
    "台面": "厨房表面",
    "烟机": "厨房表面",
    "步骤": "操作步骤",
    "流程": "操作流程",
    "刷洗": "刷洗流程",
    "清洗": "清洗流程",
    "操作": "操作流程",
    "时间": "时间成本",
}
FACT_DEFECT_OBJECTS = {"油污", "污渍", "牙渍", "牙黄", "口气", "异味", "头皮屑", "痘痘", "黑头"}
FACT_SUPPLY_OBJECTS = {"营养", "能量", "饱腹", "水分", "电解质"}
FACT_MAINTENANCE_OBJECTS = {"厨房", "厨房表面"}
FACT_PROCESS_OBJECTS = {"操作步骤", "操作流程", "刷洗流程", "清洗流程", "时间成本"}
FACT_ABNORMAL_STATE_TOKENS = {
    "顽固": "顽固附着",
    "厚重": "厚重残留",
    "发黄": "已发黄",
    "牙黄": "已发黄",
    "污": "脏污残留",
    "污渍": "污渍残留",
    "异味": "异味困扰",
    "口气": "口气困扰",
    "困扰": "已发生困扰",
    "堵": "堵塞",
    "疼": "疼痛不适",
    "不适": "疼痛不适",
}
FACT_MAINTENANCE_STATE_TOKENS = {
    "日常": "日常维持",
    "维持": "维持正常",
    "保持": "保持正常",
    "整洁": "维持整洁",
    "补给": "日常补给",
    "解馋": "日常补给",
    "充饥": "基础供给",
    "口腹": "基础供给",
}
FACT_PROCESS_STATE_TOKENS = {
    "费力": "费力麻烦",
    "麻烦": "费力麻烦",
    "繁琐": "流程繁琐",
    "复杂": "流程复杂",
    "省时": "耗时偏高",
    "省力": "费力麻烦",
    "省事": "费力麻烦",
    "一步": "步骤较多",
    "刷洗": "刷洗费力",
}
FACT_REMEDIATION_ACTION_TOKENS = {
    "去除": "去除",
    "清洁": "去除",
    "清除": "去除",
    "去油污": "去除",
    "去黄": "改善",
    "改善": "改善",
    "修复": "修复",
    "修护": "修复",
    "缓解": "缓解",
    "消除": "消除",
    "除味": "去除",
    "一擦": "去除",
}
FACT_MAINTENANCE_ACTION_TOKENS = {
    "补充": "补充",
    "维持": "维持",
    "保持": "维持",
    "支撑": "支撑",
    "供给": "补充",
    "解馋": "补充",
    "食用": "补充",
}
FACT_EFFICIENCY_ACTION_TOKENS = {
    "简化": "简化",
    "替代": "替代",
    "压缩": "压缩",
    "提速": "提速",
    "一步到位": "简化",
    "免洗": "替代",
    "日抛": "替代",
    "一喷一擦": "简化",
    "免刷洗": "简化",
    "省时": "提速",
    "省力": "简化",
    "省事": "简化",
    "省心": "简化",
}
HOUSEHOLD_COMMON_OBJECT_TOKENS = {
    "污渍": "污渍",
    "污垢": "污垢",
    "油污": "油污",
    "油垢": "油污",
    "异味": "异味",
    "残留": "残留",
}
HOUSEHOLD_COMMON_STATE_TOKENS = {
    "顽固": "顽固残留",
    "重油": "重度残留",
    "残留": "残留状态",
    "异味": "异味困扰",
    "脏": "脏污残留",
    "污": "脏污残留",
}
HOUSEHOLD_COMMON_ACTION_TOKENS = {
    "清洁": "清洁处理",
    "去污": "去污处理",
    "擦拭": "擦拭处理",
    "除味": "除味处理",
    "净味": "除味处理",
    "去除": "去除处理",
}
PERSONAL_CARE_COMMON_OBJECT_TOKENS = {
    "肌肤": "肌肤",
    "皮肤": "肌肤",
    "头皮": "头皮",
    "发丝": "头皮发丝",
    "发质": "头皮发丝",
    "口腔": "口腔",
    "牙面": "口腔",
    "牙齿": "口腔",
    "毛发": "毛发管理",
    "胡须": "毛发管理",
}
PERSONAL_CARE_COMMON_STATE_TOKENS = {
    "痘": "痘痘困扰",
    "痘痘": "痘痘困扰",
    "闭口": "闭口困扰",
    "黑头": "黑头困扰",
    "细纹": "纹路困扰",
    "颈纹": "纹路困扰",
    "暗沉": "暗沉困扰",
    "残留": "残留负担",
    "头屑": "头屑困扰",
    "出油": "出油困扰",
    "口气": "口气困扰",
    "牙黄": "牙渍发黄",
}
PERSONAL_CARE_COMMON_ACTION_TOKENS = {
    "修护": "修护",
    "舒缓": "修护",
    "祛痘": "修护",
    "抗皱": "修护",
    "淡纹": "修护",
    "卸净": "去除",
    "卸妆": "去除",
    "乳化": "去除",
    "清洁": "去除",
    "去屑": "去除",
    "控油": "去除",
    "美白": "去除",
    "去渍": "去除",
    "防晒": "防护",
    "防护": "防护",
    "维持": "维持",
    "保持": "维持",
    "清新": "维持",
    "快速": "简化",
    "轻松上手": "简化",
    "一推即净": "简化",
}
PAPER_DEFAULT_TASK = "生存/运转维系"
PAPER_ESCALATABLE_TASK = "缺陷修复/冲突消除"
PAPER_HARD_VETO_RULES: dict[str, dict[str, Any]] = {
    "paper_material_only": {
        "desc": "仅承载物/纸张基础属性，不得升级为修复任务。",
        "match_terms": {"柔软", "厚实", "亲肤", "吸水", "不易破", "可冲散", "温和", "舒适", "原生木浆", "便携"},
    },
    "paper_no_problem_object": {
        "desc": "未出现明确异常对象或承压表面，不得升级。",
        "match_terms": set(),
    },
    "paper_no_remediation_action": {
        "desc": "未出现明确修复/去除动作，不得升级。",
        "match_terms": {"擦除", "溶解", "去污", "除味", "净味", "去油", "清除"},
    },
    "paper_cleaner_substitution": {
        "desc": "仅凭清洁剂式语言伪装，不得把纸品升级为家清修复链路。",
        "match_terms": {"去污", "除味", "净味", "清洁方便", "强力清洁"},
    },
}
HOUSEHOLD_STAGEB_SUBCATEGORY_PACKS: dict[str, dict[str, Any]] = {
    "family_env_cleaning": {
        "category_terms": {"家庭环境清洁", "马桶清洁剂/洁厕剂", "多用途清洁剂", "洁厕剂", "清洁喷雾", "地面清洁剂", "家庭玻璃清洁剂", "果蔬专用清洁剂", "水垢清洁剂/除垢剂", "油污清洁剂", "洁厕凝胶", "洁瓷剂", "洗洁精", "管道疏通剂", "铁锈清洁剂", "锅底黑垢清洁剂"},
        "problem_object_terms": {"马桶": "马桶", "灶台": "灶台", "台面": "台面", "油烟机": "油烟机", "地板": "地面", "瓷砖": "瓷砖", "厨房": "厨房表面"},
        "problem_state_terms": {"黄垢": "黄垢残留", "水垢": "水垢残留", "尿渍": "尿渍残留", "油垢": "油垢残留", "污膜": "污膜残留", "污渍": "污渍残留", "油污": "油污残留", "重油污": "油污残留", "厚重油污": "油污残留"},
        "action_mechanism_terms": {"除垢": "去除", "去黄": "去除", "去污": "去除", "清洁": "去除", "溶解": "溶解", "擦除": "去除", "干净": "去除"},
        "fact_groups": [
            {
                "group_name": "defect_repair",
                "problem_object_terms": {"马桶": "马桶", "灶台": "灶台", "台面": "台面", "油烟机": "油烟机", "地板": "地面", "瓷砖": "瓷砖", "厨房": "厨房表面"},
                "problem_state_terms": {"黄垢": "黄垢残留", "水垢": "水垢残留", "尿渍": "尿渍残留", "油垢": "油垢残留", "污膜": "污膜残留", "污渍": "污渍残留", "油污": "油污残留", "重油污": "油污残留", "厚重油污": "油污残留"},
                "action_mechanism_terms": {"除垢": "去除", "去黄": "去除", "去污": "去除", "清洁": "去除", "溶解": "溶解", "擦除": "去除", "干净": "去除"},
                "default_object": "厨房表面",
            },
            {
                "group_name": "operation_ease",
                "problem_object_terms": {"刷洗": "刷洗流程", "清洗": "清洗流程", "操作": "操作流程"},
                "problem_state_terms": {"刷洗": "刷洗费力", "省事": "费力麻烦"},
                "action_mechanism_terms": {"一喷一擦": "简化", "免刷洗": "简化", "省事": "简化"},
                "default_object": "刷洗流程",
                "allow_action_only_without_object": True,
                "default_state_when_action_only": "费力麻烦",
                "group_candidate_task": "降本增效/懒人替代",
            },
        ],
        "negative_terms": {"居家可用", "多用途", "轻松擦拭", "日常可用"},
        "match_threshold": 3,
        "candidate_task": "缺陷修复/冲突消除",
    },
    "laundry_cleaning": {
        "category_terms": {"衣物清洁", "洗衣液", "洗衣凝珠", "洗衣粉", "洗衣皂", "衣物柔顺剂", "留香珠", "织物喷雾", "衣物除菌剂", "吸色片", "内衣皂/内衣洗涤剂", "即时去渍剂", "干洗剂", "彩漂", "洗衣啫喱", "洗衣片", "洗衣膏", "漂白剂", "爆炸盐", "衣物鞋类清洁泡泡/慕斯", "衣领净", "织物染色剂"},
        "problem_object_terms": {"衣物": "衣物", "纤维": "衣物", "衣领": "衣物", "袖口": "衣物", "面料": "衣物", "织物": "衣物", "内衣": "衣物", "鞋面": "衣物", "鞋边": "衣物"},
        "problem_state_terms": {"去渍": "污渍残留", "污渍": "污渍残留", "汗味": "异味困扰", "异味": "异味困扰", "霉味": "异味困扰", "发硬": "维持正常", "发旧": "维持正常", "褪色": "颜色异常", "掉色": "颜色异常", "串色": "颜色异常", "染色": "颜色异常"},
        "action_mechanism_terms": {"去异味": "去除", "去霉味": "去除", "除汗味": "去除", "去渍": "去除", "除味": "去除", "洗净": "去除", "洁净": "去除", "柔顺": "改善", "护理": "改善", "留香": "改善", "除菌": "改善", "护色": "改善", "防串色": "改善", "补色": "修复", "改色": "修复", "翻新": "修复", "固色": "修复", "还原": "修复", "染色": "修复"},
        "negative_terms": {"清香"},
        "match_threshold": 3,
        "candidate_task": "缺陷修复/冲突消除",
        "default_object": "衣物",
        "allow_action_only_with_object": True,
        "default_state_when_action_only": "维持正常",
        "action_only_terms": {"改善"},
    },
    "deodorization": {
        "category_terms": {"除臭用品", "空气清新/净化/芳香剂", "空气清新剂", "净化剂", "芳香剂", "鞋袜除臭剂/干爽剂", "冰箱除味剂", "其他除臭喷雾/剂/除臭用品", "除臭贴/除臭粘贴垫", "干燥剂", "活性炭", "甲醛清除剂", "除醛果冻/凝胶"},
        "problem_object_terms": {"鞋袜": "鞋袜", "鞋柜": "鞋柜", "冰箱": "冰箱", "柜体": "柜体", "空气": "空间", "空间": "空间", "房间": "空间", "室内": "空间", "异味源": "空间", "车内": "空间", "甲醛源": "甲醛源"},
        "problem_state_terms": {"异味": "异味困扰", "臭味": "异味困扰", "闷味": "异味困扰", "返味": "异味困扰", "霉味": "异味困扰", "潮味": "异味困扰", "甲醛": "异味困扰"},
        "action_mechanism_terms": {"净味": "去除", "除味": "去除", "除臭": "去除", "吸附": "改善", "净化": "改善", "干爽": "改善", "缓释": "改善", "清新": "改善", "恢复清新": "改善"},
        "negative_terms": {"芳香", "香味"},
        "match_threshold": 3,
        "candidate_task": "缺陷修复/冲突消除",
        "default_object": "空间",
    },
    "appliance_cleaning": {
        "category_terms": {"电器清洁", "洗衣机槽清洁剂", "洗衣机槽", "家电清洁", "空调清洁剂", "其他电器清洁用品", "洗碗机用洗涤剂", "洗衣机槽泡腾片"},
        "problem_object_terms": {"洗衣机槽": "洗衣机槽", "洗衣机内桶": "洗衣机槽", "洗衣机": "洗衣机槽", "空调": "空调", "滤网": "空调", "蒸发器": "空调", "洗碗机": "洗碗机", "油烟机": "油烟机", "电器": "电器"},
        "problem_state_terms": {"槽垢": "污垢残留", "污垢": "污垢残留", "残留": "残留污膜", "污膜": "污膜残留", "异味": "异味困扰", "霉味": "异味困扰", "油垢": "污垢残留"},
        "action_mechanism_terms": {"泡腾": "瓦解", "渗透": "瓦解", "瓦解": "瓦解", "槽洗": "去除", "清洗": "去除", "清洁": "去除", "除垢": "去除", "去味": "去除", "免拆": "简化", "日常维护": "维持", "维护": "维持", "长效": "维持"},
        "negative_terms": set(),
        "match_threshold": 3,
        "candidate_task": "缺陷修复/冲突消除",
        "allow_action_only_with_object": True,
        "default_state_when_action_only": "维持正常",
        "action_only_terms": {"维持", "简化"},
    },
    "paper_products": {
        "category_terms": {"纸品", "普通抽纸", "厨房纸巾", "湿厕纸", "清洁湿巾", "功能湿巾", "保湿纸巾", "卷纸", "平板卫生纸", "手帕纸", "擦手纸", "棉柔巾/洗脸巾", "生鲜专用吸水纸/食材擦拭纸", "静电除尘纸", "马桶垫纸"},
        "problem_object_terms": {"灶台": "灶台", "台面": "台面", "桌面": "桌面", "餐桌": "桌面", "油烟机": "油烟机", "屏幕": "屏幕", "镜片": "镜片", "马桶圈": "马桶圈"},
        "problem_state_terms": {"油垢": "油垢残留", "污膜": "污膜残留", "污渍": "污渍残留", "异味": "异味困扰", "黄垢": "黄垢残留"},
        "action_mechanism_terms": {"溶解": "去除", "擦除": "去除", "去污": "去除", "除味": "去除", "净味": "去除", "去油": "去除", "清除": "去除"},
        "negative_terms": {"柔软", "亲肤", "厚实", "吸水", "不易破", "可冲散", "便携"},
        "match_threshold": 3,
        "candidate_task": "缺陷修复/冲突消除",
    },
    "skincare_repair": {
        "category_terms": {"精华液", "次抛精华", "眼部精华", "颈霜", "面膜", "祛痘身体乳", "泥膜", "清洁面膜", "泥膜/清洁面膜", "面部精华"},
        "problem_object_terms": {"肌肤": "肌肤", "皮肤": "肌肤", "面部": "肌肤", "脸部": "肌肤", "颈部": "肌肤", "身体": "肌肤", "眼部": "肌肤", "眼周": "肌肤", "眼肌": "肌肤"},
        "problem_state_terms": {"痘痘": "痘痘困扰", "痘": "痘痘困扰", "闭口": "闭口困扰", "黑头": "黑头困扰", "细纹": "纹路困扰", "淡纹": "纹路困扰", "颈纹": "纹路困扰", "暗沉": "暗沉困扰"},
        "action_mechanism_terms": {"修护": "修护", "祛痘": "修护", "舒缓": "修护", "抗皱": "修护", "淡纹": "修护", "净颜": "去除", "维稳": "修护", "保湿": "维持"},
        "negative_terms": set(),
        "match_threshold": 3,
        "candidate_task": "缺陷修复/冲突消除",
        "default_object": "肌肤",
        "allow_action_only_with_object": True,
        "default_state_when_action_only": "维持正常",
        "action_only_terms": {"修护", "维持"},
    },
    "cleanse_protection": {
        "category_terms": {"洁面乳", "卸妆膏/卸妆油", "卸妆膏", "卸妆油", "防晒喷雾", "防晒霜/乳", "防晒霜", "防晒乳", "卸妆", "洁面"},
        "problem_object_terms": {"彩妆": "面部残留", "妆面": "面部残留", "油脂": "面部残留", "残留": "面部残留", "肌肤": "肌肤", "皮肤": "肌肤", "紫外线": "防晒风险"},
        "problem_state_terms": {"残留": "残留负担", "闷痘": "残留负担", "卸妆": "残留负担", "卸净": "残留负担", "卸除彩妆": "残留负担", "卸除防晒": "残留负担", "彩妆残留": "残留负担", "防晒残留": "残留负担", "晒伤": "晒伤风险", "晒黑": "晒黑风险", "紫外线": "紫外线风险"},
        "action_mechanism_terms": {"卸净": "去除", "卸妆": "去除", "乳化": "去除", "清洁": "去除", "防晒": "防护", "隔离": "防护", "防护": "防护"},
        "negative_terms": set(),
        "match_threshold": 3,
        "candidate_task": "缺陷修复/冲突消除",
        "default_object": "肌肤",
        "allow_action_only_with_object": True,
        "allow_action_only_without_object": True,
        "default_state_when_action_only": "维持正常",
        "action_only_terms": {"去除", "防护"},
    },
    "hair_scalp_care": {
        "category_terms": {"洗发水", "洗发皂", "发膜", "护发素/发膜"},
        "problem_object_terms": {"头皮": "头皮", "发丝": "头皮发丝", "发质": "头皮发丝", "头发": "头皮发丝"},
        "problem_state_terms": {"头屑": "头屑困扰", "出油": "出油困扰", "毛躁": "毛躁受损", "干枯": "毛躁受损", "断发": "受损断裂", "扁塌": "维持正常"},
        "action_mechanism_terms": {"去屑": "去除", "控油": "去除", "修护": "修护", "防脱": "修护", "滋养": "修护", "蓬松": "维持", "清洁": "维持", "洗净": "维持", "顺滑": "维持"},
        "negative_terms": set(),
        "match_threshold": 3,
        "candidate_task": "缺陷修复/冲突消除",
        "default_object": "头皮发丝",
        "allow_action_only_with_object": True,
        "default_state_when_action_only": "维持正常",
        "action_only_terms": {"维持"},
    },
    "oral_care": {
        "category_terms": {"牙膏", "牙线棒", "口喷", "口腔喷剂", "牙线/牙签/牙线棒"},
        "problem_object_terms": {"口腔": "口腔", "牙面": "口腔", "牙齿": "口腔", "牙缝": "口腔"},
        "problem_state_terms": {"口气": "口气困扰", "口臭": "口气困扰", "异味": "口气困扰", "牙黄": "牙渍发黄", "牙渍": "牙渍发黄", "残留": "残留负担"},
        "action_mechanism_terms": {"美白": "去除", "去渍": "去除", "去除": "去除", "清新": "维持", "清洁": "维持", "护理": "维持", "剔除": "去除", "剔牙": "去除"},
        "negative_terms": set(),
        "match_threshold": 3,
        "candidate_task": "缺陷修复/冲突消除",
        "default_object": "口腔",
        "allow_action_only_with_object": True,
        "allow_action_only_without_object": True,
        "default_state_when_action_only": "维持正常",
        "action_only_terms": {"维持", "去除"},
    },
    "hair_removal_tools": {
        "category_terms": {"剃须刀", "刮毛刀", "脱毛刀", "手动剃须刀", "脱毛工具"},
        "problem_object_terms": {"毛发": "毛发管理", "汗毛": "毛发管理", "胡须": "毛发管理", "胡茬": "毛发管理", "剃须": "毛发管理"},
        "problem_state_terms": {"新手": "流程负担", "费劲": "流程负担", "麻烦": "流程负担", "刮伤": "风险损伤", "刺痛": "风险损伤", "划伤": "风险损伤"},
        "action_mechanism_terms": {"快速": "简化", "轻松上手": "简化", "一推即净": "简化", "剃净": "简化", "防刮伤": "防护", "减刺激": "防护", "防护": "防护"},
        "negative_terms": set(),
        "match_threshold": 3,
        "candidate_task": "降本增效/懒人替代",
        "default_object": "毛发管理",
        "allow_action_only_with_object": True,
        "default_state_when_action_only": "流程负担",
        "action_only_terms": {"简化"},
    },
}
PERSONAL_CARE_STAGEB_SUBCATEGORY_CONTEXTS = {"skincare_repair", "cleanse_protection", "hair_scalp_care", "oral_care", "hair_removal_tools"}

DIFFERENTIATOR_COMPARISON_OBJECTS = {"同类旧方案", "同赛道竞品", "跨品类旧动作", "旧形态方案"}
DIFFERENTIATOR_COMPARISON_OBJECT_EVIDENCE_TYPES = {"user_provided", "text_extracted", "jtbd_inferred", "null"}
DIFFERENTIATOR_DOMAIN_TYPES: dict[str, set[str]] = {
    "functional": {"自身卖点陈述", "步骤压缩", "效果增强", "风险降低", "成本优化", "体验升级", "新形态替代"},
    "emotional": {"情绪安抚", "确定感提升", "自我犒赏", "仪式感增强", "新奇刺激", "感官满足", "疗愈放松", "氛围营造"},
    "social": {"圈层归属强化", "身份锚定", "身份跃迁", "审美表达", "品位发信", "礼赠体面", "关系表达"},
    "trust": {"信任缓释"},
}
DIFFERENTIATOR_RELATIVE_FUNCTIONAL_TYPES = {"步骤压缩", "效果增强", "风险降低", "成本优化", "体验升级", "新形态替代"}
DIFFERENTIATOR_RELATIVE_SEMANTIC_TOKENS = {
    "更", "更强", "更快", "更省", "更安全", "更舒服", "更舒适", "更高级", "升级", "替代", "比普通", "比传统", "相对", "相比",
}
DIFFERENTIATOR_ALLOWED_DOMAINS = set(DIFFERENTIATOR_DOMAIN_TYPES)
DIFFERENTIATOR_DIFFERENCE_TYPES = {
    difference_type
    for difference_types in DIFFERENTIATOR_DOMAIN_TYPES.values()
    for difference_type in difference_types
}
DIFFERENTIATOR_EVIDENCE_SOURCES = {
    "OCR",
    "ASR",
    "商品信息",
    "评论",
    "详情页",
    "人工标注",
    "JTBD推断",
    "caller_provided.core_selling_points",
}
LEGACY_DIFFERENTIATOR_FIELDS = {"diff_point", "compare_to", "compare_target", "compare_object", "selling_point_summary"}
LEGACY_DIFFERENCE_TYPES = {
    "流程步骤减少",
    "效果稳定性提升",
    "流程步骤减少/效果稳定性提升",
    "卖点升级",
    "超级好用",
}
MARKETING_TOKENS = {"王炸", "神器", "绝绝子", "YYDS", "神仙"}
DIFFERENTIATOR_TYPE_KEYWORDS: dict[str, set[str]] = {
    key: set(value)
    for key, value in get_mapping_of_string_lists("product_diagnoser.differentiator_type_keywords").items()
}
PEOPLE_SUMMARY_KEYWORDS: dict[str, tuple[str, ...]] = {
    key: tuple(value)
    for key, value in get_mapping_of_string_lists("product_diagnoser.people_summary").items()
}
COMPARISON_OBJECT_KEYWORDS: dict[str, tuple[str, ...]] = {
    key: tuple(value)
    for key, value in get_mapping_of_string_lists("product_diagnoser.comparison_object_keywords").items()
}
DIFFERENTIATOR_CONCLUSION_STRONG_CLAIMS: dict[str, tuple[tuple[str, ...], ...]] = {
    key: ((tuple(value),) if value else ())
    for key, value in get_mapping_of_string_lists(
        "product_diagnoser.differentiator_assertions.conclusion_strong_claim_tokens"
    ).items()
}
DIFFERENTIATOR_OLD_SCHEME_REQUIRED_TOKENS = set(
    get_string_list("product_diagnoser.differentiator_assertions.old_scheme_required_tokens")
)
DIFFERENTIATOR_EFFECT_ENHANCEMENT_TOKENS = set(
    get_string_list("product_diagnoser.differentiator_assertions.effect_enhancement_tokens")
)
DIFFERENTIATOR_CONVENIENCE_ONLY_TOKENS = set(
    get_string_list("product_diagnoser.differentiator_assertions.convenience_only_tokens")
)
DIFFERENTIATOR_RELATIVE_DIFFERENCE_TYPE_ANCHORS = {
    key: set(value)
    for key, value in get_mapping_of_string_lists(
        "product_diagnoser.differentiator_assertions.relative_difference_type_anchors"
    ).items()
}
DIFFERENTIATOR_JUDGE_RULES: tuple[dict[str, str], ...] = (
    {
        "id": "R_diff_evidence_conclusion_support",
        "desc": "差异化卖点的 evidence_chain 必须独立支撑 difference_type 与 conclusion；禁止把 conclusion 文本本身当作证据做反向放行。若结论包含时长、量化或强功效主张，证据中必须出现对应事实锚点。",
    },
    {
        "id": "R_diff_specific_old_scheme_anchor_required",
        "desc": "当 comparison_object=同类旧方案 且 comparison_object_evidence_type 不属于 jtbd_inferred / user_provided 时，evidence_chain 必须包含可定位的具体旧方案锚点，用于回答相对什么旧方案/旧动作/旧流程更优。仅有当前商品自身卖点、抽象升级词或便利型描述，不构成旧方案锚点。difference_type=自身卖点陈述 时跳过该断言。",
    },
    {
        "id": "R_diff_convenience_not_effect_enhancement",
        "desc": "当 difference_type=效果增强 时，证据必须直接支撑效果结果变强，而不是只支撑更方便使用。轻薄、易穿脱、易披挂、便携、顺手、省步骤、操作便利等便利型锚点，不能单独推出效果增强。",
    },
    {
        "id": "R_diff_self_statement_relative_residue_block",
        "desc": "当 difference_type=自身卖点陈述 时，conclusion / summary 中不得残留未被证据支撑的相对语义；若出现更强、更快、更省、更安全、更舒服、替代传统、比普通更好等比较语义，但 evidence_chain 无对应比较锚点，必须拒绝。",
    },
)

DURABLE_BUSINESS_CATEGORIES = set(get_string_list("product_diagnoser.frequency.durable_business_categories"))

CATEGORY_INTENT_COPY: dict[tuple[str, str | None, str], str] = {
    ("蓝海", None, "快消"): "先建立任务意识与新品类合理性，再逐步解释为什么值得马上尝试。",
    ("蓝海", None, "耐消"): "先建立需求与新解法正当性，再降低用户对长期决策的理解门槛。",
    ("红海", "破圈", "快消"): "优先剥离旧 SOP 与使用惯性，证明新方案更省事、更值得立刻切换。",
    ("红海", "核心", "快消"): "优先完成存量同类替换，证明在熟悉用法下换它更值。",
    ("红海", "破圈", "耐消"): "优先放大旧方案的长期损失，制造不换就继续吃亏的替换动机。",
    ("红海", "核心", "耐消"): "优先降低选型风险，通过参数、适配和避坑信息提供确定性。",
}

PRODUCT_INTENT_COPY: dict[tuple[str, str], str] = {
    ("大牌官方", "低"): "官方信任资产最强，可直接承接低价试单与顺手成交。",
    ("大牌官方", "中"): "官方信任资产充足，重点补足常规价位下的场景价值与购买理由。",
    ("大牌官方", "高"): "官方信任资产极强，重点转向高价位的价值证明、价格锚定与算账。",
    ("大牌经销", "低"): "虽命中大牌资产，但非旗舰店存在信任折损，需先补供应链/防伪安心感，再承接低价成交。",
    ("大牌经销", "中"): "经销店信任资产打折，需先解释货源与真伪，再说明常规价位下为什么值得选。",
    ("大牌经销", "高"): "经销店在高价位下同时面临真伪与溢价质疑，必须先建信，再做价值证明。",
    ("白牌", "低"): "先补基础可信度，再承接低门槛试单与顺手成交。",
    ("白牌", "中"): "先补足可信度，再说明常规价位下为什么值得选它而不是默认选项。",
    ("白牌", "高"): "信任与价格压力同时偏高，必须先建信，再做价值证明，最后才能收单。",
}

RULE_TABLE: dict[str, dict[str, str]] = {
    "category_intent_copy": {str(key): value for key, value in CATEGORY_INTENT_COPY.items()},
    "product_intent_copy": {str(key): value for key, value in PRODUCT_INTENT_COPY.items()},
}

JTBDClassifier = Callable[["DiagnosticInput"], Mapping[str, Any] | "JTBDProposal"]


@lru_cache(maxsize=1)
def _load_price_band_dict() -> tuple[dict[str, float], ...]:
    price_band_path = ROOT / "memory/topics/price_band_dict.csv"
    if not price_band_path.exists():
        raise FileNotFoundError(f"价格带字典缺失: {price_band_path}")

    category_columns = ("叶子类目名称", "业务自定义类目")
    median_columns = ("按订单量加权价格中位数", "价格中位数", "中位数价格", "价格带中位数")
    legacy_low_column = "低价格带阈值(元)"
    legacy_high_columns = ("高价格带阈值", "高价格带阈值(元)")

    with price_band_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = tuple(reader.fieldnames or [])
        category_column = next((column for column in category_columns if column in fieldnames), None)
        median_column = next((column for column in median_columns if column in fieldnames), None)
        legacy_high_column = next((column for column in legacy_high_columns if column in fieldnames), None)
        if category_column is None:
            raise ValueError("price_band_dict.csv 缺少叶子类目列，无法执行价格带断言。")
        if median_column is None and (legacy_low_column not in fieldnames or legacy_high_column is None):
            raise ValueError(
                "price_band_dict.csv 列定义不完整，必须包含价格中位数列；过渡期仅兼容旧版低/高阈值列。"
            )

        rows: list[dict[str, float]] = []
        for row in reader:
            category = str(row.get(category_column, "")).strip()
            if not category:
                continue
            median_raw = str(row.get(median_column, "")).strip() if median_column else ""
            if median_raw:
                try:
                    median_threshold = float(median_raw)
                except ValueError as exc:
                    raise ValueError(f"price_band_dict.csv 存在非法价格中位数: {category}") from exc
            else:
                low_raw = str(row.get(legacy_low_column, "")).strip()
                high_raw = str(row.get(legacy_high_column, "")).strip() if legacy_high_column else ""
                if not low_raw or not high_raw:
                    raise ValueError(f"price_band_dict.csv 缺少价格中位数，且旧版低/高阈值也不完整: {category}")
                try:
                    low_threshold = float(low_raw)
                    high_threshold = float(high_raw)
                except ValueError as exc:
                    raise ValueError(f"price_band_dict.csv 存在非法旧版价格阈值: {category}") from exc
                if low_threshold <= 0 or high_threshold <= 0 or low_threshold > high_threshold:
                    raise ValueError(f"price_band_dict.csv 旧版价格阈值非法: {category}")
                median_threshold = (low_threshold + high_threshold) / 2
            if median_threshold <= 0:
                raise ValueError(f"price_band_dict.csv 价格中位数必须大于 0: {category}")
            rows.append({category: median_threshold})

    if not rows:
        raise ValueError("price_band_dict.csv 为空，无法执行价格带断言。")
    return tuple(rows)


@lru_cache(maxsize=1)
def _build_price_band_lookup() -> dict[str, float]:
    merged: dict[str, float] = {}
    for row in _load_price_band_dict():
        merged.update(row)
    return merged


@lru_cache(maxsize=1)
def _load_brand_whitelist() -> frozenset[str]:
    whitelist_path = ROOT / "memory/topics/brand_whitelist.csv"
    if not whitelist_path.exists():
        raise FileNotFoundError(f"品牌白名单缺失: {whitelist_path}")

    names: set[str] = set()
    with whitelist_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = set(reader.fieldnames or [])
        if "店铺名称" not in fieldnames:
            raise ValueError("brand_whitelist.csv 缺少 店铺名称 列，无法执行白名单断言。")
        for row in reader:
            shop_name = str(row.get("店铺名称", "")).strip()
            if shop_name:
                names.add(shop_name)
    if not names:
        raise ValueError("品牌白名单为空，无法执行白名单断言。")
    return frozenset(names)


@lru_cache(maxsize=1)
def _load_store_suffix_trust_dict() -> tuple[tuple[str, str, str], ...]:
    dict_path = ROOT / "memory/topics/store_suffix_trust_dict.csv"
    if not dict_path.exists():
        raise FileNotFoundError(f"店铺后缀信任字典缺失: {dict_path}")

    rows: list[tuple[str, str, str]] = []
    with dict_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = set(reader.fieldnames or [])
        required_fields = {"店铺后缀", "品牌资产判定", "信任阻力水位"}
        if not required_fields.issubset(fieldnames):
            raise ValueError("store_suffix_trust_dict.csv 列定义不完整，无法执行信任资产字典判定。")
        for row in reader:
            suffix = str(row.get("店铺后缀", "")).strip()
            brand_tier = str(row.get("品牌资产判定", "")).strip()
            trust_barrier = str(row.get("信任阻力水位", "")).strip()
            if suffix and brand_tier and trust_barrier:
                rows.append((suffix, brand_tier, trust_barrier))
    if not rows:
        raise ValueError("store_suffix_trust_dict.csv 为空，无法执行信任资产字典判定。")
    return tuple(rows)


@dataclass(slots=True)
class DiagnosticInput:
    # ===== 商品侧字段（SSOT / 前端输入） =====
    leaf_category: str
    shop_name: str
    second_level_category: str = ""
    third_level_category: str = ""

    # scope gate 允许读取的商品字段之一（若上游可提供）。
    category_path: str = ""

    brand_name: str = ""
    product_name: str = ""
    price: str = ""

    # scope gate 允许读取的商品字段之一（兼容 core_selling_points 的聚合结果）。
    core_selling_point: str = ""

    # scope gate 允许读取的商品字段之一（若上游可提供）。
    product_detail_summary: str = ""

    field_provenance: dict[str, str] | None = None
    core_selling_point_source: str = ""
    target_people: str = ""

    # ===== 桥接层/推理字段（不应参与 scope gate） =====
    differentiator: Any = ""
    bridge_comparison_object: str = ""
    bridge_comparison_object_evidence_type: str = "null"
    bridge_difference_domain: str = ""
    bridge_difference_type: str = ""
    bridge_source_evidence: list[str] | None = None
    bridge_evidence_source: str = "商品信息"
    product_id: str = ""
    sample_tags: dict[str, str] | None = None
    engine_node: dict[str, Any] | None = None

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "DiagnosticInput":
        def pick(*keys: str) -> str:
            for key in keys:
                if key in payload and payload[key] is not None:
                    return str(payload[key]).strip()
            return ""

        def pick_raw(*keys: str) -> Any:
            for key in keys:
                if key in payload:
                    return payload[key]
            return ""

        raw_tags = payload.get("sample_tags")
        sample_tags = raw_tags if isinstance(raw_tags, dict) else None
        raw_engine_node = payload.get("engine_node")
        engine_node = raw_engine_node if isinstance(raw_engine_node, dict) else None
        raw_field_provenance = payload.get("field_provenance")
        if raw_field_provenance is None:
            basic_product_info = payload.get("basic_product_info")
            if isinstance(basic_product_info, Mapping):
                raw_field_provenance = basic_product_info.get("field_provenance")
        if isinstance(raw_field_provenance, Mapping):
            field_provenance = {
                str(key).strip(): str(value).strip()
                for key, value in raw_field_provenance.items()
                if str(key).strip() and str(value).strip()
            } or None
        else:
            field_provenance = None
        raw_bridge_source_evidence = pick_raw("bridge_source_evidence")
        if isinstance(raw_bridge_source_evidence, str):
            bridge_source_evidence = [raw_bridge_source_evidence.strip()] if raw_bridge_source_evidence.strip() else None
        elif isinstance(raw_bridge_source_evidence, list):
            bridge_source_evidence = [str(item).strip() for item in raw_bridge_source_evidence if str(item).strip()] or None
        else:
            bridge_source_evidence = None
        raw_differentiator = pick_raw("differentiator", "差异化卖点")
        if isinstance(raw_differentiator, str):
            differentiator: Any = raw_differentiator.strip()
        else:
            differentiator = raw_differentiator
        return cls(
            leaf_category=pick(
                "leaf_category",
                "category",
                "类目",
                "叶子类目",
                "third_level_category",
                "third_level_catgeory",
                "三级类目",
                "second_level_category",
                "二级类目",
            ),
            shop_name=pick("shop_name", "shop", "店铺", "店铺名称"),
            second_level_category=pick("second_level_category", "二级类目"),
            third_level_category=pick("third_level_category", "third_level_catgeory", "三级类目"),
            category_path=pick("category_path", "categoryPath", "类目路径"),
            brand_name=pick("brand_name", "brand", "品牌", "品牌名称", "品牌名"),
            product_name=pick("product_name", "商品名", "商品名称"),
            price=pick("price", "价格", "售价"),
            core_selling_point=pick("core_selling_point", "核心卖点"),
            product_detail_summary=pick(
                "product_detail_summary",
                "product_detail",
                "product_detail_info",
                "reviewed_problem_summary",
                "content_summary",
                "商品详情摘要",
            ),
            field_provenance=field_provenance,
            core_selling_point_source=pick("core_selling_point_source"),
            target_people=pick("target_people", "目标人群"),
            differentiator=differentiator,
            bridge_comparison_object=pick("bridge_comparison_object"),
            bridge_comparison_object_evidence_type=pick("bridge_comparison_object_evidence_type") or "null",
            bridge_difference_domain=pick("bridge_difference_domain"),
            bridge_difference_type=pick("bridge_difference_type"),
            bridge_source_evidence=bridge_source_evidence,
            bridge_evidence_source=pick("bridge_evidence_source") or "商品信息",
            product_id=pick("product_id", "商品ID", "sku_id"),
            sample_tags=sample_tags,
            engine_node=engine_node,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def category(self) -> str:
        return self.leaf_category

    def joined_text(self) -> str:
        if isinstance(self.differentiator, str):
            differentiator_text = self.differentiator
        elif isinstance(self.differentiator, StructuredDifferentiator):
            evidence_text = " ".join(item.evidence_text for item in self.differentiator.evidence_chain)
            differentiator_text = " ".join(
                part for part in [
                    self.differentiator.summary or self.differentiator.conclusion,
                    self.differentiator.comparison_object,
                    self.differentiator.difference_type,
                    evidence_text,
                ] if part
            )
        else:
            differentiator_text = json.dumps(self.differentiator, ensure_ascii=False)
        return "｜".join(
            part for part in [
                self.leaf_category,
                self.brand_name,
                self.product_name,
                self.core_selling_point,
                self.target_people,
                differentiator_text,
            ] if part
        )


@dataclass(slots=True)
class DifferentiatorEvidence:
    evidence_source: str
    evidence_text: str


@dataclass(slots=True)
class StructuredDifferentiator:
    comparison_object: str
    comparison_object_evidence_type: str = "null"
    difference_domain: str = ""
    difference_type: str = ""
    conclusion: str = ""
    evidence_chain: list[DifferentiatorEvidence] = field(default_factory=list)
    summary: str = ""


@dataclass(slots=True)
class Module1Output:
    leaf_category: str
    shop_name: str
    product_name: str
    price: str
    core_selling_point: str
    core_selling_point_source: str
    target_people: str
    differentiator: StructuredDifferentiator
    second_level_category: str = ""
    third_level_category: str = ""
    # 接入层（第五批）：保留模块 1 目标人群的**原始未截断信号**，供 gift_context
    # 送礼场景识别使用。target_people 仍是归一化后的「XX人群」标签（不破坏既有
    # JTBD/people_summary 逻辑）；target_people_raw 不剥离节日/送礼/对象/关系词。
    target_people_raw: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class JTBDProposal(BaseModel):
    domain: Literal["功能域", "情绪域", "社会域"]
    primary_task: str
    sub_task: str | None = None
    reasoning: str
    reasoning_path: list[str] = Field(default_factory=list)
    candidate_tasks: list[str] = Field(default_factory=list)
    candidate_reasons: dict[str, list[str]] = Field(default_factory=dict)
    excluded_tasks: dict[str, list[str]] = Field(default_factory=dict)
    triggered_rule: str = ""
    gate_reasons: list[str] = Field(default_factory=list)
    trace_tokens: list[str] = Field(default_factory=list)
    evidence_chain: list[dict[str, str]] = Field(default_factory=list)
    functional_facts: list[dict[str, Any]] = Field(default_factory=list)
    candidate_pool: list[dict[str, Any]] = Field(default_factory=list)
    subcategory_context: str = ""
    veto_trace: list[str] = Field(default_factory=list)
    # PRD-1.1：辅助字段。secondary_benefits 元素必须是 VALID_JTBD 的 12 标签枚举值；
    # non_selected_task_reasons 每元素含 task/reason/gate，gate ∈ {blocked, dropped, demoted_to_secondary}。
    secondary_benefits: list[str] = Field(default_factory=list)
    non_selected_task_reasons: list[dict[str, str]] = Field(default_factory=list)

    @validator("primary_task")
    def validate_primary_task(cls, value: str) -> str:
        if value == NEEDS_REVIEW:  # PRD-1.2：needs_review 为协议合法值，短路放行
            return value
        if value not in VALID_JTBD:
            raise ValueError(f"非法一级任务: {value}")
        return value

    @root_validator
    def validate_domain_and_sub_task(cls, values: dict[str, Any]) -> dict[str, Any]:
        primary_task = values.get("primary_task", "")
        if primary_task == NEEDS_REVIEW:  # PRD-1.2：needs_review 短路放行，不校验 domain 一致/sub_task
            return values
        domain = values.get("domain", "")
        expected_domain = TASK_DOMAIN_MAP.get(primary_task)
        if expected_domain and domain != expected_domain:
            raise ValueError(f"任务 {primary_task} 与域 {domain} 不一致，应为 {expected_domain}")
        sub_task = values.get("sub_task")
        if primary_task == "阶层与审美发信":
            if sub_task not in ALLOWED_SUB_TASKS:
                raise ValueError("阶层与审美发信 必须且只能选择一个合法 sub_task。")
        elif sub_task:
            raise ValueError(f"任务 {primary_task} 不允许携带 sub_task。")
        return values



# =============================================================================
# 说服要求建模模块（Persuasion Requirement Modeling）协议族 —— V3.1 一期
# =============================================================================
DecisionGap = Literal[
    "need_gap",
    "fit_gap",
    "value_gap",
    "proof_gap",
    "trust_gap",
    "risk_gap",
    "action_gap",
]
ContentGoal = Literal[
    "conversion",
    "purchase",
    "add_to_cart",
    "coupon_claim",
    "shop_entry",
    "seeding",
    "education",
    "brand_awareness",
    "unknown",
]
Priority = Literal["high", "medium", "low"]
JTBDTemplateStatus = Literal["matched", "fallback_generic"]
RequirementCompletionStatus = Literal["completed", "partial", "missing", "not_applicable"]
ACTION_GOALS: frozenset[str] = frozenset({"conversion", "purchase", "add_to_cart", "coupon_claim", "shop_entry"})
ACTIVE_REQUIREMENT_WHITELIST: tuple[str, ...] = (
    "expose_current_pain",
    "clarify_usage_scenario",
    "identify_target_user",
    "prove_user_fit",
    "prove_scenario_fit",
    "prove_spec_fit",
    "prove_core_benefit",
    "prove_new_solution_efficiency",
    "establish_clear_difference",
    "prove_replacement_value",
    "prove_price_reasonableness",
    "provide_visible_result",
    "prove_effect_not_degraded",
    "prove_quality_stability",
    "establish_basic_trust",
    "prove_source_credibility",
    "provide_authority_endorsement",
    "reduce_trial_risk",
    "resolve_quality_risk",
    "resolve_safety_risk",
    "resolve_value_risk",
    "prove_current_purchase_reason",
    "clarify_purchase_threshold",
)
REQUIREMENT_STATUS_ENUM: tuple[str, ...] = ("completed", "partial", "missing", "not_applicable")
DIAGNOSIS_DIMENSIONS: tuple[str, ...] = (
    "whether_requirement_appears",
    "whether_evidence_is_sufficient",
    "whether_sequence_is_reasonable",
    "whether_risk_is_resolved",
)


class StrictBaseModel(BaseModel):
    class Config:
        extra = "forbid"
        anystr_strip_whitespace = True
        validate_assignment = True


class PersuasionRequirement(StrictBaseModel):
    requirement_id: str
    requirement_name: str
    decision_gap: DecisionGap
    source: list[str] = Field(default_factory=list)
    priority: Priority
    required: bool
    sequence_rank: int = Field(ge=10, le=59)
    success_criteria: str
    related_decision_criteria: list[str] = Field(default_factory=list)
    required_evidence_requirements: list[str] = Field(default_factory=list)
    risk_points: list[str] = Field(default_factory=list)

    @validator("requirement_id")
    def _requirement_in_whitelist(cls, value: str) -> str:
        if value not in ACTIVE_REQUIREMENT_WHITELIST:
            raise ValueError(f"requirement_id={value} 不在 23 条 active MVP 白名单内。")
        return value

    @validator("source")
    def _source_not_empty(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("persuasion_requirement.source 不允许为空。")
        return value


class NotApplicableRequirement(StrictBaseModel):
    requirement_id: str
    decision_gap: DecisionGap
    status: Literal["not_applicable"] = "not_applicable"
    reason: str

    @validator("requirement_id")
    def _requirement_in_whitelist(cls, value: str) -> str:
        if value not in ACTIVE_REQUIREMENT_WHITELIST:
            raise ValueError(f"not_applicable requirement_id={value} 不在 23 条 active 白名单内。")
        return value


class PrimaryJTBD(StrictBaseModel):
    level1: str
    level2: str


class CategoryResistance(StrictBaseModel):
    rule: str
    summary: str


class ProductConversionBarrier(StrictBaseModel):
    rule: str


class MainPersuasionRoute(StrictBaseModel):
    primary_jtbd: PrimaryJTBD
    category_resistance: CategoryResistance
    product_conversion_barrier: ProductConversionBarrier


class ActivatedCategoryRequirements(StrictBaseModel):
    category_group: str
    routing_confidence: str = ""
    activated_decision_criteria: list[str] = Field(default_factory=list)
    activated_evidence_requirements: list[str] = Field(default_factory=list)
    activated_risk_points: list[str] = Field(default_factory=list)


class RequirementCompletionSchema(StrictBaseModel):
    status_enum: list[RequirementCompletionStatus] = Field(default_factory=lambda: list(REQUIREMENT_STATUS_ENUM))
    minimum_required_requirements: list[str] = Field(default_factory=list)
    diagnosis_dimensions: list[str] = Field(default_factory=lambda: list(DIAGNOSIS_DIMENSIONS))

    @validator("status_enum")
    def _status_enum_fixed(cls, value: list[str]) -> list[str]:
        if list(value) != list(REQUIREMENT_STATUS_ENUM):
            raise ValueError(f"status_enum 必须固定为 {list(REQUIREMENT_STATUS_ENUM)}。")
        return value

    @validator("minimum_required_requirements")
    def _minimum_in_whitelist(cls, value: list[str]) -> list[str]:
        illegal = [rid for rid in value if rid not in ACTIVE_REQUIREMENT_WHITELIST]
        if illegal:
            raise ValueError(f"minimum_required_requirements 含非白名单项 {illegal}。")
        return value

    @validator("diagnosis_dimensions")
    def _dimensions_fixed(cls, value: list[str]) -> list[str]:
        if list(value) != list(DIAGNOSIS_DIMENSIONS):
            raise ValueError(f"diagnosis_dimensions 必须固定为 {list(DIAGNOSIS_DIMENSIONS)}。")
        return value


class DiagnosisContract(StrictBaseModel):
    requirement_completion_schema: RequirementCompletionSchema


class PersuasionRequirementProfile(StrictBaseModel):
    profile_version: str = "v3.1"
    content_goal: ContentGoal
    category_group: str
    jtbd_template_status: JTBDTemplateStatus
    requirement_dictionary_version: str
    category_purchase_criteria_version: str = ""
    main_persuasion_route: MainPersuasionRoute
    activated_category_requirements: ActivatedCategoryRequirements
    persuasion_requirements: list[PersuasionRequirement] = Field(default_factory=list)
    not_applicable_requirements: list[NotApplicableRequirement] = Field(default_factory=list)
    diagnosis_contract: DiagnosisContract

    @root_validator
    def _action_gap_governance(cls, values: dict[str, Any]) -> dict[str, Any]:
        content_goal = values.get("content_goal")
        requirements = values.get("persuasion_requirements") or []
        if content_goal not in ACTION_GOALS:
            leaked = [
                r.requirement_id
                for r in requirements
                if getattr(r, "decision_gap", None) == "action_gap"
            ]
            if leaked:
                raise ValueError(
                    f"content_goal={content_goal} 非转化目标，action_gap 要求 {leaked} "
                    "不得进入 persuasion_requirements，必须输出 not_applicable。"
                )
        return values


class CategoryIntentMatrix(BaseModel):
    ocean: Literal["蓝海", "红海"]
    competition_focus: Literal["核心", "破圈"] | None = None
    frequency: Literal["快消", "耐消"]
    domain_route_rule: str
    matrix_label: str
    category_intent: str
    competition_focus_reason: str = ""
    competition_focus_evidence_chain: list[dict[str, str]] = Field(default_factory=list)
    difference_type_route_result: str = ""
    reasoning: list[str] = Field(default_factory=list)


class ProductIntentMatrix(BaseModel):
    brand_tier: Literal["大牌官方", "大牌经销", "白牌"]
    trust_barrier: Literal["极低", "中", "高"]
    financial_risk: Literal["高", "中", "低"]
    relative_price_level: Literal["高水位", "低水位"]
    matrix_label: str
    business_category: str
    median_price_threshold: float
    price_value: float
    product_intent: str
    reasoning: list[str] = Field(default_factory=list)


class AudienceGroupJudgment(BaseModel):
    audience_group: str   # 八大人群枚举
    fit_level: str        # primary / secondary / weak_fit
    reason: str


class AudienceReasoningChain(BaseModel):
    task_to_role: str
    role_category_to_age_gender: str
    brand_price_to_consumption_power: str
    # F3：第四段必填——必须引用 profile 中真实 required/high requirement（requirement_id/name），
    # 不允许自由文本泛化或写死占位。
    persuasion_profile_to_audience: str


class ProductTargetAudience(BaseModel):
    primary_audiences: list[AudienceGroupJudgment]
    secondary_audiences: list[AudienceGroupJudgment] = Field(default_factory=list)
    weak_fit_audiences: list[AudienceGroupJudgment] = Field(default_factory=list)
    reasoning_chain: AudienceReasoningChain
    caveats: list[str] = Field(default_factory=list)


class ProductDiagnosisOutput(BaseModel):
    product_id: str = ""
    leaf_category: str
    shop_name: str
    product_name: str
    price: float
    domain: Literal["功能域", "情绪域", "社会域"]
    primary_task: str
    sub_task: str | None = None
    category_intent: str
    product_intent: str
    category_intent_matrix: CategoryIntentMatrix
    product_intent_matrix: ProductIntentMatrix
    reasoning_path: list[str]
    warnings: list[str] = Field(default_factory=list)
    persuasion_requirement_profile: Optional[PersuasionRequirementProfile] = None
    product_target_audience: Optional[ProductTargetAudience] = None

    category: str
    jtbd: str
    resistance_profile: dict[str, Any]
    core_intent: dict[str, Any]
    candidate_set: dict[str, Any] = Field(default_factory=dict)
    product_ec_skeletons: list[dict[str, Any]] = Field(default_factory=list)
    product_hecs: list[dict[str, Any]] = Field(default_factory=list)
    assembly_status: dict[str, Any] | None = None
    assertions: list[str] = Field(default_factory=list)
    evidence: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    schema_version: str = "v2.2"
    # PRD-1.1：商品事实向量辅助字段。jtbd（jtbd_level1）仍为唯一主任务字段，不新增 primary_task 协议字段。
    secondary_benefits: list[str] = Field(default_factory=list)
    non_selected_task_reasons: list[dict[str, str]] = Field(default_factory=list)

    @validator("primary_task")
    def validate_primary_task(cls, value: str) -> str:
        if value == NEEDS_REVIEW:  # PRD-1.2：needs_review 为协议合法值，短路放行
            return value
        if value not in VALID_JTBD:
            raise ValueError(f"非法一级任务: {value}")
        return value

    @root_validator
    def validate_output(cls, values: dict[str, Any]) -> dict[str, Any]:
        primary_task = values.get("primary_task", "")
        if primary_task == NEEDS_REVIEW:  # PRD-1.2：needs_review 短路放行，不要求 domain 一致/sub_task 规则
            return values
        domain = values.get("domain", "")
        expected_domain = TASK_DOMAIN_MAP.get(primary_task)
        if expected_domain and domain != expected_domain:
            raise ValueError(f"输出 domain 与 primary_task 不一致: {domain} vs {primary_task}")
        joined_reasoning = " ".join(values.get("reasoning_path") or [])
        if domain == SOCIAL_DOMAIN:
            # PRD §5.1.5：禁止把社会任务专有词作为模板写入 reasoning，必须绑定真实证据
            if not (values.get("evidence_chain") or []):
                raise ValueError("社会域输出缺少 evidence_chain（必须绑定商品属性证据）。")
            if not (values.get("gate_reasons") or []):
                raise ValueError("社会域输出缺少 gate_reasons（必须含真实命中的证据描述）。")
            banned_template_fragments = (
                "符合圈层共识，因此进入社会任务",
                "命中社会域门槛",
                "规则树直接判定为社会任务",
            )
            if any(frag in joined_reasoning for frag in banned_template_fragments):
                raise ValueError("社会域 reasoning 命中预设模板文本，违反 PRD §5.1.5。")
        sub_task = values.get("sub_task")
        if primary_task == "阶层与审美发信":
            if sub_task not in ALLOWED_SUB_TASKS:
                raise ValueError("阶层与审美发信 输出必须带合法 sub_task。")
        elif sub_task:
            raise ValueError(f"任务 {primary_task} 不允许输出 sub_task。")
        return values

    def to_dict(self) -> dict[str, Any]:
        return self.dict(exclude_none=True)

    def to_protocol_dict(self) -> dict[str, Any]:
        return {
            "product_id": self.product_id,
            "product_name": self.product_name,
            "category": self.category,
            # PRD-1.1 AC2：jtbd（即 jtbd_level1）为唯一主任务字段，协议层不输出 primary_task 键。
            "jtbd": self.jtbd,
            "secondary_benefits": list(self.secondary_benefits),
            "non_selected_task_reasons": [dict(item) for item in self.non_selected_task_reasons],
            "resistance_profile": self.resistance_profile,
            "core_intent": self.core_intent,
            "candidate_set": self.candidate_set,
            "product_ec_skeletons": self.product_ec_skeletons,
            "product_hecs": self.product_hecs,
            "assembly_status": self.assembly_status,
            "assertions": self.assertions,
            "evidence": self.evidence,
            "metadata": self.metadata,
            "schema_version": self.schema_version,
        }


class JTBDLLMClassifier:
    """模块二 2.A：受限 JTBD 分类器。"""

    def __init__(
        self,
        *,
        model: str = "doubao-1.5-pro-32k-250115",
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: int = 60,
        llm_tag: str = "product_diagnoser_module2a",
    ) -> None:
        self.model = model
        self.llm_config = resolve_llm_config(base_url=base_url, api_key=api_key, model=model, timeout=timeout)
        self.model = self.llm_config.model
        self.base_url = self.llm_config.endpoint
        self.api_key = self.llm_config.api_key
        self.timeout = self.llm_config.timeout
        self.llm_tag = llm_tag

    def classify(self, payload: DiagnosticInput, candidate_tasks: list[str] | None = None) -> JTBDProposal:
        messages = self._build_messages(payload, candidate_tasks=candidate_tasks)
        raw = self._call_llm(messages)
        parsed = self._parse_json(raw)
        return JTBDProposal(**parsed)

    def _build_messages(self, payload: DiagnosticInput, candidate_tasks: list[str] | None = None) -> list[dict[str, str]]:
        allowed_tasks = candidate_tasks or [
            "生存/运转维系",
            "缺陷修复/冲突消除",
            "降本增效/懒人替代",
            "物理安全与风险规避",
            "情绪安心/主观降险",
            "新奇探索/瞬时刺激",
            "自我犒赏与秩序掌控",
            "照护与责任履行",
            "礼赠与关系表达",
            "圈层认同（圈层归属/身份锚定）",
            "阶层与审美发信",
        ]
        allowed_lines = "\n".join(f"{index}. {task}" for index, task in enumerate(allowed_tasks, start=1))
        system = (
            "你是模块二 2.A 的 JTBD 分类器。\n"
            "你只负责 Stage C 候选池内归并：只能在给定 candidate_tasks 内选择唯一 primary_task，且 domain 必须严格属于 功能域 / 情绪域 / 社会域 三选一。\n"
            "严禁自造标签，严禁输出营销话术，严禁跳过 reasoning。\n"
            "你只能从候选池内选择，不得越权，不得改写规则树已经给出的候选边界，更不得跳出 candidate_tasks。\n"
            "必须明确说明为什么选择当前任务，以及为什么排除其它候选；尤其要显式比较：生存/运转维系 vs 缺陷修复/冲突消除、缺陷修复/冲突消除 vs 降本增效/懒人替代。\n"
            "如果 primary_task = 阶层与审美发信，必须且只能输出 sub_task=身份跃迁 或 审美/阶层标榜。\n"
            "如果不是 阶层与审美发信，sub_task 必须为 null。\n"
            "reasoning_path 必须是数组，明确写出命中的锚点、候选比较结论或被排除的门槛。\n"
            "若输入 differentiator 为结构化字段，必须把 functional.difference_type 中的 自身卖点陈述 视为合法枚举值；它表示商品自身功能/属性/工艺/品质/结果事实成立，不等价于相对型比较。\n"
            f"本轮允许任务池：\n{allowed_lines}\n"
            "输出必须是严格 JSON，格式："
            "{\"domain\":\"...\",\"primary_task\":\"...\",\"sub_task\":null,\"reasoning\":\"...\",\"reasoning_path\":[\"...\"]}"
        )
        user_payload = {
            "product_id": payload.product_id,
            "leaf_category": payload.leaf_category,
            "shop_name": payload.shop_name,
            "brand_name": payload.brand_name,
            "product_name": payload.product_name,
            "price": payload.price,
            "core_selling_point": payload.core_selling_point,
            "target_people": payload.target_people,
            "differentiator": self._stringify_differentiator(payload.differentiator),
        }
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False, indent=2)},
        ]

    def _call_llm(self, messages: list[dict[str, str]]) -> str:
        require_llm_config(self.llm_config, purpose="模块二 JTBD 分类模型")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "X-LLM-TAG": self.llm_tag,
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            data=json.dumps(payload, ensure_ascii=False),
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def _parse_json(self, text: str) -> dict[str, Any]:
        cleaned = str(text).strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].strip()
        return json.loads(cleaned)

    @staticmethod
    def _stringify_differentiator(differentiator: Any) -> str:
        if isinstance(differentiator, Mapping):
            return json.dumps(differentiator, ensure_ascii=False)
        return str(differentiator or "").strip()


class DifferentiatorConclusionLLM:
    """模块一差异化结论生成器：严格依据当前商品 evidence 生成 conclusion。"""

    def __init__(
        self,
        *,
        model: str = "doubao-1.5-pro-32k-250115",
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: int = 60,
        llm_tag: str = "product_diagnoser_module1_differentiator",
    ) -> None:
        self.model = model
        self.llm_config = resolve_llm_config(base_url=base_url, api_key=api_key, model=model, timeout=timeout)
        self.model = self.llm_config.model
        self.base_url = self.llm_config.endpoint
        self.api_key = self.llm_config.api_key
        self.timeout = self.llm_config.timeout
        self.llm_tag = llm_tag

    def generate(
        self,
        *,
        comparison_object: str,
        difference_type: str,
        evidence_text: str,
        payload: DiagnosticInput,
    ) -> str:
        messages = self._build_messages(
            comparison_object=comparison_object,
            difference_type=difference_type,
            evidence_text=evidence_text,
            payload=payload,
        )
        raw = self._call_llm(messages)
        parsed = self._parse_json(raw)
        conclusion = str(parsed.get("conclusion") or "").strip()
        if not conclusion:
            raise ValueError("模块 1 差异化结论生成失败：LLM 未返回 conclusion。")
        return conclusion

    def _build_messages(
        self,
        *,
        comparison_object: str,
        difference_type: str,
        evidence_text: str,
        payload: DiagnosticInput,
    ) -> list[dict[str, str]]:
        system = (
            "你是模块 1 的差异化卖点 conclusion 生成器。\n"
            "你的唯一任务：根据当前商品 evidence 生成一句客观 conclusion。\n"
            "硬约束：\n"
            "1. conclusion 必须只基于当前 evidence_text 推导，禁止补充 evidence 中不存在的新事实；\n"
            "2. 禁止输出任何品类专属硬编码模板；\n"
            "3. 禁止出现营销词；\n"
            "4. 若 evidence_text 无法支撑 conclusion，必须返回原子化、可验证的事实表达；\n"
            "5. 当 difference_type=自身卖点陈述 时，禁止自动添加‘相对旧方案/竞品/旧动作/旧形态’等相对前缀；若 evidence_text 仅支撑商品自身卖点，就只输出自身事实。\n"
            "6. 输出必须是严格 JSON，格式：{\"conclusion\":\"...\"}。"
        )
        user_payload = {
            "leaf_category": payload.leaf_category,
            "product_name": payload.product_name,
            "core_selling_point": payload.core_selling_point,
            "comparison_object": comparison_object,
            "difference_type": difference_type,
            "evidence_text": evidence_text,
        }
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False, indent=2)},
        ]

    def _call_llm(self, messages: list[dict[str, str]]) -> str:
        require_llm_config(self.llm_config, purpose="模块一差异化结论模型")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "X-LLM-TAG": self.llm_tag,
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            data=json.dumps(payload, ensure_ascii=False),
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def _parse_json(self, text: str) -> dict[str, Any]:
        cleaned = str(text).strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].strip()
        return json.loads(cleaned)


class DifferentiatorSemanticJudgeLLM:
    def __init__(
        self,
        *,
        model: str = "doubao-1.5-pro-32k-250115",
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: int = 60,
        llm_tag: str = "product_diagnoser_module1_judge",
    ) -> None:
        self.model = model
        self.llm_config = resolve_llm_config(base_url=base_url, api_key=api_key, model=model, timeout=timeout)
        self.model = self.llm_config.model
        self.base_url = self.llm_config.endpoint
        self.api_key = self.llm_config.api_key
        self.timeout = self.llm_config.timeout
        self.llm_tag = llm_tag

    def judge(
        self,
        *,
        comparison_object: str = "",
        difference_domain: str,
        difference_type: str,
        conclusion: str,
        evidence_text: str,
        payload: DiagnosticInput,
    ) -> dict[str, Any]:
        require_llm_config(self.llm_config, purpose="模块一差异化语义 Judge")
        messages = self._build_messages(
            comparison_object=comparison_object,
            difference_domain=difference_domain,
            difference_type=difference_type,
            conclusion=conclusion,
            evidence_text=evidence_text,
            payload=payload,
        )
        raw = self._call_llm(messages)
        parsed = self._parse_json(raw)
        required_fields = ("supports_difference_type", "supports_conclusion", "reason", "judge_mode")
        missing_fields = [field for field in required_fields if field not in parsed]
        if missing_fields:
            raise ValueError(f"差异化卖点语义 Judge 返回格式非法：缺少约定字段 {missing_fields}。")
        supports_difference_type = parsed.get("supports_difference_type")
        supports_conclusion = parsed.get("supports_conclusion")
        reason = str(parsed.get("reason") or "").strip()
        judge_mode = str(parsed.get("judge_mode") or "").strip()
        if not isinstance(supports_difference_type, bool) or not isinstance(supports_conclusion, bool):
            raise ValueError("差异化卖点语义 Judge 返回格式非法：supports_difference_type / supports_conclusion 必须为布尔值。")
        if not reason:
            raise ValueError("差异化卖点语义 Judge 返回格式非法：reason 不允许为空。")
        if judge_mode != "llm_judge":
            raise ValueError(f"差异化卖点语义 Judge 返回格式非法：judge_mode 必须为 llm_judge，实际为 {judge_mode or '空值'}。")
        return {
            "supports_difference_type": supports_difference_type,
            "supports_conclusion": supports_conclusion,
            "reason": reason,
            "judge_mode": judge_mode,
        }

    def _build_messages(
        self,
        *,
        comparison_object: str = "",
        difference_domain: str,
        difference_type: str,
        conclusion: str,
        evidence_text: str,
        payload: DiagnosticInput,
    ) -> list[dict[str, str]]:
        system = (
            "你是模块 1 的 LLM-as-a-Judge 语义质检节点。\n"
            "你的唯一任务：独立判断 evidence_text 是否足以语义支撑 comparison_object、difference_type 和 conclusion。\n"
            "硬约束：\n"
            "1. 只能基于 evidence_text 做判断，禁止引用 conclusion 自身文本做反向放行；\n"
            "2. 禁止使用品类固定词表或经验模板，必须做语义判断；\n"
            "3. 若 comparison_object_evidence_type 不属于 jtbd_inferred / user_provided，且 evidence 只说明当前商品自身事实，但无法支持所声明的 comparison_object 成立方式，则 supports_difference_type=false；\n"
            "4. 若 evidence 只说明商品存在，但没有说明对应差异方向，则 supports_difference_type=false；\n"
            "5. 若 conclusion 包含时长、量化指标、强功效、确定性结果，而 evidence 中无对应事实锚点，则 supports_conclusion=false；\n"
            "6. 不要因为品类 unfamiliar 就保守拒绝，医疗护理、个护、美妆、家清等都按同一语义标准判断；\n"
            "7. 若 comparison_object=同类旧方案 且 comparison_object_evidence_type 不属于 jtbd_inferred / user_provided，evidence_text 中必须存在具体旧方案锚点；若只有当前商品自身卖点、抽象升级词或便利型描述，而没有明确指向被替代的旧方案对象，则 supports_difference_type=false；\n"
            "8. 若 comparison_object_evidence_type=jtbd_inferred，说明 comparison_object 是基于商品事实做的受限推断；此时不要再额外要求 evidence_text 逐字出现旧方案锚点，只需独立判断 evidence_text 是否足以支撑 difference_type 与 conclusion，且不要把跨品类旧动作/同赛道竞品当作可自由推断对象。\n"
            "9. 若 comparison_object_evidence_type=user_provided，说明 comparison_object 来自业务侧人工确认输入；此时不要因为 evidence_text 未逐字出现 comparison_object 本身就判否，但仍需独立判断 evidence_text 是否足以支撑 difference_type 与 conclusion。\n"
            "10. 若 difference_type=效果增强，evidence_text 中必须存在效果结果层锚点；若证据仅体现轻薄、便携、易穿脱、易披挂、省步骤、顺手、操作便利等便利型变化，而没有效果提升、持续增强、性能增强、结果增强的事实依据，则 supports_difference_type=false；\n"
            "11. 当 difference_type=自身卖点陈述 时，不得因为 comparison_object 为空就判否，也不得要求 evidence_text 出现旧方案、竞品、旧动作、旧形态或比较前缀；只需判断 evidence_text 是否足以支撑商品自身功能、属性、工艺、品质或结果卖点。\n"
            "12. 当 difference_type=自身卖点陈述 时，若 conclusion 中残留‘更强’‘更快’‘更省’‘更安全’‘更舒服’‘替代传统’‘比普通更好’等相对语义，但 evidence_text 无法独立支撑该比较关系，则 supports_difference_type=false 或 supports_conclusion=false。\n"
            "13. 若 conclusion 只是使用标准比较前缀（如‘相对旧方案’‘相对同赛道竞品’）来承载差异表达，除非 conclusion 额外新增证据中不存在的量化/结果/因果主张，否则不要因为 evidence_text 未逐字出现‘旧方案/竞品’而判否。\n"
            "14. 对于 comparison_object=旧形态方案 且 difference_type=新形态替代，只要 evidence_text 同句出现喷雾、喷雾剂、贴片、液体敷料、棒状、滚珠、日抛、免洗等任一形态锚点，并同时说明其对应的功能/对象，就必须判定 supports_difference_type=true；严禁再要求 evidence_text 额外出现‘替代旧方案’或‘旧形态方案’字样。若 evidence_text 是商品标题式表达，例如‘冷冻除疣喷雾剂用于低温去除寻常疣/传染性软疣’，其中‘喷雾剂’已提供形态锚点，‘用于低温去除寻常疣/传染性软疣’已提供功能/对象，必须视为满足该规则，禁止误判为“只提到商品本身”。\n"
            "15. 对于 comparison_object=旧形态方案 且 difference_type=新形态替代，只要 conclusion 只是把 evidence 中已经出现的形态锚点标准化转述为‘承接旧需求 / 替代旧形态 / 改写旧 SOP’，且未新增量化、时长、强功效或确定性结果主张，就必须判定 supports_conclusion=true。\n"
            '16. 输出必须是严格 JSON：{"supports_difference_type":true/false,"supports_conclusion":true/false,"reason":"...","judge_mode":"llm_judge"}。'
        )
        user_payload = {
            "leaf_category": payload.leaf_category,
            "product_name": payload.product_name,
            "core_selling_point": payload.core_selling_point,
            "comparison_object": comparison_object,
            "comparison_object_evidence_type": str(payload.bridge_comparison_object_evidence_type or "null").strip() or "null",
            "difference_domain": difference_domain,
            "difference_type": difference_type,
            "conclusion": conclusion,
            "evidence_text": evidence_text,
            "judge_rules": DIFFERENTIATOR_JUDGE_RULES,
        }
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False, indent=2)},
        ]

    def _call_llm(self, messages: list[dict[str, str]]) -> str:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "X-LLM-TAG": self.llm_tag,
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            data=json.dumps(payload, ensure_ascii=False),
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def _parse_json(self, text: str) -> dict[str, Any]:
        cleaned = str(text).strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].strip()
        return json.loads(cleaned)


class ProductDiagnosisEngine:
    """商品诊断主引擎：模块 1 加工 + 模块 2 诊断 + 模块 3 寻址 + 模块 4 装配。"""

    def __init__(
        self,
        classifier: JTBDClassifier | None = None,
        *,
        llm_classifier: JTBDLLMClassifier | None = None,
        differentiator_conclusion_llm: DifferentiatorConclusionLLM | None = None,
        differentiator_semantic_judge_llm: DifferentiatorSemanticJudgeLLM | None = None,
        max_retries: int = 3,
    ) -> None:
        self.classifier = classifier
        self.llm_classifier = llm_classifier or JTBDLLMClassifier()
        self.differentiator_conclusion_llm = differentiator_conclusion_llm or DifferentiatorConclusionLLM()
        self.differentiator_semantic_judge_llm = differentiator_semantic_judge_llm or DifferentiatorSemanticJudgeLLM()
        self.max_retries = max(1, max_retries)
        self.variant_assembler = ProductVariantAssembler()
        # F5：说服要求引擎在 __init__ 实例化一次，避免每次 diagnose 重复加载字典。
        # PersuasionRequirementEngine 构造时已对 4 份字典做启动期体检（Crash Early），
        # 字典缺失/越界会在此处直接抛错，禁止 try/except 吞掉。
        self.persuasion_engine = PersuasionRequirementEngine()
        self._keyword_rule_traces: list[dict[str, str]] = []

    def _reset_keyword_rule_traces(self) -> None:
        self._keyword_rule_traces = []

    def _record_keyword_rule_trace(self, field_name: str, output_value: str, rule_path: str, matched_keyword: str, source_rule: str | None = None) -> None:
        trace = assert_rule_trace(build_rule_trace(rule_path, matched_keyword, source_rule), rule_path)
        self._keyword_rule_traces.append(
            {
                "field": field_name,
                "output_value": output_value,
                "rule_path": trace.rule_path,
                "matched_keyword": trace.matched_keyword,
                "source_rule": trace.source_rule,
                "source_evidence": trace.source_evidence,
            }
        )

    def _find_first_keyword(self, text: str, keywords: tuple[str, ...] | set[str]) -> str | None:
        for keyword in keywords:
            if keyword in text:
                return keyword
        return None

    def diagnose(self, payload: DiagnosticInput | Mapping[str, Any]) -> ProductDiagnosisOutput:
        self._reset_keyword_rule_traces()
        if not isinstance(payload, DiagnosticInput):
            self._assert_no_legacy_differentiator_protocol(payload)
            payload = DiagnosticInput.from_payload(payload)
        self._assert_input(payload)
        module1_output = self._run_module1(payload)

        llm_proposal, warnings = self._resolve_jtbd(payload, module1_output)
        gated_proposal, gate_notes, gate_warnings = self._apply_hard_gates(payload, module1_output, llm_proposal)
        warnings.extend(gate_warnings)

        category_matrix = self._derive_category_intent_matrix(module1_output, gated_proposal)
        product_matrix = self._derive_product_intent_matrix(payload, module1_output)

        # F5：先生成 persuasion_requirement_profile，再生成 product_target_audience（audience 强依赖 profile）。
        # content_goal 取集中常量 DEFAULT_PRODUCT_UNDERSTANDING_CONTENT_GOAL（P0 默认 "purchase"=转化目标），
        # 不在此处散落写死字符串；后续如需支持非转化目标，由调用侧显式透传 content_goal。
        product_fact = self._build_persuasion_product_fact(
            payload, module1_output, gated_proposal, category_matrix, product_matrix
        )
        persuasion_requirement_profile = self.persuasion_engine.generate_profile(
            product_fact, content_goal=DEFAULT_PRODUCT_UNDERSTANDING_CONTENT_GOAL
        )
        # Crash Early：引擎输出的 profile 必须非空且 persuasion_requirements 非空，禁止兜底生成。
        if not persuasion_requirement_profile or not persuasion_requirement_profile.get(
            "persuasion_requirements"
        ):
            raise ValueError(
                "persuasion_requirement_profile 为空或 persuasion_requirements 为空 —— "
                "商品诊断必须由引擎产出非空说服要求档案，禁止兜底生成（Crash Early）。"
            )

        product_target_audience = self._derive_product_target_audience(
            module1_output,
            gated_proposal,
            product_matrix,
            persuasion_requirement_profile=persuasion_requirement_profile,
            product_fact=product_fact,
        )
        reasoning_path = self._compose_reasoning_path(llm_proposal, gated_proposal, gate_notes, category_matrix, product_matrix)
        assertions = self._build_assertions()
        diagnosis = self._build_output(
            payload=payload,
            module1_output=module1_output,
            proposal=gated_proposal,
            raw_proposal=llm_proposal,
            category_matrix=category_matrix,
            product_matrix=product_matrix,
            reasoning_path=reasoning_path,
            warnings=warnings,
            assertions=assertions,
            gate_notes=gate_notes,
            product_target_audience=product_target_audience,
            persuasion_requirement_profile=persuasion_requirement_profile,
        )
        assert_product_diagnosis(diagnosis.to_protocol_dict())
        return diagnosis

    def _assert_input(self, payload: DiagnosticInput) -> None:
        required = {
            "leaf_category": payload.leaf_category,
            "shop_name": payload.shop_name,
            "product_name": payload.product_name,
            "price": payload.price,
            "core_selling_point": payload.core_selling_point,
            "core_selling_point_source": payload.core_selling_point_source,
        }
        for field_name, value in required.items():
            if not str(value).strip():
                raise ValueError(f"模块一输入缺少必填字段：{field_name}")
        if payload.core_selling_point_source not in TITLE_CORE_SELLING_POINT_ALLOWED_SOURCES:
            raise ValueError(f"模块一输入 core_selling_point_source 非法：{payload.core_selling_point_source}")
        if payload.field_provenance is not None:
            invalid_provenance = {
                field_name: source
                for field_name, source in payload.field_provenance.items()
                if source not in FIELD_PROVENANCE_ALLOWED_VALUES
            }
            if invalid_provenance:
                raise ValueError(f"field_provenance 含非法来源枚举：{invalid_provenance}")
            if payload.field_provenance.get("core_selling_points") == BLOCKED_CORE_SELLING_POINT_PROVENANCE:
                raise ValueError(
                    "field_provenance.core_selling_points=video_extracted_candidate，core_selling_points 不得来自视频 ASR/OCR/VLM/video_factpack，按协议必须 Crash Early。"
                )
        if self._safe_parse_price(payload.price) <= 0:
            raise ValueError("模块一输入 price 无法解析为大于 0 的数值。")

    def _assert_no_legacy_differentiator_protocol(self, payload: Mapping[str, Any]) -> None:
        differentiator = payload.get("differentiator")
        for key in LEGACY_DIFFERENTIATOR_FIELDS:
            if key in payload:
                raise ValueError(f"差异化卖点输入命中旧字段：{key}")
        if isinstance(differentiator, Mapping):
            for key in LEGACY_DIFFERENTIATOR_FIELDS:
                if key in differentiator:
                    raise ValueError(f"差异化卖点输入命中旧字段：{key}")
            diff_type = differentiator.get("difference_type")
            if isinstance(diff_type, str) and diff_type.strip() in LEGACY_DIFFERENCE_TYPES:
                raise ValueError(f"差异化卖点输入命中旧枚举：{diff_type.strip()}")

    def _run_module1(self, payload: DiagnosticInput) -> Module1Output:
        target_people = self._normalize_target_people(payload)
        differentiator = self._normalize_differentiator(payload)
        # 接入层（第五批）：保留原始未截断的目标人群信号，不剥离节日/送礼/对象/关系词。
        target_people_raw = str(payload.target_people or "").strip()
        module1_output = Module1Output(
            leaf_category=payload.leaf_category.strip(),
            shop_name=payload.shop_name.strip(),
            second_level_category=payload.second_level_category.strip(),
            third_level_category=payload.third_level_category.strip(),
            product_name=payload.product_name.strip(),
            price=payload.price.strip(),
            core_selling_point=payload.core_selling_point.strip(),
            core_selling_point_source=payload.core_selling_point_source.strip(),
            target_people=target_people,
            differentiator=differentiator,
            target_people_raw=target_people_raw,
        )
        self._assert_module1_output(module1_output, payload)
        # 接入层 raw signal trace：若原始人群线索/卖点/证据命中送礼信号，记录来源，便于追溯。
        self._record_gift_signal_trace(payload, module1_output)
        return module1_output

    def _record_gift_signal_trace(
        self, payload: DiagnosticInput, module1_output: Module1Output
    ) -> None:
        """接入层 gift signal raw trace：识别送礼信号来源并登记到 keyword_rule_traces。

        仅做可追溯记录，不改变任何派生结果；非送礼样本无 trace。
        """
        segments = [
            module1_output.product_name,
            module1_output.core_selling_point,
            module1_output.target_people_raw,
            module1_output.leaf_category,
            *(str(x) for x in (payload.bridge_source_evidence or [])),
        ]
        gift_context = detect_gift_context(segments)
        if not gift_context:
            return
        evidence = gift_context.get("evidence") or []
        self._record_keyword_rule_trace(
            field_name="gift_context",
            output_value=(
                f"is_gift=true;scene={gift_context.get('gift_scene')};"
                f"recipient={gift_context.get('gift_recipient')};"
                f"decider={gift_context.get('purchase_decider')}"
            ),
            rule_path="product_diagnoser.gift_context",
            matched_keyword="、".join(evidence) if evidence else "gift_signal",
            source_rule="gift_context_raw_signal_trace",
        )

    def _normalize_target_people(self, payload: DiagnosticInput) -> str:
        raw = str(payload.target_people or "").strip()
        selling = str(payload.core_selling_point or "").strip()
        diff = self._stringify_differentiator(payload.differentiator)
        text = " ".join(part for part in [raw, selling, diff, payload.product_name, payload.leaf_category] if part)

        normalized = raw or self._summarize_people_from_text(text)
        normalized = re.sub(r"下班后|深夜|晚上|通勤路上|约会前|洗澡后", "", normalized).strip(" ，,、；;")
        for splitter in ("/", "、", "，", ",", "和", "及"):
            if splitter in normalized:
                normalized = normalized.split(splitter)[0].strip()
                break
        if normalized and not normalized.endswith("人群"):
            normalized = f"{normalized}人群"
        return normalized

    def _summarize_people_from_text(self, text: str) -> str:
        for people_label, keywords in PEOPLE_SUMMARY_KEYWORDS.items():
            matched_keyword = self._find_first_keyword(text, keywords)
            if matched_keyword:
                self._record_keyword_rule_trace(
                    field_name="target_people_summary",
                    output_value=people_label,
                    rule_path=f"product_diagnoser.people_summary.{people_label}",
                    matched_keyword=matched_keyword,
                )
                return people_label
        return "具体需求人群"

    def _inject_protection_risk_implicit_comparison(
        self,
        payload: DiagnosticInput,
        bridge_source_evidence: list[str],
        bridge_comparison_object: str,
    ) -> tuple[str, str, str, str] | None:
        """P2：防护/防虫/防晒/安全等风险类目的隐式对比对象注入。

        当商品类目命中防护风险类目，且证据中存在风险/防护锚点，而调用方未显式提供
        comparison_object 时，注入隐式旧方案基线（语义为“未使用任何驱避/防护产品”），
        把 difference_type 从“自身卖点陈述”改写为“风险降低”，避免被降级为纯自身卖点。
        返回 (difference_domain, difference_type, comparison_object, comparison_object_evidence_type)；
        不满足注入条件时返回 None，保持原“自身卖点陈述清空 comparison_object”协议。
        """
        # 调用方已显式提供 comparison_object 时不注入，尊重上游输入。
        if bridge_comparison_object:
            return None
        category_text = f"{payload.leaf_category} {payload.product_name}"
        category_keyword = self._find_first_keyword(category_text, PROTECTION_RISK_CATEGORY_TOKENS)
        if not category_keyword:
            return None
        joined_evidence = " ".join(str(item).strip() for item in bridge_source_evidence if str(item).strip())
        risk_anchor_tokens = DIFFERENTIATOR_RELATIVE_DIFFERENCE_TYPE_ANCHORS.get("风险降低", set())
        # 证据缺少风险/防护锚点时不注入，避免下游相对锚点断言 Crash Early。
        if not self._contains_any(joined_evidence, risk_anchor_tokens):
            return None
        self._record_keyword_rule_trace(
            field_name="protection_risk_implicit_comparison",
            output_value="同类旧方案",
            rule_path="product_diagnoser.jtbd.protection_risk_category_tokens",
            matched_keyword=category_keyword,
        )
        return ("functional", "风险降低", "同类旧方案", "jtbd_inferred")

    def _normalize_differentiator(self, payload: DiagnosticInput) -> StructuredDifferentiator:
        raw = payload.differentiator
        if isinstance(raw, Mapping):
            return self._validate_structured_differentiator(raw)

        raw_text = str(raw or "").strip()
        selling = str(payload.core_selling_point or "").strip()
        bridge_comparison_object = str(payload.bridge_comparison_object or "").strip()
        bridge_comparison_object_evidence_type = str(payload.bridge_comparison_object_evidence_type or "null").strip() or "null"
        bridge_difference_domain = str(payload.bridge_difference_domain or "").strip()
        bridge_difference_type = str(payload.bridge_difference_type or "").strip()
        bridge_source_evidence = [str(item).strip() for item in (payload.bridge_source_evidence or []) if str(item).strip()]
        has_bridge_contract = any(
            [
                bridge_comparison_object,
                bridge_difference_domain,
                bridge_difference_type,
                bool(bridge_source_evidence),
            ]
        )
        if any(token in raw_text for token in MARKETING_TOKENS):
            raise ValueError("模块 1 差异化卖点断言失败：存在营销词。")
        if bridge_comparison_object and bridge_comparison_object not in DIFFERENTIATOR_COMPARISON_OBJECTS:
            raise ValueError(f"差异化卖点 comparison_object 非法：{bridge_comparison_object}")
        if bridge_comparison_object_evidence_type not in DIFFERENTIATOR_COMPARISON_OBJECT_EVIDENCE_TYPES:
            raise ValueError(f"差异化卖点 comparison_object_evidence_type 非法：{bridge_comparison_object_evidence_type}")
        if bridge_difference_domain and bridge_difference_domain not in DIFFERENTIATOR_ALLOWED_DOMAINS:
            raise ValueError(f"差异化卖点 difference_domain 非法：{bridge_difference_domain}")
        if bridge_difference_type and bridge_difference_type in LEGACY_DIFFERENCE_TYPES:
            raise ValueError(f"差异化卖点输入命中旧枚举：{bridge_difference_type}")
        if bridge_difference_type and bridge_difference_type not in DIFFERENTIATOR_DIFFERENCE_TYPES:
            raise ValueError(f"差异化卖点 difference_type 非法：{bridge_difference_type}")
        if bridge_comparison_object and not (bridge_difference_domain and bridge_difference_type):
            raise ValueError("桥接层已提供 comparison_object，但缺少 difference_domain 或 difference_type，按 PRD 必须 Crash Early。")
        if bridge_difference_domain and not bridge_difference_type:
            raise ValueError("桥接层已提供 difference_domain，但缺少 difference_type，按 PRD 必须 Crash Early。")
        if bridge_difference_type and not bridge_difference_domain:
            raise ValueError("桥接层已提供 difference_type，但缺少 difference_domain，按 PRD 必须 Crash Early。")
        if bridge_source_evidence and not (bridge_difference_domain and bridge_difference_type):
            raise ValueError("桥接层已提供 source_evidence，但缺少 difference_domain / difference_type，按 PRD 必须 Crash Early。")
        if (bridge_difference_domain or bridge_difference_type or bridge_source_evidence) and not bridge_comparison_object:
            if bridge_comparison_object_evidence_type != "null":
                raise ValueError("桥接层 comparison_object 为空时，comparison_object_evidence_type 必须为 null。")
        if bridge_difference_type == "自身卖点陈述":
            # P2：防护/防虫/防晒/安全等风险类目命中且证据含风险锚点时，注入隐式对比对象，
            # 不直接降级为纯“自身卖点陈述”；否则保持原协议清空 comparison_object。
            injected = self._inject_protection_risk_implicit_comparison(
                payload,
                bridge_source_evidence,
                bridge_comparison_object,
            )
            if injected is not None:
                (
                    bridge_difference_domain,
                    bridge_difference_type,
                    bridge_comparison_object,
                    bridge_comparison_object_evidence_type,
                ) = injected
            else:
                bridge_comparison_object = ""
                bridge_comparison_object_evidence_type = "null"
        if bridge_comparison_object and bridge_comparison_object_evidence_type == "null":
            raise ValueError("桥接层已提供 comparison_object，但 comparison_object_evidence_type= null，按 PRD 必须 Crash Early。")
        if not bridge_comparison_object and bridge_comparison_object_evidence_type != "null":
            raise ValueError("桥接层 comparison_object 为空，但 comparison_object_evidence_type 非 null，按 PRD 必须 Crash Early。")
        if bridge_comparison_object_evidence_type == "jtbd_inferred" and bridge_comparison_object not in {"", "同类旧方案", "旧形态方案"}:
            raise ValueError("桥接层 comparison_object=jtbd_inferred 时，仅允许同类旧方案/旧形态方案，按 PRD 必须 Crash Early。")
        if bridge_difference_domain and bridge_difference_type and bridge_difference_type not in DIFFERENTIATOR_DOMAIN_TYPES[bridge_difference_domain]:
            raise ValueError(
                f"差异化卖点 difference_domain / difference_type 跨域：{bridge_difference_domain} / {bridge_difference_type}"
            )
        if has_bridge_contract and not bridge_source_evidence:
            raise ValueError("桥接层已提供 comparison_object / difference_domain / difference_type，但缺少 source_evidence，按 PRD 必须 Crash Early。")

        if not has_bridge_contract:
            source_text = raw_text or selling
            if not source_text:
                raise ValueError("模块 1 差异化卖点断言失败：缺少可推导事实。")
            self._observe_legacy_comparison_object_keywords(source_text, payload)
            self._observe_legacy_difference_type_keywords(source_text, payload)
            raise ValueError(
                "模块 1 差异化卖点断言失败：非桥接路径已禁用；comparison_object / difference_domain / difference_type / source_evidence 必须由桥接层显式提供，按 PRD Crash Early。"
            )
        source_text = " ".join(bridge_source_evidence)
        if not source_text:
            raise ValueError("模块 1 差异化卖点断言失败：缺少可推导事实。")
        comparison_object = bridge_comparison_object
        difference_type = bridge_difference_type
        difference_domain = bridge_difference_domain
        conclusion = self._build_differentiator_conclusion(comparison_object, difference_type, source_text, payload)
        evidence_source_value = str(payload.bridge_evidence_source or "商品信息").strip() or "商品信息"
        if bridge_comparison_object_evidence_type == "jtbd_inferred":
            evidence_source = "JTBD推断"
        elif bridge_comparison_object_evidence_type == "user_provided":
            evidence_source = "人工标注"
        else:
            evidence_source = evidence_source_value
        evidence_chain = [DifferentiatorEvidence(evidence_source=evidence_source, evidence_text=item) for item in bridge_source_evidence]
        structured = StructuredDifferentiator(
            comparison_object=comparison_object,
            comparison_object_evidence_type=bridge_comparison_object_evidence_type,
            difference_domain=difference_domain,
            difference_type=difference_type,
            conclusion=conclusion,
            evidence_chain=evidence_chain,
            summary=conclusion,
        )
        self._assert_structured_differentiator(structured, payload)
        return structured

    def _validate_structured_differentiator(self, payload: Mapping[str, Any]) -> StructuredDifferentiator:
        required_fields = {"comparison_object", "difference_domain", "difference_type", "conclusion", "evidence_chain"}
        missing = [field for field in required_fields if field not in payload]
        if missing:
            raise ValueError(f"差异化卖点结构化四元组缺少字段：{missing}")
        if any(field in payload for field in LEGACY_DIFFERENTIATOR_FIELDS):
            hit = next(field for field in LEGACY_DIFFERENTIATOR_FIELDS if field in payload)
            raise ValueError(f"差异化卖点输入命中旧字段：{hit}")
        diff_domain = str(payload.get("difference_domain") or "").strip()
        diff_type = str(payload.get("difference_type") or "").strip()
        comparison_object_evidence_type = str(payload.get("comparison_object_evidence_type") or "null").strip() or "null"
        if comparison_object_evidence_type not in DIFFERENTIATOR_COMPARISON_OBJECT_EVIDENCE_TYPES:
            raise ValueError(f"差异化卖点 comparison_object_evidence_type 非法：{comparison_object_evidence_type}")
        if diff_type in LEGACY_DIFFERENCE_TYPES:
            raise ValueError(f"差异化卖点输入命中旧枚举：{diff_type}")
        evidence_chain_raw = payload.get("evidence_chain")
        if not isinstance(evidence_chain_raw, list) or not evidence_chain_raw:
            raise ValueError("差异化卖点 evidence_chain 必须是非空数组。")
        evidence_chain: list[DifferentiatorEvidence] = []
        for index, item in enumerate(evidence_chain_raw):
            if not isinstance(item, Mapping):
                raise ValueError(f"差异化卖点 evidence_chain[{index}] 必须是对象。")
            if "evidence_source" not in item or not str(item.get("evidence_source") or "").strip():
                raise ValueError(f"差异化卖点 evidence_chain[{index}] 缺少 evidence_source。")
            if "evidence_text" not in item or not str(item.get("evidence_text") or "").strip():
                raise ValueError(f"差异化卖点 evidence_chain[{index}] 缺少 evidence_text。")
            evidence_chain.append(
                DifferentiatorEvidence(
                    evidence_source=str(item.get("evidence_source") or "").strip(),
                    evidence_text=str(item.get("evidence_text") or "").strip(),
                )
            )
        structured = StructuredDifferentiator(
            comparison_object=str(payload.get("comparison_object") or "").strip(),
            comparison_object_evidence_type=comparison_object_evidence_type,
            difference_domain=diff_domain,
            difference_type=diff_type,
            conclusion=str(payload.get("conclusion") or "").strip(),
            evidence_chain=evidence_chain,
            summary=str(payload.get("summary") or "").strip(),
        )
        self._assert_structured_differentiator(structured, None)
        return structured

    def _assert_structured_differentiator(self, differentiator: StructuredDifferentiator, payload: DiagnosticInput | None = None) -> None:
        if differentiator.comparison_object and differentiator.comparison_object not in DIFFERENTIATOR_COMPARISON_OBJECTS:
            raise ValueError(f"差异化卖点 comparison_object 非法：{differentiator.comparison_object}")
        if differentiator.comparison_object_evidence_type not in DIFFERENTIATOR_COMPARISON_OBJECT_EVIDENCE_TYPES:
            raise ValueError(f"差异化卖点 comparison_object_evidence_type 非法：{differentiator.comparison_object_evidence_type}")
        if differentiator.difference_domain not in DIFFERENTIATOR_ALLOWED_DOMAINS:
            raise ValueError(f"差异化卖点 difference_domain 非法：{differentiator.difference_domain}")
        if differentiator.difference_type in LEGACY_DIFFERENCE_TYPES:
            raise ValueError(f"差异化卖点 difference_type 命中旧枚举：{differentiator.difference_type}")
        if differentiator.difference_type not in DIFFERENTIATOR_DIFFERENCE_TYPES:
            raise ValueError(f"差异化卖点 difference_type 非法：{differentiator.difference_type}")
        if differentiator.difference_type == "自身卖点陈述":
            if differentiator.comparison_object:
                raise ValueError("差异化卖点 difference_type=自身卖点陈述 时，comparison_object 必须为空。")
            if differentiator.comparison_object_evidence_type != "null":
                raise ValueError("差异化卖点 difference_type=自身卖点陈述 时，comparison_object_evidence_type 必须为 null。")
        if differentiator.difference_type not in DIFFERENTIATOR_DOMAIN_TYPES[differentiator.difference_domain]:
            raise ValueError(
                f"差异化卖点 difference_domain / difference_type 跨域：{differentiator.difference_domain} / {differentiator.difference_type}"
            )
        if not differentiator.conclusion.strip():
            raise ValueError("差异化卖点 conclusion 不允许为空。")
        if not differentiator.evidence_chain:
            raise ValueError("差异化卖点 evidence_chain 不能为空。")
        evidence_texts: list[str] = []
        judge_evidence_clauses: list[str] = []
        for index, evidence in enumerate(differentiator.evidence_chain):
            if evidence.evidence_source not in DIFFERENTIATOR_EVIDENCE_SOURCES:
                raise ValueError(f"差异化卖点 evidence_chain[{index}].evidence_source 非法：{evidence.evidence_source}")
            if not evidence.evidence_text.strip():
                raise ValueError(f"差异化卖点 evidence_chain[{index}].evidence_text 不允许为空。")
            if any(token in evidence.evidence_text for token in MARKETING_TOKENS):
                raise ValueError("模块 1 差异化卖点断言失败：证据链存在营销词。")
            evidence_texts.append(evidence.evidence_text)
            segments = [segment.strip() for segment in re.split(r"[，,。；;、/]+", evidence.evidence_text) if segment.strip()]
            for segment in segments or [evidence.evidence_text.strip()]:
                if segment not in judge_evidence_clauses:
                    judge_evidence_clauses.append(segment)
        joined_evidence = " | ".join(judge_evidence_clauses) if judge_evidence_clauses else " ".join(evidence_texts)
        self._assert_specific_old_scheme_anchor_requirement(differentiator, joined_evidence)
        self._assert_effect_enhancement_not_convenience_only(differentiator, joined_evidence)
        self._assert_relative_difference_type_anchor_requirement(differentiator, joined_evidence)
        self._assert_self_statement_relative_residue_block(differentiator, joined_evidence)
        self._assert_differentiator_semantic_support(differentiator, joined_evidence, payload)
        if any(token in differentiator.conclusion for token in MARKETING_TOKENS):
            raise ValueError("模块 1 差异化卖点断言失败：结论存在营销词。")
        if differentiator.summary and any(token in differentiator.summary for token in MARKETING_TOKENS):
            raise ValueError("模块 1 差异化卖点断言失败：summary 存在营销词。")

    def _observe_legacy_comparison_object_keywords(self, text: str, payload: DiagnosticInput) -> None:
        product_text = f"{payload.product_name} {payload.leaf_category} {text}"
        for comparison_object in ("同赛道竞品", "跨品类旧动作", "旧形态方案", "同类旧方案"):
            matched_keyword = self._find_first_keyword(product_text, COMPARISON_OBJECT_KEYWORDS[comparison_object])
            if matched_keyword:
                self._record_keyword_rule_trace(
                    field_name="legacy_comparison_object_observation",
                    output_value=comparison_object,
                    rule_path=f"product_diagnoser.comparison_object_keywords.{comparison_object}",
                    matched_keyword=matched_keyword,
                )
                return

    def _observe_legacy_difference_type_keywords(self, text: str, payload: DiagnosticInput) -> None:
        normalized_text = f"{payload.product_name} {payload.core_selling_point} {text}"
        for diff_type in ("步骤压缩", "效果增强", "风险降低", "成本优化", "体验升级", "新形态替代", "信任缓释"):
            matched_keyword = self._find_first_keyword(normalized_text, DIFFERENTIATOR_TYPE_KEYWORDS[diff_type])
            if matched_keyword:
                self._record_keyword_rule_trace(
                    field_name="legacy_difference_type_observation",
                    output_value=diff_type,
                    rule_path=f"product_diagnoser.differentiator_type_keywords.{diff_type}",
                    matched_keyword=matched_keyword,
                )
                return

    def _infer_comparison_object(self, text: str, payload: DiagnosticInput) -> str:
        """遗留观测逻辑：正式路径已禁用，仅保留关键词命中日志。"""
        self._observe_legacy_comparison_object_keywords(text, payload)
        raise ValueError("差异化卖点 comparison_object 必须由桥接层显式提供，正式路径已禁用本地推断，按 PRD 必须 Crash Early。")

    def _infer_difference_type(self, text: str, payload: DiagnosticInput) -> str:
        """遗留观测逻辑：正式路径已禁用，仅保留关键词命中日志。"""
        self._observe_legacy_difference_type_keywords(text, payload)
        raise ValueError("差异化卖点 difference_type 必须由桥接层显式提供，正式路径已禁用本地推断，按 PRD 必须 Crash Early。")

    def _infer_difference_domain(self, difference_type: str) -> str:
        for difference_domain, difference_types in DIFFERENTIATOR_DOMAIN_TYPES.items():
            if difference_type in difference_types:
                return difference_domain
        raise ValueError(f"差异化卖点 difference_type 缺少对应 difference_domain：{difference_type}")

    def _build_differentiator_conclusion(
        self,
        comparison_object: str,
        difference_type: str,
        text: str,
        payload: DiagnosticInput,
    ) -> str:
        conclusion = self.differentiator_conclusion_llm.generate(
            comparison_object=comparison_object,
            difference_type=difference_type,
            evidence_text=text,
            payload=payload,
        )
        if self._conclusion_introduces_unsupported_old_scheme_claim(difference_type, text, conclusion):
            raise ValueError("模块 1 差异化卖点 conclusion 引入了 evidence 未支撑的旧方案主张，按 PRD 必须 Crash Early。")
        return conclusion

    def _build_differentiator_conclusion_from_evidence(
        self,
        comparison_object: str,
        difference_type: str,
        evidence_text: str,
    ) -> str:
        clauses = self._extract_supported_evidence_clauses(difference_type, evidence_text)
        if not clauses:
            raise ValueError("差异化卖点无法从 evidence_chain 推导 conclusion，按 PRD 必须 Crash Early。")
        comparison_prefix = "" if difference_type == "自身卖点陈述" else self._render_comparison_object_prefix(comparison_object)
        clause = clauses[0]
        if difference_type == "自身卖点陈述":
            return clause
        if difference_type == "步骤压缩":
            if clause.startswith("快速"):
                clause = f"更快完成{clause.removeprefix('快速')}"
            elif not self._contains_any(clause, {"少", "一步", "更快", "省时", "免洗", "快"}):
                clause = f"围绕{clause}减少完成需求的步骤负担"
        elif difference_type == "效果增强":
            if not self._contains_any(clause, {"更", "强", "稳", "去黄", "美白", "清洁", "口气", "修复", "改善", "持妆", "显色", "立体"}):
                clause = f"更突出{clause}这类结果表现"
        elif difference_type == "风险降低" and not self._contains_any(clause, {"风险", "安全", "顾虑", "安心", "保护", "刺激", "防"}):
            clause = f"降低{clause}相关顾虑"
        elif difference_type == "成本优化" and not self._contains_any(clause, {"省", "性价比", "成本", "划算"}):
            clause = f"围绕{clause}降低使用成本"
        elif difference_type == "体验升级" and not self._contains_any(clause, {"舒适", "轻松", "新手", "易上手", "体验"}):
            clause = f"围绕{clause}提升使用体验"
        elif difference_type == "新形态替代":
            clause = clauses[0]
            if comparison_object == "旧形态方案":
                return clause
        elif difference_type == "信任缓释" and not self._contains_any(clause, {"认证", "背书", "资质", "专利", "官方", "机构"}):
            clause = f"用{clause}降低决策顾虑"
        if not comparison_prefix:
            return clause
        return f"相对{comparison_prefix}，{clause}"

    def _conclusion_introduces_unsupported_old_scheme_claim(self, difference_type: str, evidence_text: str, conclusion: str) -> bool:
        if difference_type != "新形态替代":
            return False
        old_scheme_tokens = ("替代", "旧方案", "旧形态", "承接")
        if not any(token in conclusion for token in old_scheme_tokens):
            return False
        support_tokens = old_scheme_tokens + (
            "喷雾",
            "喷雾剂",
            "贴片",
            "液体敷料",
            "液体",
            "棒状",
            "滚珠",
            "日抛",
            "免洗",
            "凝胶",
            "敷料",
        )
        return not any(token in evidence_text for token in support_tokens)

    def _extract_supported_evidence_clauses(self, difference_type: str, text: str) -> list[str]:
        normalized_text = re.sub(r"[\n\r\t]+", " ", str(text or "")).strip()
        segments = [segment.strip() for segment in re.split(r"[，,。；;、/]+", normalized_text) if segment.strip()]
        matched: list[str] = []
        for segment in segments:
            if self._supports_difference_type_evidence(difference_type, segment):
                cleaned = self._sanitize_evidence_clause(segment)
                if cleaned and cleaned not in matched:
                    matched.append(cleaned)
        if matched:
            return matched
        if self._supports_difference_type_evidence(difference_type, normalized_text):
            cleaned = self._sanitize_evidence_clause(normalized_text)
            if cleaned:
                return [cleaned]
        return []

    def _sanitize_evidence_clause(self, clause: str) -> str:
        cleaned = re.sub(r"^(相比|比|相对)(普通款|同类旧方案|旧方案|传统方案|传统用法|传统流程)", "", clause).strip(" ：:，,。；;")
        cleaned = re.sub(r"^(普通款|同类旧方案|旧方案|传统方案|传统用法|传统流程)", "", cleaned).strip(" ：:，,。；;")
        if cleaned.startswith("解决"):
            cleaned = f"更能{cleaned}"
        return cleaned

    def _render_comparison_object_prefix(self, comparison_object: str) -> str:
        mapping = {
            "同类旧方案": "旧方案",
            "同赛道竞品": "同赛道竞品",
            "跨品类旧动作": "旧动作",
            "旧形态方案": "旧形态方案",
        }
        return mapping.get(comparison_object, comparison_object)

    def _assert_specific_old_scheme_anchor_requirement(
        self,
        differentiator: StructuredDifferentiator,
        joined_evidence: str,
    ) -> None:
        if differentiator.difference_type == "自身卖点陈述":
            return
        if differentiator.comparison_object != "同类旧方案":
            return
        if differentiator.comparison_object_evidence_type in {"jtbd_inferred", "user_provided"}:
            return
        if any(evidence.evidence_source == "JTBD推断" for evidence in differentiator.evidence_chain):
            return
        if self._contains_any(joined_evidence, DIFFERENTIATOR_OLD_SCHEME_REQUIRED_TOKENS):
            return
        raise ValueError(
            "差异化卖点断言失败：comparison_object=同类旧方案，但 evidence_chain 缺少具体旧方案锚点，按 PRD 必须 Crash Early。"
        )

    def _assert_effect_enhancement_not_convenience_only(
        self,
        differentiator: StructuredDifferentiator,
        joined_evidence: str,
    ) -> None:
        if differentiator.difference_type != "效果增强":
            return
        has_effect_anchor = self._contains_any(joined_evidence, DIFFERENTIATOR_EFFECT_ENHANCEMENT_TOKENS)
        has_convenience_only_anchor = self._contains_any(joined_evidence, DIFFERENTIATOR_CONVENIENCE_ONLY_TOKENS)
        if has_convenience_only_anchor and not has_effect_anchor:
            raise ValueError(
                "差异化卖点断言失败：evidence_chain 仅体现便利型锚点，不足以支撑 difference_type=效果增强，按 PRD 必须 Crash Early。"
            )

    def _assert_relative_difference_type_anchor_requirement(
        self,
        differentiator: StructuredDifferentiator,
        joined_evidence: str,
    ) -> None:
        if differentiator.difference_type not in DIFFERENTIATOR_RELATIVE_FUNCTIONAL_TYPES:
            return
        if differentiator.comparison_object_evidence_type not in {"jtbd_inferred", "null"}:
            return
        relative_anchor_tokens = DIFFERENTIATOR_RELATIVE_DIFFERENCE_TYPE_ANCHORS.get(differentiator.difference_type)
        if not relative_anchor_tokens:
            return
        if self._contains_any(joined_evidence, relative_anchor_tokens):
            return
        raise ValueError(
            "差异化卖点断言失败：相对性 difference_type 缺少比较锚点，不得依赖 jtbd_inferred/null 注入，按 PRD 必须 Crash Early。"
        )

    def _contains_relative_semantic_residue(self, text: str) -> bool:
        normalized = str(text or "").strip()
        return any(token in normalized for token in DIFFERENTIATOR_RELATIVE_SEMANTIC_TOKENS)

    def _assert_self_statement_relative_residue_block(
        self,
        differentiator: StructuredDifferentiator,
        joined_evidence: str,
    ) -> None:
        if differentiator.difference_type != "自身卖点陈述":
            return
        targets = [differentiator.conclusion, differentiator.summary]
        if not any(self._contains_relative_semantic_residue(text) for text in targets if text):
            return
        if self._contains_any(joined_evidence, DIFFERENTIATOR_RELATIVE_DIFFERENCE_TYPE_ANCHORS["步骤压缩"]):
            return
        if self._contains_any(joined_evidence, DIFFERENTIATOR_RELATIVE_DIFFERENCE_TYPE_ANCHORS["效果增强"]):
            return
        if self._contains_any(joined_evidence, DIFFERENTIATOR_RELATIVE_DIFFERENCE_TYPE_ANCHORS["风险降低"]):
            return
        if self._contains_any(joined_evidence, DIFFERENTIATOR_RELATIVE_DIFFERENCE_TYPE_ANCHORS["成本优化"]):
            return
        if self._contains_any(joined_evidence, DIFFERENTIATOR_RELATIVE_DIFFERENCE_TYPE_ANCHORS["体验升级"]):
            return
        if self._contains_any(joined_evidence, DIFFERENTIATOR_RELATIVE_DIFFERENCE_TYPE_ANCHORS["新形态替代"]):
            return
        raise ValueError("差异化卖点断言失败：自身卖点陈述样本残留未被证据支撑的相对语义，按 PRD 必须 Crash Early。")

    def _allow_effect_enhancement_whitelist(
        self,
        differentiator: StructuredDifferentiator,
        joined_evidence: str,
    ) -> bool:
        if differentiator.difference_type != "效果增强":
            return False
        if differentiator.comparison_object != "同类旧方案":
            return False
        if differentiator.comparison_object_evidence_type != "jtbd_inferred":
            return False
        if not self._contains_any(joined_evidence, DIFFERENTIATOR_EFFECT_ENHANCEMENT_TOKENS):
            return False
        strong_claim_groups = DIFFERENTIATOR_CONCLUSION_STRONG_CLAIMS.get(differentiator.difference_type, ())
        for token_group in strong_claim_groups:
            if all(token in differentiator.conclusion for token in token_group):
                return False
        return True

    def _assert_differentiator_semantic_support(
        self,
        differentiator: StructuredDifferentiator,
        joined_evidence: str,
        payload: DiagnosticInput | None,
    ) -> None:
        if not DIFFERENTIATOR_JUDGE_RULES:
            raise ValueError("差异化卖点质检规则库为空，无法执行语义一致性校验。")
        if payload is None:
            return
        if self._allow_effect_enhancement_whitelist(differentiator, joined_evidence):
            return
        judge_result = self.differentiator_semantic_judge_llm.judge(
            comparison_object=differentiator.comparison_object,
            difference_domain=differentiator.difference_domain,
            difference_type=differentiator.difference_type,
            conclusion=differentiator.conclusion,
            evidence_text=joined_evidence,
            payload=payload,
        )
        if not judge_result["supports_difference_type"]:
            raise ValueError(
                f"差异化卖点语义一致性校验失败：Judge 判定 supports_difference_type=false，已按 PRD Crash Early。原因：{judge_result['reason']}"
            )
        if not judge_result["supports_conclusion"]:
            raise ValueError(
                f"差异化卖点语义一致性校验失败：Judge 判定 supports_conclusion=false，已按 PRD Crash Early。原因：{judge_result['reason']}"
            )

    def _supports_difference_type_evidence(self, difference_type: str, text: str) -> bool:
        dummy_payload = DiagnosticInput(
            leaf_category="",
            shop_name="",
            product_name="",
            core_selling_point="",
        )
        self._observe_legacy_difference_type_keywords(text, dummy_payload)
        return False

    def _evidence_supports_conclusion(self, difference_type: str, evidence_text: str, conclusion: str) -> bool:
        strong_claim_groups = DIFFERENTIATOR_CONCLUSION_STRONG_CLAIMS.get(difference_type, ())
        for claim_group in strong_claim_groups:
            conclusion_hit = any(claim in conclusion for claim in claim_group)
            if conclusion_hit and not any(claim in evidence_text for claim in claim_group):
                return False
        conclusion_numeric_claims = set(re.findall(r"\d+(?:\.\d+)?\s*(?:小时|分钟|秒|天|周|月|次|倍|%|级)", conclusion))
        if conclusion_numeric_claims and not all(claim in evidence_text for claim in conclusion_numeric_claims):
            return False
        return True

    def _stringify_differentiator(self, differentiator: Any) -> str:
        if isinstance(differentiator, StructuredDifferentiator):
            evidence_text = " ".join(item.evidence_text for item in differentiator.evidence_chain)
            return " ".join(
                part for part in [
                    differentiator.summary or differentiator.conclusion,
                    differentiator.comparison_object,
                    differentiator.difference_domain,
                    differentiator.difference_type,
                    evidence_text,
                ] if part
            )
        if isinstance(differentiator, Mapping):
            return json.dumps(differentiator, ensure_ascii=False)
        return str(differentiator or "").strip()

    def _module1_joined_text(self, module1_output: Module1Output) -> str:
        return "｜".join(
            str(value)
            for value in [
                module1_output.second_level_category,
                module1_output.third_level_category,
                module1_output.leaf_category,
                module1_output.shop_name,
                module1_output.product_name,
                module1_output.price,
                module1_output.core_selling_point,
                module1_output.core_selling_point_source,
                module1_output.target_people,
                self._stringify_differentiator(module1_output.differentiator),
            ]
            if value
        )

    def _assert_module1_output(self, module1_output: Module1Output, payload: DiagnosticInput) -> None:
        expected_keys = {
            "leaf_category",
            "shop_name",
            "second_level_category",
            "third_level_category",
            "product_name",
            "price",
            "core_selling_point",
            "core_selling_point_source",
            "target_people",
            "differentiator",
            # 第五批：原始未截断目标人群信号（供 gift_context 识别）。
            "target_people_raw",
        }
        if set(module1_output.to_dict().keys()) != expected_keys:
            raise ValueError("模块 1 出参字段越界。")
        if module1_output.core_selling_point_source not in TITLE_CORE_SELLING_POINT_ALLOWED_SOURCES:
            raise ValueError(f"模块 1 core_selling_point_source 非法：{module1_output.core_selling_point_source}")
        if (
            module1_output.core_selling_point_source == "title_llm_extracted"
            and _normalize_compact_text(module1_output.core_selling_point) == _normalize_compact_text(module1_output.product_name)
        ):
            raise ValueError("模块 1 core_selling_point 断言失败：标题抽取结果等于完整商品标题。")
        generic_people_tokens = {"全网", "大众", "所有人", "年轻人", "男女通用", "用户"}
        if any(token in module1_output.target_people for token in generic_people_tokens):
            raise ValueError("模块 1 目标人群断言失败：存在泛人群词。")
        pure_scene_tokens = {"下班后", "深夜", "约会前", "通勤路上"}
        if any(token in module1_output.target_people for token in pure_scene_tokens):
            raise ValueError("模块 1 目标人群断言失败：存在纯场景词。")
        self._assert_structured_differentiator(module1_output.differentiator, payload)
        if isinstance(payload.differentiator, str) and payload.differentiator.strip() and module1_output.differentiator.summary == payload.differentiator.strip():
            raise ValueError("模块 1 差异化卖点断言失败：疑似直抄原始营销文案，未完成客观翻译。")

    def _resolve_jtbd(self, payload: DiagnosticInput, module1_output: Module1Output) -> tuple[JTBDProposal, list[str]]:
        warnings: list[str] = []
        errors: list[str] = []
        rule_context = self._build_rule_tree_context(payload, module1_output)
        candidate_tasks = list(rule_context["candidate_tasks"])
        if len(candidate_tasks) == 1:
            proposal = self._build_rule_tree_proposal(rule_context)
            self._assert_proposal(proposal)
            return proposal, warnings

        for _ in range(self.max_retries):
            try:
                proposal = self._classify_once(payload, module1_output, candidate_tasks=candidate_tasks)
                proposal = self._merge_rule_tree_context(proposal, rule_context)
                self._assert_proposal(proposal)
                return proposal, warnings
            except Exception as exc:  # noqa: BLE001
                errors.append(str(exc))
                if any(keyword in str(exc) for keyword in ("越权", "圈层共识", "缺少合法 sub_task", "PRD §5.1.5", "evidence_chain", "gate_reasons")):
                    raise
        fallback = self._build_final_fallback(module1_output, errors, rule_context)
        warnings.append("[Task_Fallback_Warning] 规则树候选池内的分类连续失败，已降级为可审计兜底任务。")
        return fallback, warnings

    def _classify_once(
        self,
        payload: DiagnosticInput,
        module1_output: Module1Output,
        *,
        candidate_tasks: list[str],
    ) -> JTBDProposal:
        if self.classifier is not None:
            raw = self.classifier(payload)
            if isinstance(raw, JTBDProposal):
                return raw
            return JTBDProposal(**dict(raw))
        enriched_payload = DiagnosticInput(
            leaf_category=module1_output.leaf_category,
            shop_name=module1_output.shop_name,
            brand_name=payload.brand_name,
            product_name=module1_output.product_name,
            price=module1_output.price,
            core_selling_point=module1_output.core_selling_point,
            target_people=module1_output.target_people,
            differentiator=module1_output.differentiator,
            product_id=payload.product_id,
            sample_tags=payload.sample_tags,
            engine_node=payload.engine_node,
        )
        return self.llm_classifier.classify(enriched_payload, candidate_tasks=candidate_tasks)

    def _assert_proposal(self, proposal: JTBDProposal) -> None:
        if proposal.primary_task == NEEDS_REVIEW:
            # PRD-1.2：needs_review 中止态，协议层短路放行，不校验 domain/越权/TASK_DOMAIN_MAP。
            return
        if proposal.primary_task not in VALID_JTBD:
            raise ValueError(f"LLM 输出了非法任务名: {proposal.primary_task}")
        expected_domain = TASK_DOMAIN_MAP[proposal.primary_task]
        if proposal.domain != expected_domain:
            raise ValueError(f"LLM 输出 domain 与 task 不一致: {proposal.domain} vs {proposal.primary_task}")
        if proposal.primary_task == "阶层与审美发信" and proposal.sub_task not in ALLOWED_SUB_TASKS:
            raise ValueError("LLM 输出阶层与审美发信时必须带合法 sub_task。")
        if proposal.domain == SOCIAL_DOMAIN:
            if not proposal.gate_reasons:
                raise ValueError("社会域输出缺少 gate_reasons（必须含真实命中的证据描述）。")
            banned_template_fragments = (
                "符合圈层共识，因此进入社会任务",
                "命中社会域门槛",
                "规则树直接判定为社会任务",
            )
            if any(frag in self._join_reasoning(proposal) for frag in banned_template_fragments):
                raise ValueError("社会域 reasoning 命中预设模板文本，违反 PRD §5.1.5。")
        if proposal.candidate_tasks and proposal.primary_task not in proposal.candidate_tasks:
            raise ValueError(f"JTBD 输出越权：{proposal.primary_task} 不在规则树候选池内。")
        if not proposal.triggered_rule.strip():
            raise ValueError("JTBD 输出缺少 triggered_rule。")
        if not proposal.evidence_chain:
            raise ValueError("JTBD 输出缺少 evidence_chain。")
        if not proposal.candidate_tasks:
            raise ValueError("JTBD 输出缺少 candidate_tasks。")
        if not isinstance(proposal.candidate_reasons, dict):
            raise ValueError("JTBD 输出缺少 candidate_reasons。")
        if not isinstance(proposal.excluded_tasks, dict):
            raise ValueError("JTBD 输出缺少 excluded_tasks。")
        if not isinstance(proposal.gate_reasons, list):
            raise ValueError("JTBD 输出缺少 gate_reasons。")
        if not isinstance(proposal.trace_tokens, list):
            raise ValueError("JTBD 输出缺少 trace_tokens。")
        if proposal.domain == FUNCTIONAL_DOMAIN:
            if not isinstance(proposal.functional_facts, list):
                raise ValueError("JTBD 输出缺少 functional_facts。")
            if not isinstance(proposal.candidate_pool, list):
                raise ValueError("JTBD 输出缺少 candidate_pool。")
            if not isinstance(proposal.subcategory_context, str):
                raise ValueError("JTBD 输出缺少 subcategory_context。")
            if not isinstance(proposal.veto_trace, list):
                raise ValueError("JTBD 输出缺少 veto_trace。")
        if proposal.subcategory_context == "paper_products" and proposal.primary_task == PAPER_ESCALATABLE_TASK and not self._paper_upgrade_has_closed_loop(proposal.functional_facts):
            raise ValueError("纸品升级候选缺少对象+状态+动作闭环，不允许放行。")
        if proposal.primary_task == "物理安全与风险规避" and not self._proposal_has_physical_safety_evidence(proposal):
            raise ValueError("物理安全任务缺少最小证据，不允许放行。")

    def _build_rule_tree_context(self, payload: DiagnosticInput, module1_output: Module1Output) -> dict[str, Any]:
        text = self._module1_joined_text(module1_output)
        evidence_chain = self._build_jtbd_evidence_chain(module1_output)

        if self._is_physical_safety_fact(text):
            gate_reasons = ["Stage A 物理安全门槛成立：存在客观伤害或可预见风险证据。"]
            return {
                "candidate_tasks": ["物理安全与风险规避"],
                "candidate_reasons": {"物理安全与风险规避": gate_reasons.copy()},
                "excluded_tasks": {
                    "生存/运转维系": ["已被物理安全强唯一前置裁决覆盖。"],
                    "缺陷修复/冲突消除": ["已被物理安全强唯一前置裁决覆盖。"],
                    "降本增效/懒人替代": ["已被物理安全强唯一前置裁决覆盖。"],
                },
                "triggered_rule": "safety_priority_rule",
                "gate_reasons": gate_reasons,
                "trace_tokens": [],
                "evidence_chain": evidence_chain,
                "functional_facts": [],
                "candidate_pool": [
                    {
                        "task_name": "物理安全与风险规避",
                        "supporting_fact_ids": [],
                        "mapping_reason": "Stage A 已基于客观伤害/可预见风险证据完成硬锁定。",
                        "priority": "hard_gate",
                    }
                ],
                "subcategory_context": "stage_a_hard_gate",
                "veto_trace": [],
                "reasoning_path": [
                    "Stage A：命中物理安全硬边界。",
                    "物理安全在前置层唯一裁决，后续阶段不再重复改写 primary_task。",
                ],
                "reasoning": "规则树前置锁定为物理安全与风险规避。",
            }

        social_task, social_evidence_lines, social_rejection_reasons = self._infer_social_task(payload, module1_output, text)
        if social_task:
            gate_reasons = list(social_evidence_lines)
            reasoning_text = " ".join(social_evidence_lines)
            return {
                "candidate_tasks": [social_task],
                "candidate_reasons": {social_task: gate_reasons.copy()},
                "excluded_tasks": {},
                "triggered_rule": "social_priority_rule",
                "gate_reasons": gate_reasons,
                "trace_tokens": [],
                "evidence_chain": evidence_chain,
                "functional_facts": [],
                "candidate_pool": [
                    {
                        "task_name": social_task,
                        "supporting_fact_ids": [],
                        "mapping_reason": social_evidence_lines[0] if social_evidence_lines else "Stage A 社会任务证据闭合。",
                        "priority": "hard_gate",
                    }
                ],
                "subcategory_context": "stage_a_hard_gate",
                "veto_trace": [],
                "reasoning_path": [
                    f"Stage A：社会任务证据链成立，候选收敛到 {social_task}。",
                    *social_evidence_lines,
                ],
                "reasoning": reasoning_text or f"Stage A 命中 {social_task} 的商品属性证据链。",
            }

        emotional_task = self._infer_emotional_task(payload, module1_output, text)
        if emotional_task:
            gate_reasons = ["Stage A 情绪域门槛成立：同时满足情绪目标证据与商品属性门槛。"]
            return {
                "candidate_tasks": [emotional_task],
                "candidate_reasons": {emotional_task: gate_reasons.copy()},
                "excluded_tasks": {},
                "triggered_rule": "emotional_priority_rule",
                "gate_reasons": gate_reasons,
                "trace_tokens": [],
                "evidence_chain": evidence_chain,
                "functional_facts": [],
                "candidate_pool": [
                    {
                        "task_name": emotional_task,
                        "supporting_fact_ids": [],
                        "mapping_reason": "Stage A 已基于情绪域门槛完成候选唯一收敛。",
                        "priority": "hard_gate",
                    }
                ],
                "subcategory_context": "stage_a_hard_gate",
                "veto_trace": [],
                "reasoning_path": [
                    f"Stage A：命中情绪任务门槛，候选收敛到 {emotional_task}。",
                    "该商品同时具备情绪目标证据与高端/享乐/疗愈属性。",
                ],
                "reasoning": "规则树直接判定为情绪任务。",
            }

        functional_candidate_context = self._infer_functional_candidates(module1_output, text)
        functional_candidates = functional_candidate_context["candidate_tasks"]
        return {
            "candidate_tasks": functional_candidates,
            "candidate_reasons": functional_candidate_context["candidate_reasons"],
            "excluded_tasks": functional_candidate_context["excluded_tasks"],
            "triggered_rule": "functional_default_rule",
            "gate_reasons": ["Stage A 未命中物理安全 / 社会域 / 情绪域强门槛，进入功能域候选池。"],
            "trace_tokens": functional_candidate_context["trace_tokens"],
            "evidence_chain": evidence_chain,
            "functional_facts": functional_candidate_context["functional_facts"],
            "candidate_pool": functional_candidate_context["candidate_pool"],
            "subcategory_context": functional_candidate_context["subcategory_context"],
            "veto_trace": functional_candidate_context["veto_trace"],
            # PRD-1.2：透传 Stage B 空候选 needs_review 的 non_selected_task_reasons（gate=dropped）。
            "non_selected_task_reasons": functional_candidate_context.get("non_selected_task_reasons", []),
            "reasoning_path": [
                "Stage B：未命中社会域/情绪域/安全域的强唯一条件，回到功能域候选池。",
                f"Stage B 子包路由：{functional_candidate_context['subcategory_context']}。",
                f"Stage B 事实层抽取 {len(functional_candidate_context['functional_facts'])} 条 functional_facts。",
                f"功能域候选池为：{'、'.join(functional_candidates)}。",
            ],
            "reasoning": "规则树仅做功能域候选池收窄，最终归因留给候选池内归并。",
        }

    def _build_jtbd_evidence_chain(self, module1_output: Module1Output) -> list[dict[str, str]]:
        evidence_chain: list[dict[str, str]] = []
        if module1_output.target_people:
            evidence_chain.append({"evidence_source": "目标人群", "evidence_text": module1_output.target_people})
        if module1_output.core_selling_point:
            evidence_chain.append({"evidence_source": "核心卖点", "evidence_text": module1_output.core_selling_point})
        evidence_chain.append({"evidence_source": "差异结论", "evidence_text": module1_output.differentiator.conclusion})
        for item in module1_output.differentiator.evidence_chain:
            evidence = {"evidence_source": item.evidence_source, "evidence_text": item.evidence_text}
            if evidence not in evidence_chain:
                evidence_chain.append(evidence)
        return evidence_chain

    def _is_physical_safety_fact(self, text: str) -> bool:
        # P0：物理安全前置词表统一走业务字典 product_diagnoser.jtbd.physical_safety_tokens，
        # 已覆盖防虫/防护类风险词（驱蚊/防虫/蚊虫叮咬/瘙痒/红肿/过敏等），避免驱蚊液在 Stage A 漏判。
        return self._contains_any(text, PHYSICAL_SAFETY_TOKENS)

    def _infer_social_task(
        self,
        payload: DiagnosticInput,
        module1_output: Module1Output,
        text: str,
    ) -> tuple[str | None, list[str], list[str]]:
        """
        PRD §5.1.5：社会任务必须基于商品属性证据链判定，关键词命中不构成证据。
        返回 (task_or_none, evidence_lines, rejection_reasons)。
        evidence_lines 仅写入本次真实命中的证据，禁止使用社会任务专有词作为模板。
        证据不足时返回 None，并将缺失原因写入 rejection_reasons，由上游决定 Crash Early 或路由到非社会域。
        """

        def hits(tokens: set[str]) -> list[str]:
            return sorted({t for t in tokens if t and t in text})

        evidence_lines: list[str] = []
        rejection_reasons: list[str] = []

        # 排除规则：基础功能型 / 低外显私密自用品 / 仅营销修辞
        basic_functional_hits = hits(SOCIAL_BASIC_FUNCTIONAL_EXCLUDE_TOKENS)
        low_visibility_hits = hits(SOCIAL_LOW_VISIBILITY_EXCLUDE_TOKENS)
        marketing_only_hits = hits(SOCIAL_MARKETING_RHETORIC_ONLY_TOKENS)

        relationship_hits = hits(SOCIAL_RELATIONSHIP_OBJECT_TOKENS)
        caregiving_hits = hits(SOCIAL_CAREGIVING_VERB_TOKENS)
        gift_hits = hits(SOCIAL_GIFT_SCENE_TOKENS)
        circle_hits = hits(SOCIAL_CIRCLE_IDENTITY_TOKENS)
        visibility_hits = hits(SOCIAL_STATUS_VISIBILITY_TOKENS)
        barrier_hits = hits(SOCIAL_STATUS_BARRIER_TOKENS)
        consensus_hits = hits(SOCIAL_STATUS_CONSENSUS_TOKENS)
        is_high_premium = self._is_high_premium(payload)

        # 1. 照护与责任履行：必须同时存在【关系对象】+【照护/责任动作】，且非基础功能型/私密自用品
        if relationship_hits and caregiving_hits and not basic_functional_hits and not low_visibility_hits:
            evidence_lines.append(
                f"商品文本同时出现关系对象（{', '.join(relationship_hits)}）"
                f"与照护/责任语义（{', '.join(caregiving_hits)}），证明商品承担关系内照护交付任务。"
            )
            return "照护与责任履行", evidence_lines, rejection_reasons

        # 2. 礼赠与关系表达：必须同时存在【礼赠场景】+【明确关系对象或礼赠对象】
        if gift_hits and relationship_hits and not low_visibility_hits:
            evidence_lines.append(
                f"商品文本出现礼赠场景（{', '.join(gift_hits)}）"
                f"与明确关系对象（{', '.join(relationship_hits)}），证明商品承担关系表达/送礼任务。"
            )
            return "礼赠与关系表达", evidence_lines, rejection_reasons

        # 3. 圈层认同：必须有具名稳定圈层证据，泛化潮流词不计
        if circle_hits and not basic_functional_hits:
            evidence_lines.append(
                f"商品文本出现可追溯的具名圈层证据（{', '.join(circle_hits)}），"
                f"证明商品已被该群体稳定识别。"
            )
            return "圈层认同（圈层归属/身份锚定）", evidence_lines, rejection_reasons

        # 4. 阶层与审美发信：必须同时满足三道硬门槛（高外显 + 获取门槛/稀缺 + 圈层/审美共识）
        consensus_satisfied = bool(consensus_hits) or bool(circle_hits)
        barrier_satisfied = bool(barrier_hits) or is_high_premium
        if visibility_hits and barrier_satisfied and consensus_satisfied and not low_visibility_hits and not basic_functional_hits:
            barrier_evidence_parts: list[str] = []
            if barrier_hits:
                barrier_evidence_parts.append(f"获取门槛/稀缺词（{', '.join(barrier_hits)}）")
            if is_high_premium:
                barrier_evidence_parts.append("价格水位达到高水位/高端门槛")
            consensus_evidence_parts: list[str] = []
            if consensus_hits:
                consensus_evidence_parts.append(f"审美共识词（{', '.join(consensus_hits)}）")
            if circle_hits:
                consensus_evidence_parts.append(f"圈层识别证据（{', '.join(circle_hits)}）")
            evidence_lines.append(
                f"高外显证据（{', '.join(visibility_hits)}）；"
                f"{'；'.join(barrier_evidence_parts)}；"
                f"{'；'.join(consensus_evidence_parts)}。三道硬门槛全部命中。"
            )
            return "阶层与审美发信", evidence_lines, rejection_reasons

        # 未命中任何社会任务 —— 收集拒绝理由，便于上游审计
        if basic_functional_hits:
            rejection_reasons.append(
                f"命中基础功能型排除条件：{', '.join(basic_functional_hits)}，默认回落到功能任务。"
            )
        if low_visibility_hits:
            rejection_reasons.append(
                f"命中低外显/私密自用品排除条件：{', '.join(low_visibility_hits)}，不进入社会任务。"
            )
        if marketing_only_hits and not (relationship_hits or circle_hits or gift_hits or visibility_hits):
            rejection_reasons.append(
                f"仅出现营销修辞词（{', '.join(marketing_only_hits)}），缺少关系/圈层/礼赠/外显证据，拒绝判定为社会任务。"
            )
        # 关键词命中但证据链不闭合的情况——必须明确拒绝，不得静默降级
        if (gift_hits and not relationship_hits) or (caregiving_hits and not relationship_hits):
            rejection_reasons.append(
                "出现照护/礼赠语义但缺少明确关系对象，拒绝判定为社会任务。"
            )
        if visibility_hits and not (barrier_satisfied and consensus_satisfied):
            missing = []
            if not barrier_satisfied:
                missing.append("获取门槛/稀缺性")
            if not consensus_satisfied:
                missing.append("圈层/审美共识")
            rejection_reasons.append(
                f"高外显证据存在但缺少 {' / '.join(missing)}，三门槛未集齐，拒绝判定为阶层与审美发信。"
            )
        return None, evidence_lines, rejection_reasons

    def _infer_emotional_task(self, payload: DiagnosticInput, module1_output: Module1Output, text: str) -> str | None:
        # P0-2 香氛感官主价值门槛：香水/香氛品类 + 强感官主价值信号(持久留香/前中后调/香调/高级香气/留香持久)
        # 构成「感官价值主诉」，进入自我犒赏与秩序掌控(情绪域)。普通日化留香(如留香珠)不命中香氛品类，不触发。
        if self._is_fragrance_sensory_value(module1_output, text):
            return "自我犒赏与秩序掌控"
        if not self._contains_any(text, {"治愈", "放松", "犒赏", "奖励自己", "氛围感", "仪式感"}):
            return None
        if self._is_ordinary_daily_category(module1_output) and not self._is_high_premium(payload):
            return None
        if self._contains_any(text, {"新奇", "猎奇", "刺激", "尝鲜"}):
            return "新奇探索/瞬时刺激"
        if self._contains_any(text, {"安心", "确定感", "兜底"}):
            return "情绪安心/主观降险"
        return "自我犒赏与秩序掌控"

    def _infer_functional_candidates(self, module1_output: Module1Output, text: str) -> dict[str, Any]:
        trace_tokens = self._collect_trace_tokens(text)
        subcategory_context = self._resolve_household_subcategory_context(module1_output, text)
        functional_facts = self._extract_functional_facts(module1_output, subcategory_context=subcategory_context)
        candidate_pool, veto_trace = self._build_candidate_pool_from_facts(
            module1_output,
            functional_facts,
            text,
            subcategory_context=subcategory_context,
        )

        candidates = [entry["task_name"] for entry in candidate_pool]
        candidate_reasons: dict[str, list[str]] = {
            entry["task_name"]: [entry["mapping_reason"]] for entry in candidate_pool
        }

        if subcategory_context in HOUSEHOLD_STAGEB_SUBCATEGORY_PACKS and not candidate_pool and subcategory_context != "paper_products":
            candidate_pool = self._build_dynamic_household_weak_candidates(module1_output, functional_facts, text)
            if candidate_pool:
                candidates = [entry["task_name"] for entry in candidate_pool]
                candidate_reasons = {
                    entry["task_name"]: [entry["mapping_reason"]] for entry in candidate_pool
                }

        excluded_tasks: dict[str, list[str]] = {}
        default_excluded_reasons = {
            "降本增效/懒人替代": "未发现明确流程负担与简化/替代事实组合。",
            "缺陷修复/冲突消除": "未发现“问题对象 × 已发生异常状态 × 修补/去除动作”的事实组合。",
            "生存/运转维系": "缺少基础供给、日常维持或正常运转型事实。",
        }
        for task_name, default_reason in default_excluded_reasons.items():
            if task_name in candidate_reasons:
                continue
            reasons = [default_reason]
            if task_name == PAPER_ESCALATABLE_TASK and veto_trace:
                veto_desc = [PAPER_HARD_VETO_RULES[veto_id]["desc"] for veto_id in veto_trace if veto_id in PAPER_HARD_VETO_RULES]
                if veto_desc:
                    reasons.append(f"纸品升级被 veto：{'；'.join(veto_desc)}")
            if task_name == "缺陷修复/冲突消除" and trace_tokens:
                reasons.append(f"高歧义 token 仅记入 trace_tokens：{'、'.join(trace_tokens)}。")
            excluded_tasks[task_name] = reasons

        non_selected_task_reasons: list[dict[str, str]] = []
        if not candidates:
            negative_term_veto_hit = False
            if subcategory_context in HOUSEHOLD_STAGEB_SUBCATEGORY_PACKS:
                pack_negative = HOUSEHOLD_STAGEB_SUBCATEGORY_PACKS[subcategory_context].get("negative_terms") or set()
                negative_term_veto_hit = bool(pack_negative and any(term in text for term in pack_negative))
            if negative_term_veto_hit:
                # PRD 6.2.6C Rule 2：negative_terms 命中导致子包强候选被 veto，Stage B 不得直接锁死，
                # 必须广播多候选交 Stage C 仲裁，且 candidate_pool 保持为空。（PRD-1.2：此分支保持不变）
                candidates = ["缺陷修复/冲突消除", "降本增效/懒人替代", "生存/运转维系"]
                veto_reason = "PRD 6.2.6C Rule 2：negative_terms 命中，子包强候选被 veto，转交 Stage C 仲裁。"
                candidate_reasons = {task: [veto_reason] for task in candidates}
                excluded_tasks = {}
                candidate_pool = []
            else:
                has_maintenance_fact = any(
                    (fact.get("problem_object") in FACT_SUPPLY_OBJECTS | FACT_MAINTENANCE_OBJECTS)
                    or fact.get("problem_object") == "口腹/能量补给"
                    for fact in functional_facts
                )
                if has_maintenance_fact or self._supports_maintenance_task(module1_output, text):
                    # PRD-1.2：仅当命中正向 maintenance_supply 强事实时才判“生存/运转维系”。
                    candidates.append("生存/运转维系")
                    candidate_reasons["生存/运转维系"] = ["命中供给/维持事实或常识可支撑维持任务，按 PRD 回退到基础正常运转维系。"]
                    candidate_pool = [
                        {
                            "task_name": "生存/运转维系",
                            "supporting_fact_ids": [],
                            "mapping_reason": "候选事实不足但具备供给/维持类支撑，按 PRD 回退到基础正常运转维系待 Stage C 审核。",
                            "priority": "fallback",
                        }
                    ]
                else:
                    # PRD-1.2 / AC12：无 maintenance 强事实且候选为空 → needs_review 中止态，
                    # 不再默认 append("生存/运转维系") fallback；并记录缺失原因（gate=dropped）。
                    candidates = [NEEDS_REVIEW]
                    candidate_reasons = {NEEDS_REVIEW: ["Stage B 候选为空且无 maintenance 强事实，判定为 needs_review 中止态。"]}
                    candidate_pool = []
                    non_selected_task_reasons.append(
                        {
                            "task": "生存/运转维系",
                            "reason": "Stage B 候选为空，事实不足",
                            "gate": "dropped",
                        }
                    )

        # P0-1 GR-03 例外（洗护个护类「问题状态→修复」闭环收口，泛化词表判定，无样本名硬编码）：
        # subcategory_context 属洗护个护类，且命中强缺陷信号(缺陷状态 token + 修复动作 token 同时成立，见
        # _has_strong_defect_signal)时，不按普通基础清洁(生存/运转维系)处理，收敛到缺陷修复/冲突消除单一候选。
        if (
            subcategory_context in PERSONAL_CARE_STAGEB_SUBCATEGORY_CONTEXTS
            and "缺陷修复/冲突消除" not in candidates
            and self._has_strong_defect_signal(module1_output, text)
        ):
            matched_defect_states = sorted({t for t in DEFECT_STATE_TOKENS if t and t in text})
            gr03_reason = (
                "GR-03 例外：洗护个护类命中已显缺陷状态("
                + "、".join(matched_defect_states)
                + ")并形成「问题状态→修复」闭环，不按基础清洁(生存/运转维系)处理，收敛到缺陷修复/冲突消除。"
            )
            for demoted in list(candidates):
                if demoted == "缺陷修复/冲突消除":
                    continue
                non_selected_task_reasons.append(
                    {
                        "task": demoted,
                        "reason": "GR-03 例外：洗护类存在明确缺陷困扰+改善修复链路，基础维持/清洁让位于缺陷修复/冲突消除。",
                        "gate": "dropped",
                    }
                )
            candidates = ["缺陷修复/冲突消除"]
            candidate_reasons = {"缺陷修复/冲突消除": [gr03_reason]}
            candidate_pool = [
                {
                    "task_name": "缺陷修复/冲突消除",
                    "supporting_fact_ids": [f.get("fact_id") for f in functional_facts if f.get("fact_id")],
                    "mapping_reason": gr03_reason,
                    "priority": "gr03_exception",
                }
            ]
            excluded_tasks.pop("缺陷修复/冲突消除", None)

        return {
            "candidate_tasks": list(dict.fromkeys(candidates)),
            "candidate_reasons": candidate_reasons,
            "excluded_tasks": excluded_tasks,
            "trace_tokens": trace_tokens,
            "functional_facts": functional_facts,
            "candidate_pool": candidate_pool,
            "subcategory_context": subcategory_context or "general_functional",
            "veto_trace": veto_trace,
            "non_selected_task_reasons": non_selected_task_reasons,
        }

    def _collect_trace_tokens(self, text: str) -> list[str]:
        return [token for token in sorted(HIGH_RISK_SINGLE_CHAR_TOKENS) if token in text]

    def _has_any_token_match(self, text: str, tokens: Mapping[str, Any] | set[str]) -> bool:
        iterable = tokens.keys() if isinstance(tokens, Mapping) else tokens
        return any(token in text for token in sorted(iterable, key=len, reverse=True))

    def _extract_keyword_matches(self, text: str, token_map: Mapping[str, str]) -> list[str]:
        matches: list[str] = []
        seen: set[str] = set()
        for token, normalized in sorted(token_map.items(), key=lambda item: len(item[0]), reverse=True):
            if token in text and normalized not in seen:
                seen.add(normalized)
                matches.append(normalized)
        return matches

    def _resolve_household_subcategory_context(self, module1_output: Module1Output, text: str) -> str:
        category_text = "｜".join(
            value
            for value in [
                module1_output.second_level_category,
                module1_output.third_level_category,
                module1_output.leaf_category,
            ]
            if value
        )
        if not category_text:
            return ""

        for subcategory_context, pack in [
            ("paper_products", HOUSEHOLD_STAGEB_SUBCATEGORY_PACKS["paper_products"]),
            ("laundry_cleaning", HOUSEHOLD_STAGEB_SUBCATEGORY_PACKS["laundry_cleaning"]),
            ("deodorization", HOUSEHOLD_STAGEB_SUBCATEGORY_PACKS["deodorization"]),
            ("appliance_cleaning", HOUSEHOLD_STAGEB_SUBCATEGORY_PACKS["appliance_cleaning"]),
            ("family_env_cleaning", HOUSEHOLD_STAGEB_SUBCATEGORY_PACKS["family_env_cleaning"]),
            ("skincare_repair", HOUSEHOLD_STAGEB_SUBCATEGORY_PACKS["skincare_repair"]),
            ("cleanse_protection", HOUSEHOLD_STAGEB_SUBCATEGORY_PACKS["cleanse_protection"]),
            ("hair_scalp_care", HOUSEHOLD_STAGEB_SUBCATEGORY_PACKS["hair_scalp_care"]),
            ("oral_care", HOUSEHOLD_STAGEB_SUBCATEGORY_PACKS["oral_care"]),
            ("hair_removal_tools", HOUSEHOLD_STAGEB_SUBCATEGORY_PACKS["hair_removal_tools"]),
        ]:
            if self._has_any_token_match(category_text, set(pack.get("category_terms", set()))):
                return subcategory_context

        if "家庭环境清洁" in category_text:
            strongest_context = ""
            strongest_score = 0
            for subcategory_context, pack in HOUSEHOLD_STAGEB_SUBCATEGORY_PACKS.items():
                score = len(self._extract_keyword_matches(text, pack.get("problem_object_terms", {})))
                if score > strongest_score:
                    strongest_context = subcategory_context
                    strongest_score = score
            if strongest_context:
                return strongest_context
        return ""

    def _extract_functional_facts(self, module1_output: Module1Output, *, subcategory_context: str = "") -> list[dict[str, Any]]:
        clauses: list[str] = []
        for part in [
            module1_output.core_selling_point,
            module1_output.differentiator.conclusion,
            module1_output.differentiator.summary,
            module1_output.product_name,
        ]:
            if str(part or "").strip():
                clauses.append(str(part).strip())
        for item in module1_output.differentiator.evidence_chain:
            text = str(item.evidence_text or "").strip()
            if text:
                clauses.append(text)

        facts: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str, str, str, str]] = set()
        for clause in clauses:
            clause_facts = self._extract_clause_facts(clause, module1_output, subcategory_context=subcategory_context)
            for fact in clause_facts:
                key = (
                    fact.get("fact_layer", ""),
                    fact.get("subcategory_context", ""),
                    fact.get("problem_object", ""),
                    fact.get("problem_state", ""),
                    fact.get("action_mechanism", ""),
                    fact.get("evidence_text", ""),
                )
                if key in seen:
                    continue
                seen.add(key)
                fact["fact_id"] = f"fact_{len(facts) + 1:03d}"
                fact.setdefault("source_type", "title_or_selling_point")
                fact.setdefault("confidence", "high")
                facts.append(fact)
        return facts

    def _extract_clause_facts(
        self,
        clause: str,
        module1_output: Module1Output,
        *,
        subcategory_context: str = "",
    ) -> list[dict[str, Any]]:
        clause = str(clause or "").strip()
        if not clause:
            return []
        if subcategory_context in HOUSEHOLD_STAGEB_SUBCATEGORY_PACKS:
            pack = HOUSEHOLD_STAGEB_SUBCATEGORY_PACKS[subcategory_context]
            negative_terms = pack.get("negative_terms") or set()
            if negative_terms and any(term in clause for term in negative_terms):
                # PRD 6.2.6C Rule 2：family_env_cleaning 等子包的 negative_terms 命中即硬性 veto，
                # 子句不得形成 subcategory_pack 强候选或 common_skeleton 弱候选事实，统一下沉到 Stage C 仲裁。
                return []
            common_facts = self._extract_common_functional_facts(clause, module1_output, subcategory_context=subcategory_context)
            subcategory_facts = self._extract_subcategory_facts(subcategory_context, clause, module1_output)
            return common_facts + subcategory_facts
        return self._extract_general_clause_facts(clause, module1_output)

    def _extract_common_functional_facts(
        self,
        clause: str,
        module1_output: Module1Output,
        *,
        subcategory_context: str,
    ) -> list[dict[str, Any]]:
        if subcategory_context in PERSONAL_CARE_STAGEB_SUBCATEGORY_CONTEXTS:
            object_matches = self._extract_keyword_matches(clause, PERSONAL_CARE_COMMON_OBJECT_TOKENS)
            state_matches = self._extract_keyword_matches(clause, PERSONAL_CARE_COMMON_STATE_TOKENS)
            action_matches = self._extract_keyword_matches(clause, PERSONAL_CARE_COMMON_ACTION_TOKENS)
            source_type = "personalcare_common_skeleton"
        else:
            object_matches = self._extract_keyword_matches(clause, HOUSEHOLD_COMMON_OBJECT_TOKENS)
            state_matches = self._extract_keyword_matches(clause, HOUSEHOLD_COMMON_STATE_TOKENS)
            action_matches = self._extract_keyword_matches(clause, HOUSEHOLD_COMMON_ACTION_TOKENS)
            source_type = "household_common_skeleton"
        if not (object_matches or state_matches or action_matches):
            return []
        return [
            {
                "problem_object": object_matches[0] if object_matches else "",
                "problem_state": state_matches[0] if state_matches else "",
                "action_mechanism": action_matches[0] if action_matches else "",
                "benefit_target": module1_output.target_people or module1_output.leaf_category,
                "usage_scene": module1_output.leaf_category,
                "evidence_text": clause,
                "fact_layer": "common_skeleton",
                "subcategory_context": subcategory_context,
                "source_type": source_type,
                "confidence": "medium",
                "matched_terms": {
                    "object_terms": object_matches,
                    "state_terms": state_matches,
                    "action_terms": action_matches,
                },
            }
        ]

    def _extract_subcategory_facts(
        self,
        subcategory_context: str,
        clause: str,
        module1_output: Module1Output,
    ) -> list[dict[str, Any]]:
        pack = HOUSEHOLD_STAGEB_SUBCATEGORY_PACKS.get(subcategory_context)
        if not pack:
            return []

        fact_groups = list(pack.get("fact_groups") or [])
        if not fact_groups:
            fact_groups = [pack]

        facts: list[dict[str, Any]] = []
        for group in fact_groups:
            fact = self._build_subcategory_fact_from_group(subcategory_context, clause, module1_output, pack, group)
            if fact:
                facts.append(fact)
        return facts

    def _build_subcategory_fact_from_group(
        self,
        subcategory_context: str,
        clause: str,
        module1_output: Module1Output,
        pack: Mapping[str, Any],
        group: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        object_matches = self._extract_keyword_matches(clause, group.get("problem_object_terms") or pack.get("problem_object_terms", {}))
        state_matches = self._extract_keyword_matches(clause, group.get("problem_state_terms") or pack.get("problem_state_terms", {}))
        action_matches = self._extract_keyword_matches(clause, group.get("action_mechanism_terms") or pack.get("action_mechanism_terms", {}))
        default_object = str(group.get("default_object") or pack.get("default_object") or "").strip()
        allow_action_only_with_object = bool(group.get("allow_action_only_with_object", pack.get("allow_action_only_with_object")))
        allow_action_only_without_object = bool(group.get("allow_action_only_without_object", pack.get("allow_action_only_without_object")))
        default_state_when_action_only = str(group.get("default_state_when_action_only") or pack.get("default_state_when_action_only") or "").strip()
        action_only_terms = set(group.get("action_only_terms") or pack.get("action_only_terms", set()))

        if not object_matches and default_object and state_matches and action_matches:
            object_matches = [default_object]
        if (
            not object_matches
            and default_object
            and not state_matches
            and action_matches
            and allow_action_only_without_object
            and (not action_only_terms or any(action in action_only_terms for action in action_matches))
        ):
            object_matches = [default_object]
            state_matches = [default_state_when_action_only or "维持正常"]
        if (
            not state_matches
            and object_matches
            and action_matches
            and allow_action_only_with_object
            and (not action_only_terms or any(action in action_only_terms for action in action_matches))
        ):
            state_matches = [default_state_when_action_only or "维持正常"]

        if not (object_matches and state_matches and action_matches):
            return None

        return {
            "problem_object": object_matches[0],
            "problem_state": state_matches[0],
            "action_mechanism": action_matches[0],
            "benefit_target": object_matches[0],
            "usage_scene": module1_output.leaf_category,
            "evidence_text": clause,
            "fact_layer": "subcategory_pack",
            "subcategory_context": subcategory_context,
            "source_type": f"household_subcategory_pack:{subcategory_context}:{group.get('group_name', 'default')}",
            "confidence": "high",
            "matched_terms": {
                "object_terms": object_matches,
                "state_terms": state_matches,
                "action_terms": action_matches,
            },
            "group_name": str(group.get("group_name") or "default"),
            "group_candidate_task": str(group.get("group_candidate_task") or "").strip(),
        }

    def _extract_general_clause_facts(self, clause: str, module1_output: Module1Output) -> list[dict[str, Any]]:
        facts: list[dict[str, Any]] = []
        lower_clause = clause
        diff_type = module1_output.differentiator.difference_type

        defect_object = self._match_fact_value(lower_clause, FACT_OBJECT_TOKENS, FACT_DEFECT_OBJECTS)
        abnormal_state = self._match_fact_value(lower_clause, FACT_ABNORMAL_STATE_TOKENS)
        remediation_action = self._match_fact_value(lower_clause, FACT_REMEDIATION_ACTION_TOKENS)
        if defect_object and abnormal_state and remediation_action:
            facts.append(
                {
                    "problem_object": defect_object,
                    "problem_state": abnormal_state,
                    "action_mechanism": remediation_action,
                    "benefit_target": defect_object,
                    "usage_scene": module1_output.leaf_category,
                    "evidence_text": clause,
                    "fact_layer": "general_fact",
                    "subcategory_context": "general_functional",
                }
            )

        process_object = self._match_fact_value(lower_clause, FACT_OBJECT_TOKENS, FACT_PROCESS_OBJECTS)
        process_state = self._match_fact_value(lower_clause, FACT_PROCESS_STATE_TOKENS)
        efficiency_action = self._match_fact_value(lower_clause, FACT_EFFICIENCY_ACTION_TOKENS)
        if process_object and (process_state or diff_type in {"步骤压缩", "新形态替代", "成本优化"}) and efficiency_action:
            facts.append(
                {
                    "problem_object": process_object,
                    "problem_state": process_state or "流程负担较高",
                    "action_mechanism": efficiency_action,
                    "benefit_target": process_object,
                    "usage_scene": module1_output.leaf_category,
                    "evidence_text": clause,
                    "fact_layer": "general_fact",
                    "subcategory_context": "general_functional",
                }
            )

        supply_object = self._match_fact_value(lower_clause, FACT_OBJECT_TOKENS, FACT_SUPPLY_OBJECTS | FACT_MAINTENANCE_OBJECTS)
        maintenance_state = self._match_fact_value(lower_clause, FACT_MAINTENANCE_STATE_TOKENS)
        maintenance_action = self._match_fact_value(lower_clause, FACT_MAINTENANCE_ACTION_TOKENS)
        if supply_object and (maintenance_state or maintenance_action or self._is_food_like_category(module1_output.leaf_category)):
            facts.append(
                {
                    "problem_object": supply_object,
                    "problem_state": maintenance_state or "维持正常",
                    "action_mechanism": maintenance_action or "维持",
                    "benefit_target": supply_object,
                    "usage_scene": module1_output.leaf_category,
                    "evidence_text": clause,
                    "fact_layer": "general_fact",
                    "subcategory_context": "general_functional",
                }
            )

        if not facts and self._is_food_like_category(module1_output.leaf_category) and self._contains_any(clause, PREFERENCE_ONLY_TOKENS | MAINTENANCE_SUPPLY_TOKENS):
            facts.append(
                {
                    "problem_object": "口腹/能量补给",
                    "problem_state": "日常补给",
                    "action_mechanism": "补充",
                    "benefit_target": module1_output.target_people,
                    "usage_scene": module1_output.leaf_category,
                    "evidence_text": clause,
                    "fact_layer": "general_fact",
                    "subcategory_context": "general_functional",
                }
            )
        return facts

    def _match_fact_value(self, text: str, token_map: dict[str, str], allowed_values: set[str] | None = None) -> str:
        for token, normalized in sorted(token_map.items(), key=lambda item: len(item[0]), reverse=True):
            if token in text and (allowed_values is None or normalized in allowed_values):
                return normalized
        return ""

    def _build_candidate_pool_from_facts(
        self,
        module1_output: Module1Output,
        facts: list[dict[str, Any]],
        text: str,
        *,
        subcategory_context: str = "",
    ) -> tuple[list[dict[str, Any]], list[str]]:
        if subcategory_context in HOUSEHOLD_STAGEB_SUBCATEGORY_PACKS:
            return self._build_household_candidate_pool_by_pack(subcategory_context, facts, module1_output, text)
        return self._build_general_candidate_pool_from_facts(module1_output, facts, text), []

    def _build_general_candidate_pool_from_facts(
        self,
        module1_output: Module1Output,
        facts: list[dict[str, Any]],
        text: str,
    ) -> list[dict[str, Any]]:
        candidate_pool: list[dict[str, Any]] = []

        defect_fact_ids = [
            fact["fact_id"] for fact in facts
            if fact.get("problem_object") in FACT_DEFECT_OBJECTS
            and any(keyword in fact.get("problem_state", "") for keyword in {"附着", "发黄", "困扰", "堵塞", "疼痛", "残留", "异常"})
            and fact.get("action_mechanism") in {"去除", "改善", "修复", "缓解", "消除"}
        ]
        if defect_fact_ids:
            candidate_pool.append(
                {
                    "task_name": "缺陷修复/冲突消除",
                    "supporting_fact_ids": defect_fact_ids,
                    "mapping_reason": "问题对象为已出现的缺陷/残留/不适，且存在异常状态与修补/去除动作的事实组合。",
                    "priority": "normal",
                }
            )

        efficiency_fact_ids = [
            fact["fact_id"] for fact in facts
            if fact.get("problem_object") in FACT_PROCESS_OBJECTS and fact.get("action_mechanism") in {"简化", "替代", "压缩", "提速"}
        ]
        if (
            efficiency_fact_ids
            or module1_output.differentiator.difference_type in {"步骤压缩", "新形态替代", "成本优化"}
            or self._contains_any(text, OPERATION_EASE_TOKENS)
        ):
            candidate_pool.append(
                {
                    "task_name": "降本增效/懒人替代",
                    "supporting_fact_ids": efficiency_fact_ids,
                    "mapping_reason": "问题对象指向流程/步骤/时间成本，或存在明确操作门槛下降信号，且卖点动作是简化、替代、压缩、提速或更易上手。",
                    "priority": "normal",
                }
            )

        maintenance_fact_ids = [
            fact["fact_id"] for fact in facts
            if fact.get("problem_object") in FACT_SUPPLY_OBJECTS | FACT_MAINTENANCE_OBJECTS or fact.get("problem_object") == "口腹/能量补给"
        ]
        if maintenance_fact_ids or self._supports_maintenance_task(module1_output, text):
            candidate_pool.append(
                {
                    "task_name": "生存/运转维系",
                    "supporting_fact_ids": maintenance_fact_ids,
                    "mapping_reason": "商品主要承接基础补给、日常维持或正常运转维系，且不以修补已发生问题为前提。",
                    "priority": "normal",
                }
            )

        deduped: list[dict[str, Any]] = []
        seen_tasks: set[str] = set()
        for entry in candidate_pool:
            task_name = entry["task_name"]
            if task_name in seen_tasks:
                continue
            seen_tasks.add(task_name)
            deduped.append(entry)
        return deduped

    def _build_household_candidate_pool_by_pack(
        self,
        subcategory_context: str,
        facts: list[dict[str, Any]],
        module1_output: Module1Output,
        text: str,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        # PRD 6.2.6C Rule 2 后置断言：若有 subcategory_pack 强候选事实其 evidence_text 命中 negative_terms，
        # 说明上游 veto 失效，必须 Crash Early 阻断输出，绝不允许污染候选池。
        pack_def = HOUSEHOLD_STAGEB_SUBCATEGORY_PACKS.get(subcategory_context, {})
        pack_negative_terms = pack_def.get("negative_terms") or set()
        if pack_negative_terms:
            for fact in facts:
                if fact.get("fact_layer") != "subcategory_pack":
                    continue
                evidence_text = str(fact.get("evidence_text") or "")
                hit = [t for t in pack_negative_terms if t in evidence_text]
                if hit:
                    raise AssertionError(
                        f"PRD 6.2.6C Rule 2 violation: subcategory_pack 强候选事实 {fact.get('fact_id')} "
                        f"在 {subcategory_context} 包内命中 negative_terms={hit}，evidence_text={evidence_text!r}"
                    )
        sub_pack_facts = [fact for fact in facts if fact.get("fact_layer") == "subcategory_pack"]
        if subcategory_context == "paper_products":
            return self._build_paper_candidate_pool(module1_output, text, sub_pack_facts)
        if subcategory_context == "laundry_cleaning":
            return self._build_laundry_candidate_pool(sub_pack_facts)
        if subcategory_context == "appliance_cleaning":
            return self._build_appliance_candidate_pool(sub_pack_facts)
        if subcategory_context == "skincare_repair":
            return self._build_skincare_candidate_pool(sub_pack_facts)
        if subcategory_context == "cleanse_protection":
            return self._build_cleanse_protection_candidate_pool(sub_pack_facts)
        if subcategory_context == "hair_scalp_care":
            return self._build_hair_scalp_candidate_pool(sub_pack_facts)
        if subcategory_context == "oral_care":
            return self._build_oral_candidate_pool(sub_pack_facts)
        if subcategory_context == "hair_removal_tools":
            return self._build_hair_removal_candidate_pool(sub_pack_facts)

        pack = HOUSEHOLD_STAGEB_SUBCATEGORY_PACKS[subcategory_context]
        candidate_pool: list[dict[str, Any]] = []

        default_task = str(pack.get("candidate_task") or "缺陷修复/冲突消除")
        strong_fact_ids = [
            fact["fact_id"]
            for fact in sub_pack_facts
            if (fact.get("group_candidate_task") or default_task) == default_task
        ]
        if strong_fact_ids:
            candidate_pool.append(
                {
                    "task_name": default_task,
                    "supporting_fact_ids": strong_fact_ids,
                    "mapping_reason": f"{subcategory_context} 子包已形成对象×状态×动作闭环，由子包事实直接承接强候选。",
                    "priority": "sub_pack_strong",
                }
            )

        efficiency_fact_ids = [
            fact["fact_id"]
            for fact in facts
            if (
                fact.get("group_candidate_task") == "降本增效/懒人替代"
                or (
                    fact.get("problem_object") in FACT_PROCESS_OBJECTS
                    and fact.get("action_mechanism") in {"简化", "替代", "压缩", "提速"}
                )
            )
        ]
        if efficiency_fact_ids:
            candidate_pool.append(
                {
                    "task_name": "降本增效/懒人替代",
                    "supporting_fact_ids": efficiency_fact_ids,
                    "mapping_reason": f"{subcategory_context} 子包命中流程负担与简化/替代动作，按子包强候选 ∪ 通用效率候选取并集输出。",
                    "priority": "sub_pack_efficiency_union",
                }
            )

        if candidate_pool:
            return self._dedupe_candidate_pool(candidate_pool), []
        return [], []

    def _dedupe_candidate_pool(self, candidate_pool: list[dict[str, Any]]) -> list[dict[str, Any]]:
        deduped: list[dict[str, Any]] = []
        seen_tasks: set[str] = set()
        for entry in candidate_pool:
            task_name = entry["task_name"]
            if task_name in seen_tasks:
                continue
            seen_tasks.add(task_name)
            deduped.append(entry)
        return deduped

    def _build_dynamic_household_weak_candidates(
        self,
        module1_output: Module1Output,
        facts: list[dict[str, Any]],
        text: str,
    ) -> list[dict[str, Any]]:
        weak_reason = "仅命中 Common Skeleton 弱事实或类目壳信息，Stage B 先保留可被事实支撑的弱候选，交由 Stage C 继续仲裁。"
        candidate_pool: list[dict[str, Any]] = []

        weak_defect_fact_ids = [
            fact["fact_id"]
            for fact in facts
            if fact.get("fact_layer") != "subcategory_pack"
            and (
                any(keyword in str(fact.get("problem_state") or "") for keyword in {"残留", "困扰", "异常", "附着", "发黄", "堵塞", "不适"})
                or fact.get("action_mechanism") in {"去除", "溶解", "改善", "修复", "缓解", "消除"}
            )
        ]
        if weak_defect_fact_ids:
            candidate_pool.append(
                {
                    "task_name": "缺陷修复/冲突消除",
                    "supporting_fact_ids": weak_defect_fact_ids,
                    "mapping_reason": weak_reason,
                    "priority": "sub_pack_weak",
                }
            )

        weak_efficiency_fact_ids = [
            fact["fact_id"]
            for fact in facts
            if fact.get("fact_layer") != "subcategory_pack"
            and fact.get("action_mechanism") in {"简化", "替代", "压缩", "提速"}
        ]
        if (
            weak_efficiency_fact_ids
            or module1_output.differentiator.difference_type in {"步骤压缩", "新形态替代", "成本优化"}
            or self._contains_any(text, OPERATION_EASE_TOKENS)
        ):
            candidate_pool.append(
                {
                    "task_name": "降本增效/懒人替代",
                    "supporting_fact_ids": weak_efficiency_fact_ids,
                    "mapping_reason": weak_reason,
                    "priority": "sub_pack_weak",
                }
            )

        maintenance_fact_ids = [
            fact["fact_id"]
            for fact in facts
            if fact.get("fact_layer") != "subcategory_pack"
            and (
                fact.get("problem_object") in FACT_SUPPLY_OBJECTS | FACT_MAINTENANCE_OBJECTS
                or fact.get("problem_object") == "口腹/能量补给"
            )
        ]
        if maintenance_fact_ids:
            candidate_pool.append(
                {
                    "task_name": "生存/运转维系",
                    "supporting_fact_ids": maintenance_fact_ids,
                    "mapping_reason": weak_reason,
                    "priority": "sub_pack_weak",
                }
            )
        return candidate_pool

    def _build_laundry_candidate_pool(
        self,
        sub_pack_facts: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[str]]:
        if not sub_pack_facts:
            return [], []

        maintenance_fact_ids = [
            fact["fact_id"]
            for fact in sub_pack_facts
            if fact.get("problem_state") == "维持正常" or fact.get("action_mechanism") == "改善"
        ]
        defect_fact_ids = [
            fact["fact_id"]
            for fact in sub_pack_facts
            if fact.get("problem_state") != "维持正常" and (
                fact.get("action_mechanism") in {"去除", "修复"}
                or "困扰" in str(fact.get("problem_state") or "")
                or "残留" in str(fact.get("problem_state") or "")
                or "异常" in str(fact.get("problem_state") or "")
            )
        ]

        candidate_pool: list[dict[str, Any]] = []
        if maintenance_fact_ids:
            candidate_pool.append(
                {
                    "task_name": "生存/运转维系",
                    "supporting_fact_ids": maintenance_fact_ids,
                    "mapping_reason": "衣物护理增效组以柔顺、留香、护色、日常护理维持为主，默认承接生存/运转维系。",
                    "priority": "laundry_care_maintenance",
                }
            )
        if defect_fact_ids:
            candidate_pool.append(
                {
                    "task_name": "缺陷修复/冲突消除",
                    "supporting_fact_ids": defect_fact_ids,
                    "mapping_reason": "衣物样本已出现汗味、霉味、污渍等异常状态，并存在去除/修复动作，可切入缺陷修复/冲突消除。",
                    "priority": "laundry_defect_repair",
                }
            )
        return candidate_pool, []

    def _build_appliance_candidate_pool(
        self,
        sub_pack_facts: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[str]]:
        if not sub_pack_facts:
            return [], []

        maintenance_fact_ids = [
            fact["fact_id"]
            for fact in sub_pack_facts
            if fact.get("problem_state") == "维持正常"
        ]
        defect_fact_ids = [
            fact["fact_id"]
            for fact in sub_pack_facts
            if fact.get("problem_state") != "维持正常"
        ]

        candidate_pool: list[dict[str, Any]] = []
        if maintenance_fact_ids:
            candidate_pool.append(
                {
                    "task_name": "生存/运转维系",
                    "supporting_fact_ids": maintenance_fact_ids,
                    "mapping_reason": "电器清洁样本命中明确电器对象，且表达为免拆、长效或日常维护，优先承接生存/运转维系。",
                    "priority": "appliance_maintenance",
                }
            )
        if defect_fact_ids:
            candidate_pool.append(
                {
                    "task_name": "缺陷修复/冲突消除",
                    "supporting_fact_ids": defect_fact_ids,
                    "mapping_reason": "电器清洁样本已形成对象×异常状态×清洁动作闭环，承接缺陷修复/冲突消除。",
                    "priority": "appliance_defect_repair",
                }
            )
        return candidate_pool, []

    def _is_skincare_quasi_repair_fact(self, fact: Mapping[str, Any]) -> bool:
        problem_object = str(fact.get("problem_object") or "").strip()
        problem_state = str(fact.get("problem_state") or "").strip()
        action_mechanism = str(fact.get("action_mechanism") or "").strip()
        evidence_text = str(fact.get("evidence_text") or "")
        if not problem_object:
            return False
        if problem_state and problem_state != "维持正常":
            return True
        if action_mechanism not in {"修护", "去除"}:
            return False
        if self._contains_any(evidence_text, {"维稳", "保湿", "舒缓"}):
            return False
        return self._contains_any(evidence_text, {"祛痘", "抗皱", "淡纹", "净颜", "修护", "修复"})

    def _is_skincare_maintenance_fact(self, fact: Mapping[str, Any]) -> bool:
        problem_object = str(fact.get("problem_object") or "").strip()
        problem_state = str(fact.get("problem_state") or "").strip()
        action_mechanism = str(fact.get("action_mechanism") or "").strip()
        evidence_text = str(fact.get("evidence_text") or "")
        if not problem_object or problem_state != "维持正常":
            return False
        if action_mechanism == "维持":
            return True
        if self._contains_any(evidence_text, {"维稳", "保湿", "舒缓"}):
            return True
        return action_mechanism in {"修护", "去除"} and not self._is_skincare_quasi_repair_fact(fact)

    def _build_skincare_candidate_pool(
        self,
        sub_pack_facts: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[str]]:
        if not sub_pack_facts:
            return [], []

        maintenance_fact_ids: list[str] = []
        defect_fact_ids: list[str] = []
        for fact in sub_pack_facts:
            fact_id = str(fact.get("fact_id") or "").strip()
            if not fact_id:
                continue
            if self._is_skincare_quasi_repair_fact(fact):
                defect_fact_ids.append(fact_id)
                continue
            if self._is_skincare_maintenance_fact(fact):
                maintenance_fact_ids.append(fact_id)

        candidate_pool: list[dict[str, Any]] = []
        if maintenance_fact_ids:
            candidate_pool.append(
                {
                    "task_name": "生存/运转维系",
                    "supporting_fact_ids": maintenance_fact_ids,
                    "mapping_reason": "护肤功效修护包命中保湿、维稳、舒缓等维持型事实，或仅形成“部位词 + 维持导向功效词”的准闭环，优先承接生存/运转维系。",
                    "priority": "skincare_maintenance",
                }
            )
        if defect_fact_ids:
            candidate_pool.append(
                {
                    "task_name": "缺陷修复/冲突消除",
                    "supporting_fact_ids": defect_fact_ids,
                    "mapping_reason": "护肤功效修护包已形成明确问题状态闭环，或命中“部位词 + 问题导向功效词”的准闭环，承接缺陷修复/冲突消除。",
                    "priority": "skincare_defect_repair",
                }
            )
        return candidate_pool, []

    def _build_cleanse_protection_candidate_pool(
        self,
        sub_pack_facts: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[str]]:
        if not sub_pack_facts:
            return [], []

        maintenance_fact_ids = [
            fact["fact_id"]
            for fact in sub_pack_facts
            if fact.get("problem_state") == "维持正常"
        ]
        defect_fact_ids = [
            fact["fact_id"]
            for fact in sub_pack_facts
            if fact.get("problem_state") != "维持正常"
        ]

        candidate_pool: list[dict[str, Any]] = []
        if maintenance_fact_ids:
            candidate_pool.append(
                {
                    "task_name": "生存/运转维系",
                    "supporting_fact_ids": maintenance_fact_ids,
                    "mapping_reason": "清洁卸净与基础防护包中的日常卸净、防护维持型事实，优先承接生存/运转维系。",
                    "priority": "cleanse_protection_maintenance",
                }
            )
        if defect_fact_ids:
            candidate_pool.append(
                {
                    "task_name": "缺陷修复/冲突消除",
                    "supporting_fact_ids": defect_fact_ids,
                    "mapping_reason": "清洁卸净与基础防护包已命中残留负担或晒伤/晒黑风险，并存在卸净或防护动作，承接缺陷修复/冲突消除。",
                    "priority": "cleanse_protection_defect_repair",
                }
            )
        return candidate_pool, []

    def _build_hair_scalp_candidate_pool(
        self,
        sub_pack_facts: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[str]]:
        if not sub_pack_facts:
            return [], []

        maintenance_fact_ids = [
            fact["fact_id"]
            for fact in sub_pack_facts
            if fact.get("problem_state") == "维持正常" or fact.get("action_mechanism") == "维持"
        ]
        defect_fact_ids = [
            fact["fact_id"]
            for fact in sub_pack_facts
            if fact.get("problem_state") != "维持正常" and fact.get("action_mechanism") in {"去除", "修护"}
        ]

        candidate_pool: list[dict[str, Any]] = []
        if maintenance_fact_ids:
            candidate_pool.append(
                {
                    "task_name": "生存/运转维系",
                    "supporting_fact_ids": maintenance_fact_ids,
                    "mapping_reason": "头皮发丝护理包中的日常清洁、蓬松、顺滑维持型事实，优先承接生存/运转维系。",
                    "priority": "hair_scalp_maintenance",
                }
            )
        if defect_fact_ids:
            candidate_pool.append(
                {
                    "task_name": "缺陷修复/冲突消除",
                    "supporting_fact_ids": defect_fact_ids,
                    "mapping_reason": "头皮发丝护理包已命中头屑、出油、毛躁、干枯等异常状态，并存在去屑、控油、修护动作，承接缺陷修复/冲突消除。",
                    "priority": "hair_scalp_defect_repair",
                }
            )
        return candidate_pool, []

    def _build_oral_candidate_pool(
        self,
        sub_pack_facts: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[str]]:
        if not sub_pack_facts:
            return [], []

        maintenance_fact_ids = [
            fact["fact_id"]
            for fact in sub_pack_facts
            if fact.get("problem_state") == "维持正常" or fact.get("action_mechanism") == "维持"
        ]
        defect_fact_ids = [
            fact["fact_id"]
            for fact in sub_pack_facts
            if fact.get("problem_state") != "维持正常" and fact.get("action_mechanism") == "去除"
        ]

        candidate_pool: list[dict[str, Any]] = []
        if maintenance_fact_ids:
            candidate_pool.append(
                {
                    "task_name": "生存/运转维系",
                    "supporting_fact_ids": maintenance_fact_ids,
                    "mapping_reason": "口腔护理包中的日常清洁维护型事实，优先承接生存/运转维系。",
                    "priority": "oral_maintenance",
                }
            )
        if defect_fact_ids:
            candidate_pool.append(
                {
                    "task_name": "缺陷修复/冲突消除",
                    "supporting_fact_ids": defect_fact_ids,
                    "mapping_reason": "口腔护理包已命中口气、牙渍、牙黄、残留等异常状态，并存在去渍/美白/清除动作，承接缺陷修复/冲突消除。",
                    "priority": "oral_defect_repair",
                }
            )
        return candidate_pool, []

    def _build_hair_removal_candidate_pool(
        self,
        sub_pack_facts: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[str]]:
        if not sub_pack_facts:
            return [], []

        efficiency_fact_ids = [
            fact["fact_id"]
            for fact in sub_pack_facts
            if fact.get("problem_state") == "流程负担" or fact.get("action_mechanism") == "简化"
        ]
        safety_fact_ids = [
            fact["fact_id"]
            for fact in sub_pack_facts
            if "风险" in str(fact.get("problem_state") or "") and fact.get("action_mechanism") == "防护"
        ]

        candidate_pool: list[dict[str, Any]] = []
        if efficiency_fact_ids:
            candidate_pool.append(
                {
                    "task_name": "降本增效/懒人替代",
                    "supporting_fact_ids": efficiency_fact_ids,
                    "mapping_reason": "毛发管理工具包已命中新手门槛、流程费劲与快速处理/一推即净等动作，承接降本增效/懒人替代。",
                    "priority": "hair_removal_efficiency",
                }
            )
        if safety_fact_ids:
            candidate_pool.append(
                {
                    "task_name": "物理安全与风险规避",
                    "supporting_fact_ids": safety_fact_ids,
                    "mapping_reason": "毛发管理工具包已出现刮伤、刺痛等风险状态，并存在防护动作，承接物理安全与风险规避。",
                    "priority": "hair_removal_safety",
                }
            )
        return candidate_pool, []

    def _build_paper_candidate_pool(
        self,
        module1_output: Module1Output,
        text: str,
        sub_pack_facts: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[str]]:
        candidate_pool = [
            {
                "task_name": PAPER_DEFAULT_TASK,
                "supporting_fact_ids": [],
                "mapping_reason": "纸品默认先进入生存/运转维系，只有在异常对象×问题状态×修复动作闭环成立时才允许升级。",
                "priority": "paper_default",
            }
        ]
        veto_trace = self._apply_paper_route_veto_rules(module1_output, text, sub_pack_facts)
        if not veto_trace and self._paper_upgrade_has_closed_loop(sub_pack_facts):
            candidate_pool.append(
                {
                    "task_name": PAPER_ESCALATABLE_TASK,
                    "supporting_fact_ids": [fact["fact_id"] for fact in sub_pack_facts],
                    "mapping_reason": "纸品样本存在明确异常对象、问题状态与修复动作闭环，允许追加修复候选。",
                    "priority": "paper_upgrade",
                }
            )
        return candidate_pool, veto_trace

    def _apply_paper_route_veto_rules(
        self,
        module1_output: Module1Output,
        text: str,
        sub_pack_facts: list[dict[str, Any]],
    ) -> list[str]:
        fact_has_object = any(str(fact.get("problem_object") or "").strip() for fact in sub_pack_facts)
        fact_has_state = any(str(fact.get("problem_state") or "").strip() for fact in sub_pack_facts)
        fact_has_action = any(str(fact.get("action_mechanism") or "").strip() for fact in sub_pack_facts)
        paper_text = "｜".join(
            value
            for value in [
                module1_output.third_level_category,
                module1_output.product_name,
                module1_output.core_selling_point,
                module1_output.differentiator.conclusion,
                module1_output.differentiator.summary,
            ]
            if value
        )
        material_signal = self._contains_any(paper_text, PAPER_HARD_VETO_RULES["paper_material_only"]["match_terms"])
        cleaner_signal = self._contains_any(paper_text, PAPER_HARD_VETO_RULES["paper_cleaner_substitution"]["match_terms"])
        toilet_scene_signal = self._contains_any(
            "｜".join(
                value
                for value in [
                    module1_output.third_level_category,
                    module1_output.product_name,
                    module1_output.target_people,
                    module1_output.core_selling_point,
                ]
                if value
            ),
            {"厕间", "湿厕纸", "如厕", "厕纸"},
        )

        if toilet_scene_signal and not fact_has_action:
            return ["paper_no_remediation_action"]
        if material_signal and not self._contains_any(paper_text, {"一擦", "即净", "去污", "除味", "清洁", "厨房"}) and not (fact_has_object or fact_has_state or fact_has_action) and not cleaner_signal:
            return ["paper_material_only"]
        if cleaner_signal and not self._paper_upgrade_has_closed_loop(sub_pack_facts):
            return ["paper_cleaner_substitution"]
        if not fact_has_object:
            return ["paper_no_problem_object"]
        if not fact_has_action:
            return ["paper_no_remediation_action"]
        return []

    def _paper_upgrade_has_closed_loop(self, facts: list[dict[str, Any]]) -> bool:
        return any(
            fact.get("problem_object") and fact.get("problem_state") and fact.get("action_mechanism")
            for fact in facts
        )

    def _has_defect_problem_state_evidence(self, text: str) -> bool:
        if self._contains_any(text, DEFECT_STATE_TOKENS):
            return True
        return False

    def _has_defect_remediation_evidence(self, module1_output: Module1Output, text: str) -> bool:
        if self._contains_any(text, DEFECT_REMEDIATION_TOKENS):
            return True
        diff_type = module1_output.differentiator.difference_type
        if diff_type in {"效果增强", "风险降低"} and self._contains_any(text, {"去黄", "祛痘", "除味", "修复", "修护", "改善", "缓解"}):
            return True
        return False

    def _is_preference_only_signal(self, text: str) -> bool:
        return self._contains_any(text, PREFERENCE_ONLY_TOKENS) and not self._has_defect_problem_state_evidence(text)

    def _has_strong_defect_signal(self, module1_output: Module1Output, text: str) -> bool:
        if self._is_preference_only_signal(text):
            return False
        return self._has_defect_problem_state_evidence(text) and self._has_defect_remediation_evidence(module1_output, text)

    def _has_risk_or_protection_semantic(self, text: str) -> bool:
        # P1：判断文本是否携带任何风险/防护语义（物理安全风险词 或 防护语义词）。
        return self._is_physical_safety_fact(text) or self._contains_any(text, PROTECTION_SEMANTIC_TOKENS)

    def _supports_maintenance_task(self, module1_output: Module1Output, text: str) -> bool:
        if self._has_strong_defect_signal(module1_output, text):
            return False
        category_text = f"{module1_output.leaf_category} {module1_output.product_name}"
        diff_type = module1_output.differentiator.difference_type
        if self._is_food_like_category(category_text):
            return True
        if self._contains_any(text, MAINTENANCE_SUPPLY_TOKENS):
            return True
        if self._is_ordinary_daily_category(module1_output):
            return True
        if diff_type == "自身卖点陈述" and not self._contains_any(text, EFFICIENCY_TOKENS.union(OPERATION_EASE_TOKENS)):
            # P1：difference_type=自身卖点陈述 不再单独充分支撑维持任务；
            # 必须叠加“无任何风险/防护/缺陷语义”的硬约束，否则交回上游任务本体裁决，
            # 避免驱蚊液等功能型风险规避商品被错误降级为“生存/运转维系”。
            if self._has_risk_or_protection_semantic(text):
                return False
            return True
        return False

    def _is_food_like_category(self, text: str) -> bool:
        return self._contains_any(text, FOOD_CATEGORY_TOKENS)

    def _build_rule_tree_proposal(self, rule_context: dict[str, Any]) -> JTBDProposal:
        primary_task = rule_context["candidate_tasks"][0]
        sub_task = None
        if primary_task == "阶层与审美发信":
            sub_task = "身份跃迁"
        # PRD-1.2：needs_review 非 12 标签任务，domain 设为 FUNCTIONAL_DOMAIN 占位（短路放行不校验一致性）。
        domain = TASK_DOMAIN_MAP.get(primary_task, FUNCTIONAL_DOMAIN)
        return JTBDProposal(
            domain=domain,
            primary_task=primary_task,
            sub_task=sub_task,
            reasoning=str(rule_context["reasoning"]),
            reasoning_path=list(rule_context["reasoning_path"]),
            candidate_tasks=list(rule_context["candidate_tasks"]),
            candidate_reasons=dict(rule_context.get("candidate_reasons") or {}),
            excluded_tasks=dict(rule_context.get("excluded_tasks") or {}),
            triggered_rule=str(rule_context["triggered_rule"]),
            gate_reasons=list(rule_context.get("gate_reasons") or []),
            trace_tokens=list(rule_context.get("trace_tokens") or []),
            evidence_chain=list(rule_context["evidence_chain"]),
            functional_facts=list(rule_context.get("functional_facts") or []),
            candidate_pool=list(rule_context.get("candidate_pool") or []),
            subcategory_context=str(rule_context.get("subcategory_context") or ""),
            veto_trace=list(rule_context.get("veto_trace") or []),
            non_selected_task_reasons=[dict(item) for item in (rule_context.get("non_selected_task_reasons") or [])],
        )

    def _merge_rule_tree_context(self, proposal: JTBDProposal, rule_context: dict[str, Any]) -> JTBDProposal:
        merged = proposal.copy(deep=True)
        merged.candidate_tasks = list(rule_context["candidate_tasks"])
        merged.candidate_reasons = dict(rule_context.get("candidate_reasons") or {})
        merged.excluded_tasks = dict(rule_context.get("excluded_tasks") or {})
        merged.triggered_rule = str(rule_context["triggered_rule"])
        merged.gate_reasons = list(rule_context.get("gate_reasons") or [])
        merged.trace_tokens = list(rule_context.get("trace_tokens") or [])
        merged.evidence_chain = list(rule_context["evidence_chain"])
        merged.functional_facts = list(rule_context.get("functional_facts") or [])
        merged.candidate_pool = list(rule_context.get("candidate_pool") or [])
        merged.subcategory_context = str(rule_context.get("subcategory_context") or "")
        merged.veto_trace = list(rule_context.get("veto_trace") or [])
        # PRD-1.2：透传 Stage B 空候选 dropped reason。
        merged.non_selected_task_reasons = [dict(item) for item in (rule_context.get("non_selected_task_reasons") or [])]
        merged.reasoning_path = list(rule_context["reasoning_path"]) + list(merged.reasoning_path)
        return merged

    def _build_final_fallback(self, module1_output: Module1Output, errors: list[str], rule_context: dict[str, Any]) -> JTBDProposal:
        candidate_tasks = list(rule_context.get("candidate_tasks") or [])
        task = candidate_tasks[0] if candidate_tasks else "生存/运转维系"
        return JTBDProposal(
            domain=TASK_DOMAIN_MAP[task],
            primary_task=task,
            sub_task=None,
            reasoning="分类连续失败后，回落到规则树候选池的可审计兜底结果。",
            reasoning_path=list(rule_context.get("reasoning_path") or []) + [
                "兜底机制触发：分类器在候选池内连续失败。",
                f"错误摘要：{' | '.join(errors) if errors else '未知错误'}",
                f"按规则树候选池回落为：{task}",
            ],
            candidate_tasks=candidate_tasks or [task],
            candidate_reasons=dict(rule_context.get("candidate_reasons") or {}),
            excluded_tasks=dict(rule_context.get("excluded_tasks") or {}),
            triggered_rule=f"{rule_context.get('triggered_rule', 'fallback_rule')}_fallback",
            gate_reasons=list(rule_context.get("gate_reasons") or []),
            trace_tokens=list(rule_context.get("trace_tokens") or []),
            evidence_chain=list(rule_context.get("evidence_chain") or self._build_jtbd_evidence_chain(module1_output)),
            functional_facts=list(rule_context.get("functional_facts") or []),
            candidate_pool=list(rule_context.get("candidate_pool") or []),
            subcategory_context=str(rule_context.get("subcategory_context") or ""),
            veto_trace=list(rule_context.get("veto_trace") or []),
        )

    def _apply_hard_gates(
        self,
        payload: DiagnosticInput,
        module1_output: Module1Output,
        proposal: JTBDProposal,
    ) -> tuple[JTBDProposal, list[str], list[str]]:
        gate_notes: list[str] = []
        warnings: list[str] = []
        current = proposal.copy(deep=True)

        if current.primary_task == NEEDS_REVIEW:
            # PRD-1.2 / AC12：needs_review 中止态在 Stage D 入口短路放行，
            # 跳过越权/domain/最小证据/GR 校验，不触发武器库/HEC/完整诊断。
            gate_notes.append("Stage D：primary_task=needs_review，Stage B 候选为空中止态，跳过 Hard Gate 与 Guard Rules。")
            return current, gate_notes, warnings

        if current.primary_task not in (current.candidate_tasks or []):
            raise ValueError(f"JTBD 输出越权：{current.primary_task} 不在 candidate_tasks 内。")

        expected_domain = TASK_DOMAIN_MAP.get(current.primary_task)
        if expected_domain and current.domain != expected_domain:
            raise ValueError(f"JTBD 输出 domain 与 task 不一致: {current.domain} vs {current.primary_task}")

        if current.primary_task == "物理安全与风险规避" and not self._proposal_has_physical_safety_evidence(current):
            raise ValueError("物理安全任务最小证据不足，Hard Gate 已 veto。")

        if current.domain == EMOTIONAL_DOMAIN and self._is_ordinary_daily_category(module1_output) and not self._is_high_premium(payload):
            raise ValueError("情绪域最小证据不足：基础功能品且不满足高端/享乐/疗愈门槛。")

        if current.domain == SOCIAL_DOMAIN:
            if not current.evidence_chain:
                raise ValueError("社会域输出缺少 evidence_chain（必须绑定商品属性证据）。")
            if not current.gate_reasons:
                raise ValueError("社会域输出缺少 gate_reasons（必须含真实命中的证据描述）。")
            banned_template_fragments = (
                "符合圈层共识，因此进入社会任务",
                "命中社会域门槛",
                "规则树直接判定为社会任务",
            )
            if any(frag in self._join_reasoning(current) for frag in banned_template_fragments):
                raise ValueError("社会域 reasoning 命中预设模板文本，违反 PRD §5.1.5。")

        if current.primary_task == "阶层与审美发信" and current.sub_task not in ALLOWED_SUB_TASKS:
            raise ValueError("阶层与审美发信 缺少合法 sub_task。")

        if current.subcategory_context == "paper_products" and current.primary_task == PAPER_ESCALATABLE_TASK and not self._paper_upgrade_has_closed_loop(current.functional_facts):
            raise ValueError("纸品升级闭环断言失败：缺少对象+状态+动作，不允许通过 Stage D。")

        # PRD-1.3：GR-01~GR-04 Guard Rules（Crash Early 后置断言）。
        # 命中即把被阻断任务写入 non_selected_task_reasons(gate=blocked) 后 raise ValueError。
        self._apply_guard_rules(payload, module1_output, current)
        # PRD-1.1：Stage C 裁决完成后的收口——计算 secondary_benefits 与 dropped/blocked non_selected_task_reasons。
        self._finalize_auxiliary_fields(payload, module1_output, current)

        gate_notes.extend(current.gate_reasons)
        gate_notes.append("Stage D：仅执行越权/最小证据/审计字段断言，本轮未发生任务改写。")
        return current, gate_notes, warnings

    def _apply_guard_rules(
        self,
        payload: DiagnosticInput,
        module1_output: Module1Output,
        current: JTBDProposal,
    ) -> None:
        """PRD-1.3 GR-01~GR-04：命中即 Crash Early（raise ValueError），并先写入 non_selected_task_reasons(gate=blocked)。

        token 集合一律从 keyword_rules.yaml 的 jtbd 节读取。GR 仅在 primary_task 被误判为错误方向时触发。
        """
        text = self._module1_joined_text(module1_output)

        def _block_and_raise(task: str, code: str, reason_body: str) -> None:
            reason = f"{code}: {reason_body}"
            current.non_selected_task_reasons.append({"task": task, "reason": reason, "gate": "blocked"})
            raise ValueError(reason)

        # GR-01 效率不争主：primary=降本增效 且命中 defect_state_tokens（已显缺陷状态）
        # 但未命中 functional_breakout/operation_ease 主链路 → 效率结果误上提。
        if current.primary_task == "降本增效/懒人替代":
            if self._contains_any(text, DEFECT_STATE_TOKENS) and not self._contains_any(
                text, FUNCTIONAL_BREAKOUT_TOKENS | OPERATION_EASE_TOKENS
            ):
                _block_and_raise(
                    "降本增效/懒人替代",
                    "GR-01",
                    "命中已显缺陷状态 token 但未命中 functional_breakout/operation_ease 主链路，效率疑似结果误上提为主任务",
                )

        # GR-02 天然≠安全：primary=物理安全 但仅命中 natural_mild_preference_tokens、无真实伤害 physical_safety_tokens。
        if current.primary_task == "物理安全与风险规避":
            if self._contains_any(text, NATURAL_MILD_PREFERENCE_TOKENS) and not self._contains_any(
                text, PHYSICAL_SAFETY_TOKENS
            ):
                _block_and_raise(
                    "物理安全与风险规避",
                    "GR-02",
                    "仅命中天然/温和词，无真实伤害 token",
                )

        # GR-03 基础清洁≠情绪：primary=情绪安心 但命中 ordinary_daily_tokens 且无强焦虑/主观降险主链路。
        if current.primary_task == "情绪安心/主观降险":
            if self._is_ordinary_daily_category(module1_output) and not self._has_strong_anxiety_main_chain(module1_output):
                _block_and_raise(
                    "情绪安心/主观降险",
                    "GR-03",
                    "基础清洁品无强主观焦虑主链路，不进入情绪域",
                )

        # GR-04 犒赏高门槛：primary=自我犒赏 但 self_reward_gate 全不满足（高客单[阈值占位]/强仪式感悦己/感官主价值/心理秩序）
        # 且命中 defect_state_tokens → 回落缺陷/冲突消除。self_reward_gate 当前以可查表的高端/享乐门槛(_is_high_premium)判定，
        # 高客单绝对阈值 PRD 标注 TBD（本轮占位）。
        if current.primary_task == "自我犒赏与秩序掌控":
            # P0-2 GR-04 修正：香氛类强感官主价值(持久留香/前中后调/高级香气)构成 self_reward_gate 的「感官主价值」满足项，
            # 不被 defect_state token 过度压制回落缺陷修复。
            if (
                not self._is_high_premium(payload)
                and not self._is_fragrance_sensory_value(module1_output, text)
                and self._contains_any(text, DEFECT_STATE_TOKENS)
            ):
                _block_and_raise(
                    "自我犒赏与秩序掌控",
                    "GR-04",
                    "self_reward_gate 全不满足（非高客单/无强仪式感悦己/无感官主价值/无心理秩序）且命中已显缺陷状态 token",
                )

    def _has_weak_emotional_secondary_evidence(self, module1_output: Module1Output, text: str) -> bool:
        weak_emotional_tokens = {
            "熬夜", "状态", "舒缓", "放松", "安心", "确定感", "仪式感", "氛围感", "治愈",
        }
        if self._contains_any(text, weak_emotional_tokens):
            return True
        joined = f"{module1_output.leaf_category} {module1_output.product_name}"
        skincare_tokens = {"面膜", "精华", "乳液", "面霜", "眼霜", "精油", "护肤"}
        return self._contains_any(joined, skincare_tokens) and self._contains_any(text, {"熬夜", "状态"})

    def _finalize_auxiliary_fields(
        self,
        payload: DiagnosticInput,
        module1_output: Module1Output,
        current: JTBDProposal,
    ) -> None:
        """PRD-1.1：Stage C 收口处计算 secondary_benefits 与 non_selected_task_reasons。

        - secondary_benefits 只写“成立但不争主”的真实收益（GR-01 效率结果、Stage C 弱证据落选真实收益、GR-04 弱仪式感）。
        - non_selected_task_reasons 只记录①进入候选池但落选(dropped)②被 Guard Rule 显式阻断(blocked，reason 标 GR 编号)
          ③ 已成立但被降级为 secondary(demoted_to_secondary)。
        - 只有 gate=blocked 的任务会从 secondary_benefits 去重剔除；demoted_to_secondary 必须保留 secondary。
        """
        text = self._module1_joined_text(module1_output)
        primary = current.primary_task
        primary_domain = TASK_DOMAIN_MAP.get(primary)

        has_efficiency = self._contains_any(text, EFFICIENCY_TOKENS | OPERATION_EASE_TOKENS)
        has_natural_mild = self._contains_any(text, NATURAL_MILD_PREFERENCE_TOKENS)
        has_physical_safety = self._contains_any(text, PHYSICAL_SAFETY_TOKENS)
        has_defect_signal = self._contains_any(text, DEFECT_STATE_TOKENS)
        is_ordinary = self._is_ordinary_daily_category(module1_output)
        has_strong_anxiety_main_chain = self._has_strong_anxiety_main_chain(module1_output)
        high_premium = self._is_high_premium(payload)
        has_weak_emotional_secondary = self._has_weak_emotional_secondary_evidence(module1_output, text)

        secondary: list[str] = list(current.secondary_benefits)
        blocked: list[dict[str, str]] = []
        demoted: list[dict[str, str]] = []

        # secondary（成立但不争主）——
        # GR-01：主=缺陷/冲突消除 且存在省时/少洗头/省步骤等效率结果 → 降本增效/懒人替代 入 secondary，
        # 同时记录“效率不上提为主任务”的 demoted_to_secondary reason。
        if primary == "缺陷修复/冲突消除" and has_efficiency:
            if "降本增效/懒人替代" not in secondary:
                secondary.append("降本增效/懒人替代")
            demoted.append(
                {
                    "task": "降本增效/懒人替代",
                    "reason": "GR-01: 效率表达是效果结果非主诉，降级为 secondary_benefit",
                    "gate": "demoted_to_secondary",
                }
            )

        # GR-04：常规功效护肤被挡回功能域后，若存在弱仪式感/弱情绪安心证据，则保留为 secondary。
        if (
            primary == "缺陷修复/冲突消除"
            and has_defect_signal
            and not high_premium
            and has_weak_emotional_secondary
        ):
            if "情绪安心/主观降险" not in secondary:
                secondary.append("情绪安心/主观降险")
            blocked.append(
                {
                    "task": "自我犒赏与秩序掌控",
                    "reason": "GR-04: 自我犒赏未达高门槛，弱情绪/体验收益仅保留为 secondary",
                    "gate": "blocked",
                }
            )

        # blocked（被 Guard Rule 显式阻断的错误方向，不得进 secondary）——
        # GR-02：仅命中天然/温和、无真实伤害 token → 物理安全方向 blocked。
        if (
            primary_domain == FUNCTIONAL_DOMAIN
            and primary != "物理安全与风险规避"
            and has_natural_mild
            and not has_physical_safety
        ):
            blocked.append(
                {
                    "task": "物理安全与风险规避",
                    "reason": "GR-02: 仅命中天然/温和词，无真实伤害 token，归信任存量",
                    "gate": "blocked",
                }
            )

        # GR-03：功能域基础清洁品（ordinary_daily）无强焦虑主链路 → 预防性补写情绪安心 blocked 审计记录。
        if (
            primary_domain == FUNCTIONAL_DOMAIN
            and is_ordinary
            and not has_strong_anxiety_main_chain
            and not has_weak_emotional_secondary
        ):
            blocked.append(
                {
                    "task": "情绪安心/主观降险",
                    "reason": "GR-03: 基础清洁品无强主观焦虑主链路，不进入情绪域",
                    "gate": "blocked",
                }
            )

        # 仅去除被 blocked 的方向；demoted_to_secondary 必须保留在 secondary。
        blocked_tasks = {item["task"] for item in blocked}
        demoted_tasks = {item["task"] for item in demoted}
        secondary = [task for task in secondary if task not in blocked_tasks and task != primary]

        # dropped：进入过候选池但落选的任务（gate=dropped）。
        existing_keys = {(item.get("task"), item.get("gate")) for item in current.non_selected_task_reasons}
        dropped: list[dict[str, str]] = []
        for task in current.candidate_tasks:
            if task == primary or task == NEEDS_REVIEW or task not in VALID_JTBD:
                continue
            if task in blocked_tasks or task in demoted_tasks or (task, "dropped") in existing_keys:
                continue
            dropped.append({"task": task, "reason": "进入候选池但 Stage C 裁决落选", "gate": "dropped"})

        # 合并去重写回（保留 _apply_guard_rules 之前可能已存在的条目）。
        merged_nstr = list(current.non_selected_task_reasons)
        for item in demoted + blocked + dropped:
            key = (item["task"], item["gate"])
            if key not in existing_keys:
                merged_nstr.append(item)
                existing_keys.add(key)

        current.secondary_benefits = list(dict.fromkeys(secondary))
        current.non_selected_task_reasons = merged_nstr

    def _derive_category_intent_matrix(self, module1_output: Module1Output, proposal: JTBDProposal) -> CategoryIntentMatrix:
        ocean = self._derive_ocean(module1_output)
        frequency = self._derive_frequency(module1_output)
        reasons = [f"Step 1：根据商品文本判定品类状态为 {ocean}。"]
        evidence_chain = self._build_jtbd_evidence_chain(module1_output)
        if ocean == "蓝海":
            matrix_label = f"蓝海×{frequency}"
            category_intent = CATEGORY_INTENT_COPY[(ocean, None, frequency)]
            reasons.append("Step 2：蓝海不再细分核心/破圈，直接进入新品类教育意图。")
            reasons.append(f"Step 3：消费频次判定为 {frequency}。")
            return CategoryIntentMatrix(
                ocean=ocean,
                competition_focus=None,
                frequency=frequency,
                domain_route_rule="blue_ocean_direct_rule",
                matrix_label=matrix_label,
                category_intent=category_intent,
                competition_focus_reason="蓝海品类无需进入核心/破圈分流，直接进入品类教育。",
                competition_focus_evidence_chain=evidence_chain,
                difference_type_route_result="blue_ocean_direct_rule",
                reasoning=reasons,
            )

        competition_focus, focus_reason, focus_evidence_chain, route_result, focus_reasons = self._derive_competition_focus(module1_output, proposal)
        reasons.extend(focus_reasons)
        reasons.append(f"Step 3：消费频次判定为 {frequency}。")
        matrix_label = f"红海-{competition_focus}×{frequency}"
        category_intent = CATEGORY_INTENT_COPY[(ocean, competition_focus, frequency)]
        return CategoryIntentMatrix(
            ocean=ocean,
            competition_focus=competition_focus,
            frequency=frequency,
            domain_route_rule=self._domain_route_rule_name(proposal.domain),
            matrix_label=matrix_label,
            category_intent=category_intent,
            competition_focus_reason=focus_reason,
            competition_focus_evidence_chain=focus_evidence_chain,
            difference_type_route_result=route_result,
            reasoning=reasons,
        )

    def _derive_ocean(self, module1_output: Module1Output) -> Literal["蓝海", "红海"]:
        text = self._module1_joined_text(module1_output)
        if self._contains_any(text, BLUE_OCEAN_TOKENS):
            return "蓝海"
        return "红海"

    def _derive_competition_focus(
        self,
        module1_output: Module1Output,
        proposal: JTBDProposal,
    ) -> tuple[Literal["核心", "破圈"], str, list[dict[str, str]], str, list[str]]:
        differentiator = module1_output.differentiator
        diff_domain = differentiator.difference_domain
        diff_type = differentiator.difference_type
        route_key = f"{diff_domain}.{diff_type}"
        comparison_object = differentiator.comparison_object
        evidence_chain = [
            {"evidence_source": item.evidence_source, "evidence_text": item.evidence_text}
            for item in differentiator.evidence_chain
        ]
        evidence_text = " ".join(item["evidence_text"] for item in evidence_chain)
        text = self._module1_joined_text(module1_output)

        if not comparison_object:
            return (
                "核心",
                "comparison_object 为空，按 PRD 空值协议默认回落核心竞争，避免无证据触发破圈路由。",
                evidence_chain,
                f"{route_key}->comparison_object:null->核心",
                [
                    f"Step 2：优先读取 difference_domain + difference_type={route_key}。",
                    "comparison_object=null，业务假设为证据不足不触发旧路径替代，因此默认回落核心。",
                ],
            )

        if diff_domain in {"emotional", "social"}:
            raise ValueError(
                f"竞争焦点断言失败：{route_key} 不得直接进入功能域核心/破圈字典路由，必须先走任务域链路。"
            )

        if diff_domain == "trust" and diff_type != "信任缓释":
            raise ValueError(f"竞争焦点断言失败：非法 trust 差异类型 {route_key}")

        if diff_type in {"步骤压缩", "体验升级", "效果增强", "成本优化"}:
            breakout_tokens = FUNCTIONAL_BREAKOUT_TOKENS.union({"旧流程", "旧动作", "传统"})
            if self._contains_any(evidence_text, breakout_tokens) or comparison_object in {"跨品类旧动作", "旧形态方案"}:
                return (
                    "破圈",
                    f"difference_type 原始路由为 `{diff_type}->核心`，但证据链明确显示旧 SOP 被改写，因此按冲突复核改判为破圈。",
                    evidence_chain,
                    f"{route_key}->核心->证据复核->破圈",
                    [
                        f"Step 2：优先读取 difference_domain + difference_type={route_key}，首轮路由为 核心。",
                        "冲突复核：证据链出现旧流程替代/新形态承接事实，因此按 PRD 冲突规则改判为破圈。",
                    ],
                )
            focus = "核心"
            route_result = f"{route_key}->核心"
            reason = f"差异类型 `{route_key}` 先路由到同类同路径优化，因此判定为核心竞争。"
            notes = [
                f"Step 2：优先读取 difference_domain + difference_type={route_key}，按 PRD 字典首轮路由到 核心。",
                "该样本仍在同类/同路径解法内竞争，未出现旧 SOP 改写证据。",
            ]
            return focus, reason, evidence_chain, route_result, notes

        if diff_type == "新形态替代":
            focus = "破圈"
            route_result = f"{route_key}->破圈"
            reason = "差异类型为新形态替代，证据说明商品以新形态承接旧需求，属于破圈竞争。"
            notes = [
                f"Step 2：优先读取 difference_domain + difference_type={route_key}，按 PRD 字典首轮路由到 破圈。",
                "证据链显示该商品在改写旧 SOP 或旧形态解决路径。",
            ]
            return focus, reason, evidence_chain, route_result, notes

        if diff_type == "风险降低":
            breakout_tokens = FUNCTIONAL_BREAKOUT_TOKENS.union({"新路径", "改写", "替代", "旧流程", "旧动作"})
            core_tokens = {"更安全", "更稳", "更稳妥", "安心", "防护", "保护", "避险"}
            if self._contains_any(evidence_text, breakout_tokens) or comparison_object in {"跨品类旧动作", "旧形态方案"}:
                return (
                    "破圈",
                    "风险降低发生在新路径/新规则下，属于通过改写旧方案来规避风险，因此判定为破圈。",
                    evidence_chain,
                    f"{route_key}->破圈",
                    [
                        f"Step 2：difference_domain + difference_type={route_key}，进入证据复核。",
                        "证据链显示风险规避依赖新路径或旧 SOP 替代，因此判定为破圈。",
                    ],
                )
            if self._contains_any(evidence_text, core_tokens) or self._contains_any(text, core_tokens):
                return (
                    "核心",
                    "风险降低仍发生在同路径内，只是更安全或更稳妥，因此判定为核心。",
                    evidence_chain,
                    f"{route_key}->核心",
                    [
                        f"Step 2：difference_domain + difference_type={route_key}，进入证据复核。",
                        "证据链仅支持同路径更安全/更稳妥，未支持路径改写，因此判定为核心。",
                    ],
                )
            raise ValueError("竞争焦点断言失败：`风险降低` 缺少可支撑核心/破圈的证据。")

        if diff_type == "信任缓释":
            if comparison_object in {"跨品类旧动作", "旧形态方案"} or self._contains_any(evidence_text, FUNCTIONAL_BREAKOUT_TOKENS):
                return (
                    "破圈",
                    "信任缓释本身不决定竞争焦点；结合比较对象与证据，当前样本仍呈现旧路径替代，因此判定为破圈。",
                    evidence_chain,
                    f"{route_key}->证据复核->破圈",
                    [
                        f"Step 2：difference_domain + difference_type={route_key}，本身不直接决定竞争焦点。",
                        "结合比较对象与证据，样本呈现旧路径替代/新形态承接，因此判定为破圈。",
                    ],
                )
            if comparison_object in {"同类旧方案", "同赛道竞品"} or self._contains_any(text, {"同类", "升级", "体验", "参数"}):
                return (
                    "核心",
                    "信任缓释本身不决定竞争焦点；结合比较对象与商品事实，当前样本仍是同赛道优化，因此判定为核心。",
                    evidence_chain,
                    f"{route_key}->证据复核->核心",
                    [
                        f"Step 2：difference_domain + difference_type={route_key}，本身不直接决定竞争焦点。",
                        "结合比较对象与商品事实，样本仍在同赛道旧方案内竞争，因此判定为核心。",
                    ],
                )
            raise ValueError("竞争焦点断言失败：`信任缓释` 不能单独决定核心/破圈，且当前缺少可复核证据。")

        raise ValueError(f"竞争焦点断言失败：未覆盖的 difference_type={diff_type}")

    def _derive_frequency(self, module1_output: Module1Output) -> Literal["快消", "耐消"]:
        text = self._module1_joined_text(module1_output)
        durable_keyword = self._find_first_keyword(text, DURABLE_TOKENS)
        if durable_keyword:
            self._record_keyword_rule_trace(
                field_name="frequency",
                output_value="耐消",
                rule_path="product_diagnoser.frequency.durable_tokens",
                matched_keyword=durable_keyword,
            )
            return "耐消"
        fast_keyword = self._find_first_keyword(text, FAST_MOVING_TOKENS)
        if fast_keyword:
            self._record_keyword_rule_trace(
                field_name="frequency",
                output_value="快消",
                rule_path="product_diagnoser.frequency.fast_moving_tokens",
                matched_keyword=fast_keyword,
            )
            return "快消"
        business_category, _ = self._resolve_price_band(module1_output)
        if business_category in DURABLE_BUSINESS_CATEGORIES:
            self._record_keyword_rule_trace(
                field_name="frequency",
                output_value="耐消",
                rule_path="product_diagnoser.frequency.durable_business_categories",
                matched_keyword=business_category,
            )
            return "耐消"
        return "快消"

    def _domain_route_rule_name(self, domain: str) -> str:
        if domain == FUNCTIONAL_DOMAIN:
            return "A_functional_competitor_rule"
        if domain == EMOTIONAL_DOMAIN:
            return "B_emotional_competitor_rule"
        return "C_social_competitor_rule"

    def _derive_product_intent_matrix(self, payload: DiagnosticInput, module1_output: Module1Output) -> ProductIntentMatrix:
        brand_tier, trust_barrier, whitelist_hit = self._resolve_brand_tier(payload)
        business_category, median_price_threshold = self._resolve_price_band(module1_output)
        relative_price_level = self._read_engine_price_level(payload)
        price_value = self._parse_price(module1_output.price)
        financial_risk: Literal["高", "中", "低"] = "高" if relative_price_level == "高水位" else "低"

        reasoning = [
            f"信任阻力：店铺 {'命中' if whitelist_hit else '未命中'} brand_whitelist.csv，因此品牌层级判定为 {brand_tier}。",
            f"财务阻力：类目 {business_category} 的价格带中位数资产为 {median_price_threshold:.2f}，但外层仅用于审计留痕。",
            f"价格属性严格读取 engine_node 输出，当前水位为 {relative_price_level}，因此财务风险为 {financial_risk}。",
        ]
        matrix_label = f"{brand_tier}×{relative_price_level}"
        product_intent = PRODUCT_INTENT_COPY[(brand_tier, financial_risk)]
        return ProductIntentMatrix(
            brand_tier=brand_tier,
            trust_barrier=trust_barrier,
            financial_risk=financial_risk,
            relative_price_level=relative_price_level,
            matrix_label=matrix_label,
            business_category=business_category,
            median_price_threshold=median_price_threshold,
            price_value=price_value,
            product_intent=product_intent,
            reasoning=reasoning,
        )

    def _resolve_brand_tier(self, payload: DiagnosticInput) -> tuple[Literal["大牌官方", "大牌经销", "白牌"], Literal["极低", "中", "高"], bool]:
        whitelist = _load_brand_whitelist()
        hit = bool(payload.shop_name and payload.shop_name in whitelist)
        if not hit:
            return "白牌", "高", False

        matched_rule: tuple[str, str, str] | None = None
        default_rule: tuple[str, str, str] | None = None
        for suffix, brand_tier, trust_barrier in _load_store_suffix_trust_dict():
            if suffix == "default":
                default_rule = (suffix, brand_tier, trust_barrier)
                continue
            if suffix and suffix in payload.shop_name:
                current_rule = (suffix, brand_tier, trust_barrier)
                if matched_rule is None or len(suffix) > len(matched_rule[0]):
                    matched_rule = current_rule

        final_rule = matched_rule or default_rule
        if final_rule is None:
            raise ValueError("store_suffix_trust_dict.csv 缺少 default 配置，无法完成白名单命中后的二层判定。")
        _, brand_tier, trust_barrier = final_rule
        return brand_tier, trust_barrier, True

    def _resolve_price_band(self, module1_output: Module1Output) -> tuple[str, float]:
        lookup = _build_price_band_lookup()
        normalized_leaf_category = module1_output.leaf_category.strip()
        if not normalized_leaf_category:
            raise ValueError("leaf_category 不能为空，无法匹配价格中位数阈值。")
        if normalized_leaf_category in lookup:
            median_price_threshold = lookup[normalized_leaf_category]
            if median_price_threshold <= 0:
                raise ValueError(f"price_band_dict.csv 中位数阈值非法: {normalized_leaf_category}")
            return normalized_leaf_category, median_price_threshold

        search_space = [module1_output.leaf_category.strip(), module1_output.product_name.strip()]
        for category, median_price_threshold in lookup.items():
            for text in search_space:
                if category and text and (category in text or text in category):
                    if median_price_threshold <= 0:
                        raise ValueError(f"price_band_dict.csv 中位数阈值非法: {category}")
                    return category, median_price_threshold

        raise ValueError(
            f"price_band_dict.csv 中不存在叶子类目映射: leaf_category={module1_output.leaf_category!r}。"
        )

    def _read_engine_price_level(self, payload: DiagnosticInput) -> Literal["高水位", "低水位"]:
        engine_node = payload.engine_node or {}
        if not engine_node:
            raise ValueError("价格引擎缺失：拿不到 engine_node 结果。")
        for key in ("relative_price_level", "price_level", "price_band_level"):
            value = str(engine_node.get(key) or "").strip()
            if value in {"高水位", "低水位"}:
                return value  # type: ignore[return-value]
            if value in {"高", "high", "HIGH"}:
                return "高水位"
            if value in {"低", "low", "LOW"}:
                return "低水位"
        raise ValueError("价格引擎缺失：engine_node 未提供合法高低水位结果。")

    def _compose_reasoning_path(
        self,
        raw_proposal: JTBDProposal,
        gated_proposal: JTBDProposal,
        gate_notes: list[str],
        category_matrix: CategoryIntentMatrix,
        product_matrix: ProductIntentMatrix,
    ) -> list[str]:
        path: list[str] = []
        path.append(
            f"规则树候选池：{'、'.join(raw_proposal.candidate_tasks or [raw_proposal.primary_task])}；触发规则：{raw_proposal.triggered_rule}"
        )
        path.extend(raw_proposal.reasoning_path or [raw_proposal.reasoning])
        path.extend(gate_notes)
        if gated_proposal.primary_task != raw_proposal.primary_task or gated_proposal.domain != raw_proposal.domain:
            path.append(f"网关后任务：{gated_proposal.domain} / {gated_proposal.primary_task}")
        path.extend(category_matrix.reasoning)
        path.extend(product_matrix.reasoning)
        return path

    def _build_assertions(self) -> list[str]:
        return [
            "已采用 Stage A 前置门槛 + Stage B 功能候选池 + Stage C 候选池内归并 + Stage D 只 veto 的四段式 JTBD 引擎。",
            "JTBD 一级任务严格限制在 11 个标准任务内，且不得越出规则树候选池。",
            "物理安全只允许在 Stage A 强唯一锁定，Stage D 仅校验最小证据与越权，不再重复改写 primary_task。",
            "竞争焦点必须优先读取 difference_type 路由，品牌白名单必须走字典路由，价格属性必须锚定 engine_node 输出。",
            "P1 降级词表已外置到 config/keyword_rules.yaml，命中规则需写入 source_evidence；若缺少追溯信息则直接断言失败。",
        ]

    def _to_module3_category_attr(self, category_matrix: CategoryIntentMatrix) -> str:
        if category_matrix.ocean == "蓝海":
            return "蓝海"
        return f"红海-{category_matrix.competition_focus}"

    def _to_module3_trust_attr(self, product_matrix: ProductIntentMatrix) -> str:
        return "大牌" if "大牌" in product_matrix.brand_tier else "白牌"

    def _to_module3_price_attr(self, product_matrix: ProductIntentMatrix) -> str:
        return "高价" if product_matrix.relative_price_level == "高水位" else "低价"

    def _derive_module3_modifiers(self, payload: DiagnosticInput, product_matrix: ProductIntentMatrix) -> list[str]:
        modifiers: list[str] = []
        if product_matrix.brand_tier == "大牌经销":
            modifiers.append("channel_risk")
        if self._has_endorsement(payload):
            modifiers.append("has_endorsement")
        return modifiers

    def _has_endorsement(self, payload: DiagnosticInput) -> bool:
        text = f"{payload.core_selling_point} {self._stringify_differentiator(payload.differentiator)}"
        return self._contains_any(text, ENDORSEMENT_TOKENS)

    def _strategy_primary_code(self, nodes: list[dict[str, Any]]) -> str:
        if not isinstance(nodes, list) or not nodes:
            return ""
        return str(nodes[0].get("code") or "").strip()

    def _strategy_primary_label(self, nodes: list[dict[str, Any]]) -> str:
        if not isinstance(nodes, list) or not nodes:
            return ""
        return str(nodes[0].get("label") or "").strip()

    def _build_persuasion_product_fact(
        self,
        payload: DiagnosticInput,
        module1_output: Module1Output,
        proposal: JTBDProposal,
        category_matrix: CategoryIntentMatrix,
        product_matrix: ProductIntentMatrix,
    ) -> dict[str, Any]:
        """F5：从 diagnose 内部已产出的 module1_output / proposal / matrices 组装 persuasion 引擎
        所需的 product_fact（不再由外部 runner 拼装）。

        仅映射 generate_profile 真实消费的键（见 persuasion_requirement_engine.generate_profile
        第 180-235 行、_build_main_route、_check_activation_condition）：
          - ``leaf_category`` = module1_output.leaf_category（类目路由）；
          - ``jtbd_level1`` = proposal.domain，``jtbd_level2`` = proposal.primary_task（JTBD 模板召回）；
          - ``cognition/frequency/trust/price_attribute`` = 品类/商品矩阵（主说服路线四属性，缺失会 Crash Early）；
          - ``selling_points`` / ``source_evidence`` / ``title`` / ``category`` = activation_condition 证据校验输入；
          - ``risk_points`` = 风险/证据类要求优先级强化（结构化输入暂无，按引擎语义传 []）。
        所有取值均来自已有真实输入，缺失字段不伪造（留空或传 []）。
        """
        competition_focus = getattr(category_matrix, "competition_focus", None)
        # cognition_attribute 与 runner 既有口径一致：蓝海/红海 × 核心/破圈；competition_focus 缺失时回落到 ocean 单值。
        cognition_attribute = (
            f"{category_matrix.ocean}-{competition_focus}" if competition_focus else category_matrix.ocean
        )
        selling_points = [s for s in [(module1_output.core_selling_point or "").strip()] if s]
        source_evidence = [
            str(item).strip() for item in (payload.bridge_source_evidence or []) if str(item).strip()
        ]
        product_fact: dict[str, Any] = {
            "leaf_category": module1_output.leaf_category or "",
            "category": module1_output.leaf_category or "",
            "title": module1_output.product_name or "",
            "jtbd_level1": proposal.domain,
            "jtbd_level2": proposal.primary_task,
            "cognition_attribute": cognition_attribute,
            "frequency_attribute": category_matrix.frequency,
            "trust_attribute": product_matrix.trust_barrier,
            "price_attribute": product_matrix.relative_price_level,
            "selling_points": selling_points,
            "source_evidence": source_evidence,
            # 第五批：透传模块 1 目标人群原始未截断信号，供 profile 引擎识别 gift_context
            # （送礼对象/时机/意图），不剥离节日/送礼/对象/关系词。
            "target_people_raw": module1_output.target_people_raw or "",
            # 结构化 risk_points 当前不在商品输入协议内，没有就传 []（禁止伪造）。
            "risk_points": [],
        }
        return product_fact

    def _derive_product_target_audience(
        self,
        module1_output: Module1Output,
        proposal: JTBDProposal,
        product_matrix: ProductIntentMatrix,
        *,
        persuasion_requirement_profile: dict[str, Any] | None = None,
        product_fact: Mapping[str, Any] | None = None,
    ) -> ProductTargetAudience:
        """确定性派生商品目标人群（八大人群坐标）。

        四属性链路：唯一任务→购买角色→年龄/性别→消费力。任一属性所需输入
        缺失即 Crash Early（代码断言，不依赖 LLM 自觉）。

        F3：audience 强依赖 persuasion_requirement_profile —— profile 为空或其
        persuasion_requirements 为空即 Crash Early；reasoning_chain 第四段
        persuasion_profile_to_audience 必须引用 profile 中真实 required/high requirement。
        """
        # ---- F3 Crash Early：profile 非空校验（函数开头，先于八大人群推导）----
        if not persuasion_requirement_profile or not persuasion_requirement_profile.get(
            "persuasion_requirements"
        ):
            raise ValueError(
                "product_target_audience: persuasion_requirement_profile 为空或 "
                "persuasion_requirements 为空，audience 强依赖 profile，禁止兜底生成（Crash Early）。"
            )
        # ---- Crash Early：四属性输入完整性校验 ----
        primary_task = (proposal.primary_task or "").strip()
        if not primary_task:
            raise ValueError("product_target_audience: primary_task 为空，无法确定购买角色（属性缺失）。")
        role = TASK_TO_PURCHASE_ROLE.get(primary_task)
        if role is None:
            raise ValueError(
                f"product_target_audience: primary_task={primary_task!r} 不在购买角色映射表内（属性缺失）。"
            )
        brand_tier = (getattr(product_matrix, "brand_tier", "") or "").strip()
        price_level = (getattr(product_matrix, "relative_price_level", "") or "").strip()
        if not brand_tier:
            raise ValueError("product_target_audience: 缺少 product_matrix.brand_tier，无法判定消费力（属性缺失）。")
        if not price_level:
            raise ValueError(
                "product_target_audience: 缺少 product_matrix.relative_price_level，无法判定消费力（属性缺失）。"
            )
        leaf_category = (module1_output.leaf_category or "").strip()
        product_name = (module1_output.product_name or "").strip()
        if not leaf_category and not product_name:
            raise ValueError(
                "product_target_audience: leaf_category 与 product_name 同时为空，无法判定年龄/性别（属性缺失）。"
            )

        # ---- 第五批：gift_context 双重角色分支（优先级高于品牌价格消费力规则）----
        # 识别送礼场景后，primary=购买决策者（送礼者），secondary=受礼者；
        # 第四段必须解释购买者与受礼者关系并引用真实 requirement。
        # 非送礼样本返回 None，完全保持原八大人群派生逻辑不变。
        # 识别口径与 profile 引擎完全一致（见 persuasion_requirement_engine
        # ._apply_gift_context_requirements）：优先消费 product_fact 的同一组信号
        # （title/selling_points/source_evidence/target_people_raw/category），
        # 保证 profile 激活礼赠类 requirement 时 audience 也能同步拆分双重角色，
        # 不出现「profile 判为送礼、audience 仍走机械八大人群」的口径漂移。
        # product_fact 缺省（仅 module1 直连场景）时回落到模块 1 字段。
        if product_fact is not None:
            gift_segments: list[Any] = [
                product_fact.get("title"),
                product_fact.get("selling_points"),
                product_fact.get("source_evidence"),
                product_fact.get("target_people_raw"),
                product_fact.get("category"),
                product_fact.get("leaf_category"),
            ]
        else:
            gift_segments = [
                module1_output.target_people_raw or "",
                product_name,
                module1_output.core_selling_point or "",
                leaf_category,
            ]
        gift_context = detect_gift_context(gift_segments)
        if gift_context and gift_context.get("is_gift"):
            return self._build_gift_context_audience(
                primary_task=primary_task,
                role=role,
                gift_context=gift_context,
                persuasion_requirement_profile=persuasion_requirement_profile,
            )

        text = "".join(
            [
                leaf_category,
                product_name,
                module1_output.core_selling_point or "",
                module1_output.target_people or "",
            ]
        )

        # ---- Step2：性别 ----
        female_hits = [kw for kw in AUDIENCE_FEMALE_KEYWORDS if kw in text]
        male_hits = [kw for kw in AUDIENCE_MALE_KEYWORDS if kw in text]
        if female_hits and not male_hits:
            gender_axis = "female"
            gender_reason = f"命中女性权重品类信号（{'、'.join(female_hits[:5])}）且无男性信号 → 女性"
        elif male_hits and not female_hits:
            gender_axis = "male"
            gender_reason = f"命中男性权重品类信号（{'、'.join(male_hits[:5])}）且无女性信号 → 男性"
        elif role in AUDIENCE_FEMALE_DEFAULT_ROLES:
            gender_axis = "female"
            gender_reason = f"品类性别信号不唯一，按购买角色「{role}」默认 → 女性"
        else:
            gender_axis = "mixed"
            gender_reason = f"品类性别信号不唯一且角色「{role}」无强默认 → 性别 mixed"

        # ---- Step2：年龄 ----
        mature_hits = [kw for kw in AUDIENCE_MATURE_KEYWORDS if kw in text]
        young_hits = [kw for kw in AUDIENCE_YOUNG_KEYWORDS if kw in text]
        if mature_hits:
            age_axis = "mature"
            age_reason = f"命中年长品类信号（{'、'.join(mature_hits[:5])}）→ 年长"
        elif young_hits:
            age_axis = "young"
            age_reason = f"命中年轻品类信号（{'、'.join(young_hits[:5])}）→ 年轻"
        elif role in AUDIENCE_AGE_DEFAULT_ROLES:
            age_axis = AUDIENCE_AGE_DEFAULT_ROLES[role]
            age_reason = f"无显式年龄信号，按购买角色「{role}」默认 → {age_axis}"
        else:
            age_axis = "mixed"
            age_reason = f"无显式年龄信号且角色「{role}」无强默认 → 年龄 mixed"

        # ---- Step3：消费力（SSOT：product_matrix 已走 brand_whitelist.csv 路由）----
        caveats: list[str] = []
        if price_level == "高水位":
            consumption_primary = "mid_high"
            consumption_extra: list[str] = []
            consumption_reason = f"{brand_tier} + 高水位（高于类目中位价，为品质/品牌付溢价）→ 中高消费力"
            if brand_tier == "大牌官方":
                # D6：套装价值也可覆盖低消费力，追加同年龄+性别的低消费力坐标作为 secondary
                consumption_extra = ["low"]
                caveats.append(
                    "大牌官方且高水位：若强调套装/分量价值，也可覆盖低消费力人群（追加低消费力 secondary）。"
                )
        else:  # 低水位
            if brand_tier == "大牌官方":
                consumption_primary = "mid_high"
                consumption_extra = []
                consumption_reason = f"{brand_tier} + 低水位 → 判为中高消费力（高端品牌低价多为中高消引流入口）"
            else:
                consumption_primary = "low"
                consumption_extra = []
                consumption_reason = f"{brand_tier} + 低水位（处类目低价带）→ 低消费力"

        # ---- 组合输出：三轴 mixed 用 expand_axis 展开，主组合进 primary ----
        age_options = expand_axis(age_axis, axis="age")
        gender_options = expand_axis(gender_axis, axis="gender")
        age_primary = age_options[0]
        gender_primary = gender_options[0]
        consumption_options = [consumption_primary] + consumption_extra

        primary_group = compose_audience_group(age_primary, gender_primary, consumption_primary)
        primary_reason = (
            f"任务「{primary_task}」对应购买角色「{role}」；{age_reason}；{gender_reason}；{consumption_reason}。"
        )
        primary_audiences = [
            AudienceGroupJudgment(audience_group=primary_group, fit_level="primary", reason=primary_reason)
        ]

        # secondary：完整笛卡尔积中除主组合外的其余组合（年龄/性别 mixed 展开 + 消费力套装覆盖）
        secondary_audiences: list[AudienceGroupJudgment] = []
        seen = {primary_group}
        for a in age_options:
            for g in gender_options:
                for c in consumption_options:
                    group = compose_audience_group(a, g, c)
                    if group in seen:
                        continue
                    seen.add(group)
                    if c != consumption_primary:
                        sec_reason = (
                            f"购买角色「{role}」下，{brand_tier}套装/分量价值也可覆盖"
                            f"{CONSUMPTION_LABELS[c]}人群，作为次级目标。"
                        )
                    else:
                        sec_reason = (
                            f"购买角色「{role}」的年龄/性别边界不唯一（{age_reason}；{gender_reason}），"
                            f"该组合作为次级覆盖目标。"
                        )
                    secondary_audiences.append(
                        AudienceGroupJudgment(audience_group=group, fit_level="secondary", reason=sec_reason)
                    )
        secondary_audiences = secondary_audiences[:3]

        # ---- caveats（AC6）----
        if brand_tier == "大牌官方" and price_level == "低水位":
            caveats.append("大牌官方却处低水位：品牌力与价格信号冲突，低价可能是中高消费力引流入口。")
        if primary_task == "礼赠与关系表达":
            caveats.append("礼赠与关系表达：送礼者≠收礼者，目标人群需区分购买决策者与使用者。")
        if gender_axis == "mixed" or age_axis == "mixed":
            caveats.append("年龄或性别判为 mixed：人群边界不清，已展开多坐标覆盖。")

        # ---- F3：第四段 persuasion_profile_to_audience —— 引用 profile 中真实 required/high requirement ----
        requirements = list(persuasion_requirement_profile.get("persuasion_requirements") or [])

        def _req_attr(req: Any, key: str) -> Any:
            if isinstance(req, Mapping):
                return req.get(key)
            return getattr(req, key, None)

        # 优先取 required 为真或 priority 为 high 的条目；都没有则回退取第一条（仍引用真实字段，不留空）。
        high_priority_reqs = [
            r
            for r in requirements
            if bool(_req_attr(r, "required")) or str(_req_attr(r, "priority") or "") == "high"
        ]
        referenced_reqs = high_priority_reqs if high_priority_reqs else requirements[:1]
        ref_labels = [
            f"「{_req_attr(r, 'requirement_name')}({_req_attr(r, 'requirement_id')})」"
            for r in referenced_reqs[:3]
        ]
        persuasion_profile_to_audience = (
            f"需满足说服要求{'、'.join(ref_labels)}（profile 共 {len(requirements)} 条说服要求，"
            f"其中 required/high 优先项 {len(high_priority_reqs)} 条）"
            f"→ 因此核心说服对象指向「{primary_group}」。"
        )

        reasoning_chain = AudienceReasoningChain(
            task_to_role=f"唯一任务「{primary_task}」对应购买角色「{role}」。",
            role_category_to_age_gender=f"购买角色「{role}」叠加品类信号：{age_reason}；{gender_reason}。",
            brand_price_to_consumption_power=f"{consumption_reason}。",
            persuasion_profile_to_audience=persuasion_profile_to_audience,
        )

        return ProductTargetAudience(
            primary_audiences=primary_audiences,
            secondary_audiences=secondary_audiences,
            weak_fit_audiences=[],
            reasoning_chain=reasoning_chain,
            caveats=caveats,
        )

    def _build_gift_context_audience(
        self,
        *,
        primary_task: str,
        role: str,
        gift_context: dict[str, Any],
        persuasion_requirement_profile: dict[str, Any],
    ) -> ProductTargetAudience:
        """第五批：gift_context 送礼场景下的双重角色人群派生。

        - primary_audiences = purchase_decider（购买决策者 / 送礼者）；
        - secondary_audiences = gift_recipient（受礼者）；
        - 第四段 persuasion_profile_to_audience 必须解释购买者与受礼者关系，
          并引用 profile 中真实 requirement（优先 gift_context_rule 来源的礼赠类要求）。

        通用规则：对任意送礼对象/时机/意图生效，不做单样本硬编码；显式送礼/受礼者
        信号优先于品牌价格消费力规则（此分支直接覆盖机械八大人群派生）。
        """
        scene = gift_context.get("gift_scene") or "通用送礼"
        recipient = gift_context.get("gift_recipient") or "受礼者"
        demographic = gift_context.get("recipient_demographic") or ""
        decider = gift_context.get("purchase_decider") or "送礼者"
        relationship = gift_context.get("relationship") or "通用送礼"
        evidence = gift_context.get("evidence") or []

        recipient_label = f"{recipient}/{demographic}" if demographic and demographic != "不限" else recipient

        # ---- 引用 profile 中真实 requirement：优先 gift_context_rule 来源，其次 required/high ----
        requirements = list(persuasion_requirement_profile.get("persuasion_requirements") or [])

        def _req_attr(req: Any, key: str) -> Any:
            if isinstance(req, Mapping):
                return req.get(key)
            return getattr(req, key, None)

        gift_sourced = [
            r for r in requirements if "gift_context_rule" in (list(_req_attr(r, "source") or []))
        ]
        high_priority_reqs = [
            r
            for r in requirements
            if bool(_req_attr(r, "required")) or str(_req_attr(r, "priority") or "") == "high"
        ]
        if gift_sourced:
            referenced_reqs = gift_sourced
        elif high_priority_reqs:
            referenced_reqs = high_priority_reqs
        else:
            referenced_reqs = requirements[:1]
        ref_labels = [
            f"「{_req_attr(r, 'requirement_name')}({_req_attr(r, 'requirement_id')})」"
            for r in referenced_reqs[:3]
        ]

        evidence_text = "、".join(evidence[:6]) if evidence else "送礼信号"
        primary_reason = (
            f"送礼场景识别（命中：{evidence_text}）：购买决策者为「{decider}」（送礼者），"
            f"该商品作为「{scene}」礼物由其购买，故核心说服对象（primary）= 购买决策者。"
        )
        secondary_reason = (
            f"受礼者为「{recipient_label}」，是商品的实际使用者但非购买决策者，"
            f"作为次级说服对象（secondary），关系：{relationship}。"
        )
        primary_audiences = [
            AudienceGroupJudgment(audience_group=decider, fit_level="primary", reason=primary_reason)
        ]
        secondary_audiences = [
            AudienceGroupJudgment(
                audience_group=recipient_label, fit_level="secondary", reason=secondary_reason
            )
        ]

        persuasion_profile_to_audience = (
            f"送礼关系「{relationship}」：购买者（{decider}）为受礼者（{recipient_label}）选礼，"
            f"二者非同一人；需满足说服要求{'、'.join(ref_labels)}"
            f"（profile 共 {len(requirements)} 条，其中 gift_context_rule 礼赠类 {len(gift_sourced)} 条）"
            f"→ 因此核心说服对象（primary）指向购买决策者「{decider}」，受礼者「{recipient_label}」为次级（secondary）。"
        )

        reasoning_chain = AudienceReasoningChain(
            task_to_role=(
                f"唯一任务「{primary_task}」对应购买角色「{role}」；"
                f"叠加 gift_context 送礼场景，实际购买决策者收敛为「{decider}」。"
            ),
            role_category_to_age_gender=(
                f"送礼双重角色：购买者「{decider}」与受礼者「{recipient_label}」"
                f"通常非同一人/年龄/性别，故不按单一八大人群坐标机械派生。"
            ),
            brand_price_to_consumption_power=(
                "显式送礼/受礼者信号优先于品牌价格消费力规则：本分支按送礼关系直接派生人群，"
                "品牌×价格消费力仅作辅助参考，不覆盖送礼信号。"
            ),
            persuasion_profile_to_audience=persuasion_profile_to_audience,
        )

        caveats = [
            f"礼赠场景（{scene}）：送礼者≠受礼者，目标人群已区分购买决策者（{decider}）与受礼者（{recipient_label}）。",
            "gift_context 由确定性查表识别（送礼对象/时机/意图三类信号），未引入 LLM 语义推断。",
        ]

        return ProductTargetAudience(
            primary_audiences=primary_audiences,
            secondary_audiences=secondary_audiences,
            weak_fit_audiences=[],
            reasoning_chain=reasoning_chain,
            caveats=caveats,
        )

    def _match_mvp_scope_gate(
        self,
        payload: DiagnosticInput,
        module1_output: Module1Output,
        proposal: JTBDProposal,
    ) -> dict[str, Any] | None:
        """PRD-0：MVP 范围控制（scope gate）。

        裁决口径：
        - 维持“token 模糊匹配”（通过去符号/大小写归一化做近似匹配），不升级严格枚举。
        - 仅允许读取商品侧字段，禁止读取任何视频侧（ASR/OCR/VLM/video_factpack）。
        - 返回命中 trace：matched_tokens / matched_fields / jtbd_hint。

        命中时：只支持 JTBD 识别，不进入模块3/4，不产出 HEC。
        """

        # 仅商品侧字段（SSOT）。
        field_texts: dict[str, str] = {
            "leaf_category": str(payload.leaf_category or module1_output.leaf_category or ""),
            "category_path": str(payload.category_path or ""),
            "product_name": str(payload.product_name or module1_output.product_name or ""),
            "core_selling_points": str(payload.core_selling_point or module1_output.core_selling_point or ""),
            "brand_name": str(payload.brand_name or ""),
            "product_detail_summary": str(payload.product_detail_summary or ""),
        }

        # token 模糊匹配：统一使用“去标点+去空格+小写”归一化后做 substring。
        normalized_fields: dict[str, str] = {
            k: _normalize_compact_text(v) for k, v in field_texts.items() if str(v).strip()
        }

        def match_tokens(tokens: set[str]) -> tuple[list[str], list[str]]:
            hit_tokens: set[str] = set()
            hit_fields: set[str] = set()
            for token in tokens:
                t = _normalize_compact_text(token)
                if not t:
                    continue
                for field, text in normalized_fields.items():
                    if t in text:
                        hit_tokens.add(token)
                        hit_fields.add(field)
            return (sorted(hit_tokens), sorted(hit_fields))

        # 1) 类目/商品文本命中（香氛/礼盒等）
        matched_tokens, matched_fields = match_tokens(MVP_UNSUPPORTED_CATEGORY_TOKENS)

        # 2) 强情绪价值/氛围感等语义线索（不依赖前端）
        emo_tokens, emo_fields = match_tokens(MVP_UNSUPPORTED_EMOTIONAL_TOKENS)
        matched_tokens = sorted(set(matched_tokens) | set(emo_tokens))
        matched_fields = sorted(set(matched_fields) | set(emo_fields))

        if not matched_tokens:
            return None

        return {
            "scope_gate_status": "out_of_scope_for_mvp",
            "reason_code": "fragrance_gift_emotional_value_not_covered",
            "matched_tokens": matched_tokens,
            "matched_fields": matched_fields,
            "jtbd_hint": str(proposal.primary_task or "").strip(),
        }

    def _build_out_of_scope_for_mvp_status(self, trace: dict[str, Any]) -> dict[str, Any]:
        # 兼容：仍保留历史字段 status，供 metadata.downstream_status 复用。
        return {
            "status": "out_of_scope_for_mvp",
            **trace,
            "supported_stage": "jtbd_recognition_only",
            "unsupported_stage": "hec_generation",
            "user_facing_message": "当前视频属于香氛/礼盒/情绪价值方向，MVP 暂不支持完整 HEC 建议生成。",
        }

    def _build_output(
        self,
        *,
        payload: DiagnosticInput,
        module1_output: Module1Output,
        proposal: JTBDProposal,
        raw_proposal: JTBDProposal,
        category_matrix: CategoryIntentMatrix,
        product_matrix: ProductIntentMatrix,
        reasoning_path: list[str],
        warnings: list[str],
        assertions: list[str],
        gate_notes: list[str],
        product_target_audience: Optional[ProductTargetAudience] = None,
        persuasion_requirement_profile: dict | None = None,
    ) -> ProductDiagnosisOutput:
        module3_modifiers = self._derive_module3_modifiers(payload, product_matrix)
        module3_category_attr = self._to_module3_category_attr(category_matrix)
        module3_trust_attr = self._to_module3_trust_attr(product_matrix)
        module3_price_attr = self._to_module3_price_attr(product_matrix)
        category_strategy_intent = derive_category_strategy_intent(
            cognition_attribute=module3_category_attr,
            frequency_attribute=category_matrix.frequency,
        )
        product_strategy_intent = derive_product_strategy_intent(
            trust_attribute=module3_trust_attr,
            price_attribute=module3_price_attr,
        )
        # PRD-1.1：装配前置后置断言（Crash Early）——校验辅助字段结构。
        self._assert_auxiliary_fields(proposal)
        resolved_product_id = payload.product_id or f"jtbd-{proposal.primary_task}"
        assembly_status: dict[str, Any] | None = None

        # PRD-0：scope gate 命中时早停（只做 JTBD 识别，不进入模块3/4，不产出 HEC）。
        scope_gate_trace = self._match_mvp_scope_gate(payload, module1_output, proposal)
        if scope_gate_trace is not None:
            assembly_status = self._build_out_of_scope_for_mvp_status(scope_gate_trace)
            candidate_set_payload = {
                "schema_version": "v0.5",
                "jtbd": proposal.primary_task,
                "persuasion_route": "",
                "r_rule": category_strategy_intent,
                "p_rule": product_strategy_intent,
                "task_domain": proposal.domain,
                "h_list": [],
                "effect_list": [],
                "cta_list": [],
            }
            product_ec_skeletons = []
            product_hecs = []
        elif proposal.primary_task == NEEDS_REVIEW:
            # PRD-1.2 / AC12：needs_review 中止态不触发模块3寻址与模块4 HEC 装配，输出空候选集骨架。
            candidate_set_payload = {
                "schema_version": "v0.5",
                "jtbd": NEEDS_REVIEW,
                "persuasion_route": "",
                "r_rule": category_strategy_intent,
                "p_rule": product_strategy_intent,
                "task_domain": "",
                "h_list": [],
                "effect_list": [],
                "cta_list": [],
            }
            product_ec_skeletons = []
            product_hecs = []
        else:
            candidate_set = derive_candidate_set(
                Module3IntentInput(
                    jtbd=proposal.primary_task,
                    cognition_attribute=module3_category_attr,
                    frequency_attribute=category_matrix.frequency,
                    trust_attribute=module3_trust_attr,
                    price_attribute=module3_price_attr,
                    modifiers=module3_modifiers,
                )
            )
            candidate_set_payload = candidate_set.to_dict()
            try:
                product_ec_skeletons = self.variant_assembler.assemble_product_ec_skeletons(candidate_set_payload)
                product_hecs = self.variant_assembler.assemble_product_hecs(
                    proposal.primary_task,
                    product_ec_skeletons,
                    candidate_set_payload["h_list"],
                    product_id=resolved_product_id,
                )
            except ValueError as exc:
                assembly_status = self._build_assembly_blocked_status(
                    proposal.primary_task,
                    candidate_set_payload,
                    exc,
                )
                if assembly_status is None:
                    raise
                product_ec_skeletons = []
                product_hecs = []
        category_axis = category_strategy_intent.split("_", 1)[0].strip()
        product_axis = product_strategy_intent.split("_", 1)[0].strip()
        resistance_profile = {
            "domain": proposal.domain,
            "category_matrix": category_matrix.matrix_label,
            "ocean": category_matrix.ocean,
            "competition_focus": category_matrix.competition_focus,
            "frequency": category_matrix.frequency,
            "trust_barrier": product_matrix.trust_barrier,
            "financial_risk": product_matrix.financial_risk,
            "brand_tier": product_matrix.brand_tier,
            "relative_price_level": product_matrix.relative_price_level,
            "channel_risk": "有风险" if "channel_risk" in module3_modifiers else "无风险",
            "endorsement": "有背书" if "has_endorsement" in module3_modifiers else None,
        }
        core_intent = {
            "category_intent": category_matrix.category_intent,
            "product_intent": product_matrix.product_intent,
            "category_strategy_intent": category_strategy_intent,
            "product_strategy_intent": product_strategy_intent,
            "intent_pair_key": f"{category_axis}×{product_axis}",
            "core_e": candidate_set_payload["effect_list"],
            "core_c": candidate_set_payload["cta_list"],
            "candidate_h": candidate_set_payload["h_list"],
            "primary_effect": self._strategy_primary_code(candidate_set_payload["effect_list"]),
            "primary_cta": self._strategy_primary_code(candidate_set_payload["cta_list"]),
        }
        evidence = {
            "input": payload.to_dict(),
            "module1_output": module1_output.to_dict(),
            "llm_proposal": raw_proposal.dict(exclude_none=True),
            "gated_proposal": proposal.dict(exclude_none=True),
            "hard_gate_notes": gate_notes,
            "category_intent_matrix": category_matrix.dict(exclude_none=True),
            "product_intent_matrix": product_matrix.dict(exclude_none=True),
            "module3_context": {
                "candidate_set": candidate_set_payload,
            },
            "product_ec_skeletons": product_ec_skeletons,
            "product_hecs": product_hecs,
            "assembly_status": assembly_status,
            "keyword_rule_traces": list(self._keyword_rule_traces),
        }
        metadata = {
            "engine": "ProductDiagnosisEngine",
            "module": "commerce_video_diagnosis/understanding/engines/product_diagnoser.py",
            "architecture": "product_diagnosis_v3_jtbd_four_stage",
            "classifier_mode": "mock_or_injected" if self.classifier is not None else "llm_default",
            "max_retries": self.max_retries,
            "keyword_rule_trace_count": len(self._keyword_rule_traces),
            # F5：透出 profile 的 content_goal 口径（P0 默认 purchase）。优先取引擎产出 profile 内的真实值，
            # 缺失时回落集中常量；便于下游/前端判别本次说服档案的目标口径。
            "content_goal": (persuasion_requirement_profile or {}).get(
                "content_goal", DEFAULT_PRODUCT_UNDERSTANDING_CONTENT_GOAL
            ),
        }
        if proposal.primary_task == NEEDS_REVIEW:
            # PRD-1.2 / AC12：标记模块3/4 因 needs_review 跳过装配。
            metadata["downstream_status"] = "skipped"
            metadata["downstream_skip_reason"] = "jtbd_insufficient"
        if assembly_status is not None:
            metadata["downstream_status"] = assembly_status["status"]
            metadata["downstream_block_reason_code"] = assembly_status["reason_code"]
            warnings.append(assembly_status["user_facing_message"])
        return ProductDiagnosisOutput(
            product_id=resolved_product_id,
            leaf_category=module1_output.leaf_category,
            shop_name=module1_output.shop_name,
            product_name=module1_output.product_name,
            price=self._parse_price(module1_output.price),
            domain=proposal.domain,
            primary_task=proposal.primary_task,
            sub_task=proposal.sub_task,
            category_intent=category_matrix.category_intent,
            product_intent=product_matrix.product_intent,
            category_intent_matrix=category_matrix,
            product_intent_matrix=product_matrix,
            reasoning_path=reasoning_path,
            warnings=warnings,
            category=module1_output.leaf_category,
            jtbd=proposal.primary_task,
            resistance_profile=resistance_profile,
            core_intent=core_intent,
            candidate_set=candidate_set_payload,
            product_ec_skeletons=product_ec_skeletons,
            product_hecs=product_hecs,
            assembly_status=assembly_status,
            assertions=assertions,
            evidence=evidence,
            metadata=metadata,
            product_target_audience=product_target_audience,
            # F5：persuasion_requirement_profile 由引擎内部产出并随主输出一并装配（不再由外部 pipeline 后挂）。
            # pydantic v1 会自动把 dict 强校验构造为 Optional[PersuasionRequirementProfile]。
            persuasion_requirement_profile=persuasion_requirement_profile,
            # PRD-1.1：辅助字段随主任务一并装配到商品事实向量输出。
            secondary_benefits=list(proposal.secondary_benefits),
            non_selected_task_reasons=[dict(item) for item in proposal.non_selected_task_reasons],
        )

    def _build_assembly_blocked_status(
        self,
        jtbd_level1: str,
        candidate_set_payload: dict[str, Any],
        exc: ValueError,
    ) -> dict[str, Any] | None:
        """仅对“武器库真实不可表达”返回协议化 blocked；其他 Crash Early 继续抛出。"""
        message = str(exc)
        if "准入失败后无可用降级目标" not in message:
            return None
        if jtbd_level1 != "自我犒赏与秩序掌控":
            return None
        return {
            "status": "assembly_blocked",
            "reason_code": "no_expressible_cta_after_admission",
            "jtbd_level1": "自我犒赏",
            "route_context": "R02xP03",
            "blocked_stage": "module4_cta_admission",
            "evidence": [
                "CTA C4 admission failed",
                "no fallback CTA available after strict intersection",
            ],
            "user_facing_message": "当前已识别为情绪/感官价值方向，但现有表达武器库缺少可用 CTA 组合，暂无法生成完整 HEC。",
        }

    def _assert_auxiliary_fields(self, proposal: JTBDProposal) -> None:
        """PRD-1.1 装配后置断言（Crash Early）：校验 secondary_benefits / non_selected_task_reasons 结构合法。"""
        for task in proposal.secondary_benefits:
            if task not in VALID_JTBD:
                raise ValueError(f"secondary_benefits 含非法任务标签（须为 12 标签枚举值之一）: {task}")
        for item in proposal.non_selected_task_reasons:
            if not isinstance(item, dict):
                raise ValueError(f"non_selected_task_reasons 元素必须是 dict: {item!r}")
            if "task" not in item or "reason" not in item or "gate" not in item:
                raise ValueError(f"non_selected_task_reasons 元素缺少 task/reason/gate 键: {item!r}")
            if item["gate"] not in {"blocked", "dropped", "demoted_to_secondary"}:
                raise ValueError(f"non_selected_task_reasons 的 gate 非法（须为 blocked/dropped/demoted_to_secondary）: {item['gate']!r}")

    def _is_ordinary_daily_category(self, module1_output: Module1Output) -> bool:
        text = f"{module1_output.leaf_category} {module1_output.product_name} {module1_output.core_selling_point}"
        return self._contains_any(text, ORDINARY_DAILY_TOKENS)

    def _has_strong_anxiety_main_chain(self, module1_output: Module1Output) -> bool:
        text = self._module1_joined_text(module1_output)
        return self._contains_any(text, GR03_STRONG_ANXIETY_MAIN_CHAIN_TOKENS)

    def _is_high_premium(self, payload: DiagnosticInput) -> bool:
        text = payload.joined_text()
        engine_level = str((payload.engine_node or {}).get("relative_price_level") or "").strip()
        return engine_level == "高水位" or self._contains_any(text, EMOTIONAL_PREMIUM_TOKENS)

    def _is_fragrance_sensory_value(self, module1_output: Module1Output, text: str) -> bool:
        # P0-2：香氛品类(香水/香氛/淡香/香精...) 且命中强感官主价值信号 → 感官价值主诉成立。
        category_text = f"{module1_output.leaf_category} {module1_output.product_name} {module1_output.core_selling_point}"
        return self._contains_any(category_text, FRAGRANCE_CATEGORY_TOKENS) and self._contains_any(
            text, FRAGRANCE_SENSORY_VALUE_TOKENS
        )

    def _fallback_functional_task(self, module1_output: Module1Output) -> str:
        text = self._module1_joined_text(module1_output)
        if self._is_physical_safety_fact(text):
            return "物理安全与风险规避"
        if self._has_strong_defect_signal(module1_output, text):
            return "缺陷修复/冲突消除"
        if self._contains_any(text, EFFICIENCY_TOKENS.union(OPERATION_EASE_TOKENS)):
            return "降本增效/懒人替代"
        if self._supports_maintenance_task(module1_output, text):
            return "生存/运转维系"
        return "生存/运转维系"

    def _parse_price(self, raw_value: str) -> float:
        price_value = self._safe_parse_price(raw_value)
        if price_value <= 0:
            raise ValueError("无法解析合法价格。")
        return price_value

    def _safe_parse_price(self, raw_value: str) -> float:
        text = str(raw_value or "")
        match = re.search(r"\d+(?:\.\d+)?", text)
        if not match:
            return 0.0
        try:
            return float(match.group())
        except ValueError:
            return 0.0

    def _join_reasoning(self, proposal: JTBDProposal) -> str:
        return " ".join([proposal.reasoning] + list(proposal.reasoning_path))

    def _proposal_has_physical_safety_evidence(self, proposal: JTBDProposal) -> bool:
        if self._is_physical_safety_fact(self._join_reasoning(proposal)):
            return True
        for evidence in proposal.evidence_chain:
            evidence_text = str((evidence or {}).get("evidence_text") or "")
            if self._is_physical_safety_fact(evidence_text):
                return True
        return False

    def _contains_any(self, text: str, tokens: set[str] | list[str]) -> bool:
        return any(token and token in text for token in tokens)


DiagnosticEngine = ProductDiagnosisEngine


__all__ = [
    "AudienceGroupJudgment",
    "AudienceReasoningChain",
    "CategoryIntentMatrix",
    "DURABLE_BUSINESS_CATEGORIES",
    "DiagnosticEngine",
    "DiagnosticInput",
    "JTBDLLMClassifier",
    "JTBDProposal",
    "ProductDiagnosisEngine",
    "ProductDiagnosisOutput",
    "ProductIntentMatrix",
    "ProductTargetAudience",
    "RULE_TABLE",
    "TASK_DOMAIN_MAP",
    "_build_price_band_lookup",
    "_load_brand_whitelist",
    "_load_price_band_dict",
]
