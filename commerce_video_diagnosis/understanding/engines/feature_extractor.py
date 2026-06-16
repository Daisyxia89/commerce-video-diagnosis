from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import requests
from commerce_video_diagnosis.understanding.llm_provider import build_chat_headers, require_llm_config, resolve_llm_config


GENERIC_TARGET_PEOPLE_TOKENS = {"大众", "所有人", "全部人", "年轻人", "女生", "女性", "宝妈", "人群广泛", "全网"}
SCENE_STRIP_TOKENS = {"下班后", "深夜", "通勤时", "上班时", "出门前", "约会时", "居家时"}
MARKETING_TOKENS = {"王炸", "天花板", "真香", "绝绝子", "神器", "重磅", "顶配", "神仙", "闭眼入"}
FORBIDDEN_INFERENCE_KEYS = {
    "jtbd",
    "core_task",
    "task",
    "认知状态",
    "cognition_state",
    "消费频次",
    "frequency_type",
    "品类策略意图",
    "商品策略意图",
    "strategy_intent",
    "hec",
    "hook",
    "effect",
    "cta",
}


@dataclass(slots=True)
class ProductFeatureInput:
    leaf_category: str
    shop_name: str
    product_name: str
    price: str
    core_selling_point: str
    product_id: str = ""

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ProductFeatureInput":
        def pick(*keys: str) -> str:
            for key in keys:
                if key in payload and payload[key] is not None:
                    return str(payload[key]).strip()
            return ""

        return cls(
            leaf_category=pick("leaf_category", "叶子类目", "category", "类目"),
            shop_name=pick("shop_name", "店铺", "店铺名称", "shop"),
            product_name=pick("product_name", "商品名", "商品名称"),
            price=pick("price", "价格", "售价"),
            core_selling_point=pick("core_selling_point", "核心卖点"),
            product_id=pick("product_id", "商品ID"),
        )

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(slots=True)
class ProductFeatureExtractionResult:
    leaf_category: str
    shop_name: str
    product_name: str
    price: str
    core_selling_point: str
    target_people: str
    differentiator: str
    product_id: str = ""
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ProductFeatureExtractor:
    """V3 架构图·模块一：前端特征提取器。

    只做无损清洗与事实补齐，禁止越权推导 JTBD、认知状态、消费频次及任何说服意图。
    """

    def __init__(
        self,
        *,
        model: str = "doubao-1.5-pro-32k-250115",
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: int = 60,
        llm_tag: str = "product_feature_extractor_v3",
    ) -> None:
        self.model = model
        self.llm_config = resolve_llm_config(base_url=base_url, api_key=api_key, model=model, timeout=timeout)
        self.model = self.llm_config.model
        self.base_url = self.llm_config.endpoint
        self.api_key = self.llm_config.api_key
        self.timeout = self.llm_config.timeout
        self.llm_tag = llm_tag

    def extract(self, payload: ProductFeatureInput | Mapping[str, Any]) -> ProductFeatureExtractionResult:
        if not isinstance(payload, ProductFeatureInput):
            payload = ProductFeatureInput.from_payload(payload)
        self._assert_input(payload)

        messages = self._build_messages(payload)
        raw = self._call_llm(messages)
        parsed = self._parse_json(raw)
        self._assert_no_forbidden_keys(parsed)

        result = ProductFeatureExtractionResult(
            leaf_category=payload.leaf_category,
            shop_name=payload.shop_name,
            product_name=payload.product_name,
            price=payload.price,
            core_selling_point=payload.core_selling_point,
            target_people=self._clean_text(parsed.get("target_people", "")),
            differentiator=self._clean_text(parsed.get("differentiator", "")),
            product_id=payload.product_id,
            metadata={
                "engine": "ProductFeatureExtractor",
                "architecture": "v3_module_1_frontend_feature_extractor",
                "model": self.model,
                "constraint_profile": "module_1_fact_only",
                "prompt_version": "v2",
            },
        )
        self._assert_output(result)
        return result

    def _build_messages(self, payload: ProductFeatureInput) -> list[dict[str, str]]:
        system = (
            "你是商品诊断引擎 V3 架构中的【模块一：前端特征提取器】。\n"
            "你的唯一任务：基于输入的基础商品事实，做无损清洗与标准化结构特征提取。\n"
            "你只能补齐 2 个字段：目标人群、差异化卖点。\n"
            "绝对禁止输出或暗示以下内容：JTBD、核心任务、认知状态、消费频次、品类策略意图、商品策略意图、HEC、Hook、Effect、CTA、任何说服路径结论。\n"
            "只允许保留事实级、商品级信息，不做深层营销推理。\n"
            "加工规则（强制执行）：\n"
            "1. 目标人群：不能空泛写成“大众”，必须落到具体痛点画像或人群特质；强制剥离表现层场景描述（如下班后、通勤时等时间/场景词）。\n"
            "2. 差异化卖点：必须执行强迫对比思维（提炼出相对旧方案/同类产品多了什么、改了什么）；禁用营销词（如王炸、天花板、绝绝子等），还原为客观参数与事实。\n"
            "3. 锁定主人群：若输入信息中涵盖多个人群，必须先锁定唯一的、最核心的主人群进行输出。\n"
            "4. 功能翻译：若原卖点仅为干巴巴的功能罗列，必须翻译成用户语言（即：解决了什么具体问题、替代了什么旧动作/旧方案）。\n"
            "5. 如果输入不足，只能做保守补齐，不得编造夸张信息。\n"
            "6. 输出必须是严格 JSON，不要输出任何额外解释。"
        )
        user_payload = {
            "product_id": payload.product_id,
            "leaf_category": payload.leaf_category,
            "shop_name": payload.shop_name,
            "product_name": payload.product_name,
            "price": payload.price,
            "core_selling_point": payload.core_selling_point,
            "output_schema": {
                "target_people": "string",
                "differentiator": "string",
            },
        }
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False, indent=2)},
        ]

    def _call_llm(self, messages: list[dict[str, str]]) -> str:
        require_llm_config(self.llm_config, purpose="模块一特征提取模型")
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

    def _assert_input(self, payload: ProductFeatureInput) -> None:
        for field_name, value in payload.to_dict().items():
            if field_name == "product_id":
                continue
            if not str(value).strip():
                raise ValueError(f"模块一输入缺少必填字段：{field_name}")

    def _assert_no_forbidden_keys(self, payload: Any) -> None:
        if isinstance(payload, Mapping):
            for key, value in payload.items():
                if str(key).strip() in FORBIDDEN_INFERENCE_KEYS:
                    raise AssertionError(f"模块一越权输出非法字段：{key}")
                self._assert_no_forbidden_keys(value)
        elif isinstance(payload, list):
            for item in payload:
                self._assert_no_forbidden_keys(item)

    def _assert_output(self, result: ProductFeatureExtractionResult) -> None:
        if result.target_people in GENERIC_TARGET_PEOPLE_TOKENS:
            raise AssertionError(f"目标人群过于空泛：{result.target_people}")
        if any(token in result.target_people for token in SCENE_STRIP_TOKENS):
            raise AssertionError(f"目标人群混入场景词，未完成剥离：{result.target_people}")
        if any(token in result.differentiator for token in MARKETING_TOKENS):
            raise AssertionError(f"差异化卖点含营销词，未完成事实化清洗：{result.differentiator}")
        serialized = json.dumps(result.to_dict(), ensure_ascii=False)
        for token in FORBIDDEN_INFERENCE_KEYS:
            if f'"{token}"' in serialized:
                raise AssertionError(f"模块一结果中混入越权推导字段：{token}")
        if any(keyword in serialized for keyword in ["JTBD", "核心任务", "认知状态", "消费频次", "说服意图", "HEC"]):
            raise AssertionError("模块一结果文本中出现越权推导痕迹。")

    def _clean_text(self, value: Any) -> str:
        text = "" if value is None else str(value).strip()
        text = re.sub(r"\s+", " ", text)
        return text


__all__ = [
    "ProductFeatureInput",
    "ProductFeatureExtractionResult",
    "ProductFeatureExtractor",
]
