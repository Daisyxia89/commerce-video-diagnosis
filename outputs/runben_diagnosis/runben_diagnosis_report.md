# 润本驱蚊液真实视频诊断报告 (runben_repellent_real_video)

## 1. 视频元信息
- **视频 ID**: `douyin_7643812223105994217`
- **来源**: 真实短视频 (抖音)
- **视频文件**: `outputs/runben_diagnosis/raw_video.mp4`
- **视频 URL**: https://www.douyin.com/video/7643812223105994217
- **时长**: 约 01:14
- **case_id**: `runben_repellent_real_video`
- **runtime_mode**: `agent_native_runtime`
- **E2E 状态**: `passed`
- **验收状态**: `PASS`

## 2. 商品理解摘要
- **主任务**: 物理安全与风险规避 (功能域)
- **JTBD**: 物理安全与风险规避
- **商品主人群**: 年长中高消费力女性
- **次级人群**: 年长低消费力女性
- **必讲要求数**: 11 条
- **非必讲要求数**: 1 条 (`clarify_purchase_threshold`)
- **商品推荐 HEC**: `H1/E1/C4`
  - H1 痛点/焦虑直击
  - E1 效果测评
  - C4 人群/场景总结
- **商品侧关键信息**: 润本驱蚊液；15%/20% 派卡瑞丁 A 级驱蚊力，最长 8h 防蚊；强调温和、安全、检测/备案背书。

## 3. 视频理解摘要

### 3.1 分镜表
| 段落 | 时间 | 角色 | 画面摘要 | ASR/OCR 摘要 | 证据类型 |
|---|---:|---|---|---|---|
| seg_hook_1 | 0.0-12.0s | hook | 主持人手持驱蚊水，宣布 700 只蚊子 15 分钟挑战。 | 口播/字幕：`700只的蚊子待15分钟挑战`；提到今年蚊子繁殖快、又凶又猛。 | visual / speech / subtitle |
| seg_effect_1 | 12.0-36.0s | effect | 全副武装投放蚊子，主持人进入玻璃房实测，蚊虫飞动但无法近身。 | 口播：`这个蚊子真的很多！但是我的周围真的很清静`；字幕：关门放蚊、15分钟啦。 | visual / speech / subtitle |
| seg_effect_2 | 36.0-50.0s | effect | 主持人走出实验室展示手臂，强调没有红肿蚊包。 | 口播/字幕：`一个蚊子包都没有`；同时展示蚊子依旧鲜活。 | visual / speech / subtitle |
| seg_cta_1 | 50.0-74.0s | cta | 介绍核心成分和大瓶/小瓶规格，演示喷雾与报告类画面。 | 口播/字幕：`15%浓度的羟哌酯`、`8h强效防蚊 7h耐汗`、`0香精不呛鼻`。 | visual / speech / subtitle |

### 3.2 Evidence Spans
| span_id | text |
|---|---|
| ev_challenge | 700只的蚊子待15分钟挑战 |
| ev_no_bite | 一个蚊子包都没有 |
| ev_ingredient | 15%浓度的羟哌酯 |
| ev_long_effect | 8h强效防蚊 7h耐汗 |
| ev_mild | 0香精不呛鼻 |

### 3.3 视频 HEC
- **视频 HEC**: `H5/E1/C4`
- **Signature**: H5猎奇挑战 → E1效果实测 → C4场景总结推荐
- **判定理由**: 视频以“700只蚊子挑战”的极端安全/猎奇场景开场，主体用实验室实测和无包结果证明效果，结尾通过大瓶居家、小瓶出门完成场景化推荐。

### 3.4 Slider 四轴
| 轴 | score | score_0_10 | label | evidence | reasoning |
|---|---:|---:|---|---|---|
| visual | 0.90 | 9.0 | 视觉实测冲击 | 实验室蚊子飞舞场景及无包手臂特写 | 画面实测感强，视觉证据直接且有冲击力。 |
| audio | 0.85 | 8.5 | 节奏紧凑讲解清晰 | 主持人口播节奏明快，语气坚定且有亲和力 | 音频表达清晰，配合挑战氛围有效吸引注意力。 |
| proof | 0.95 | 9.5 | 高信度实测证据 | 15分钟计时器 + 玻璃房蚊虫环绕 + 最终结果展示 | 全过程证据链完整，说服力极强。 |
| cta | 0.80 | 8.0 | 多规格场景化收口 | 区分大瓶居家和小瓶出门，并结合成分背书 | 收口建议具体，解决“买哪款”和“为什么买”。 |

## 4. 四类诊断结论与理由

### 4.1 Audience Match
- **状态**: `high_match`
- **理由**: 商品主目标为“年长中高消费力女性”；视频通过驱蚊/防蚊场景、温和成分、检测报告和实测效果，推断视频主目标同样为“年长中高消费力女性”。

### 4.2 Profile Match
- **状态**: `partial`
- **理由**: 11 条必讲要求中大多数被实测、成分与温和表达覆盖；但 `provide_authority_endorsement` 仅部分覆盖，视频有检测报告画面或 8h/7h 字幕，但没有充分展开具体报告编号、机构或资质。

### 4.3 HEC Match
- **状态**: `acceptable_deviation`
- **商品预期 HEC**: `H1/E1/C4`
- **视频实际 HEC**: `H5/E1/C4`
- **理由**: effect 与 cta 完全匹配；hook 从“痛点/焦虑直击”偏向“安全场景/猎奇挑战”，但对于驱蚊产品仍属可接受偏差。

### 4.4 Slider Match
- **状态**: `too_strong`
- **理由**: 视觉 0.90、音频 0.85 超出目标人群偏好区间；proof 0.95 与 cta 0.80 符合区间。主要风险是挑战画面与快节奏表达对年长中高消费力女性略显刺激。

### 4.5 Overall
- **状态**: `needs_minor_repair`
- **关键发现**:
  - 人群匹配：`high_match`
  - 说服要求覆盖：`partial`
  - HEC 匹配：`acceptable_deviation`
  - Slider 匹配：`too_strong`

## 5. Profile 逐条覆盖表
| requirement_id | completion_status | matched_evidence_spans | judgment |
|---|---|---|---|
| expose_current_pain | completed | ev_challenge | 通过 700 只凶猛蚊子挑战，直接具象化蚊虫叮咬痛点。 |
| prove_user_fit | completed | ev_mild | “0香精不呛鼻”暗示成分温和，间接证明对家庭成员及敏感人群的适配性。 |
| prove_scenario_fit | completed | ev_long_effect | 区分居家大瓶和出门小瓶，并通过“7h耐汗”覆盖主要使用场景。 |
| prove_core_benefit | completed | ev_no_bite | “一个蚊子包都没有”直接证明防蚊核心收益。 |
| provide_visible_result | completed | ev_no_bite | 手臂无红肿特写提供可视化结果。 |
| establish_basic_trust | completed | ev_ingredient | 明确“15%浓度的羟哌酯”，用专业成分建立基础信任。 |
| reduce_trial_risk | completed | ev_mild | “0香精不呛鼻”降低对刺激性的担忧。 |
| prove_source_credibility | completed | ev_ingredient | 成分浓度和驱蚊理据讲解提升可信度。 |
| provide_authority_endorsement | partial | ev_long_effect | 有“8h强效防蚊 7h耐汗”等证据，但未充分口播具体权威机构或报告编号。 |
| resolve_safety_risk | completed | ev_mild | 通过成分讲解和气味描述消解化学驱蚊安全顾虑。 |
| prove_current_purchase_reason | completed | ev_challenge | 面对蚊子“又凶又猛”的繁殖高峰，给出当下购买理由。 |
| clarify_purchase_threshold | not_applicable | - | 非必讲；视频未具体说明价格、优惠或下单门槛。 |

## 6. 修复建议
1. **补强权威背书**: 在 CTA 段增加一句可验证表达，例如“第三方检测显示 8h 强效防蚊、7h 耐汗”，如允许可补充报告编号或备案信息。
2. **收敛视觉刺激强度**: 700 只蚊子挑战可保留，但减少蚊群惊吓镜头时长，把重点转向“无包结果”和“家人安心使用”。
3. **音频节奏略降速**: 对目标人群可用更稳的讲解语速，突出安全、温和、长期保护。
4. **补充购买门槛**: 结尾增加“选大瓶/小瓶”的规格价格、套装或优惠说明，降低下单阻力。

## 7. Runtime Trace 摘要
| module | inference_source | status | summary |
|---|---|---|---|
| input | deterministic_assembler | start | case_id=runben_repellent_real_video |
| product_understanding | rule_based_fallback | ok | 主任务=物理安全与风险规避；主人群=['年长中高消费力女性']；必讲要求=11 条；HEC 首选=H1/E1/C4。 |
| video_understanding | agent_native_runtime | ok | storyboard=4 段；slider={'visual': 0.9, 'audio': 0.85, 'proof': 0.95, 'cta': 0.8}；video_hec={'hook_tag': 'H5', 'effect_tag': 'E1', 'cta_tag': 'C4'} |
| diagnosis | rule_based_fallback | ok | overall=needs_minor_repair；audience=high_match；profile=partial(agent_native)；hec=acceptable_deviation；slider=too_strong |
| assemble | deterministic_assembler | passed | e2e_status=passed |

## 8. HEC 体系说明
HEC 是电商视频的三段式表达框架：

### H: Hook（开场钩子）
负责在前几秒吸引注意力、定义用户问题或场景。
- **H1 痛点/焦虑直击**: 直接指出痛点、风险、损失。
- **H5 安全场景/猎奇**: 用极端、实验、安全风险或强视觉场景吸引注意。
- **H6/H7 等**: 通常用于其他兴趣、身份或情绪切入方式，具体以引擎标签库为准。

### E: Effect（中段利益/证明）
负责证明商品“确实有效”。
- **E1 效果测评**: 通过实测、对比、实验或可见结果证明效果。
- **E0 卖点陈述**: 直接讲参数、成分、功能。
- **E3 对比/拉踩**: 与旧方案、同类或竞品做对比，突出优势。
- **E6**: 常用于更复杂的风险/信任/机制证明表达，具体以引擎标签库为准。

### C: CTA（行动收口）
负责把用户兴趣转化为购买行动或规格选择。
- **C3 指令行动**: 明确“点链接/下单/拍几件”等强行动指令。
- **C4 人群/场景总结**: 告诉用户“谁适合、什么场景用、怎么选规格”。

本视频属于：**H5 猎奇挑战 → E1 效果实测 → C4 场景总结推荐**。
