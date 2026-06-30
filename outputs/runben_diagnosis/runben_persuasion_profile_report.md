# 润本驱蚊液 · Persuasion Profile Report

## 1. 商品与诊断概览

- 商品：【A级驱蚊力】润本驱蚊液防蚊喷雾派卡瑞丁驱蚊水防蚊叮蚊怕花露水
- 店铺：润本官方旗舰店
- 叶子类目：宝宝防蚊水
- 主任务（JTBD level1 / level2）：功能域 / 物理安全与风险规避
- category_resistance.rule：`红海-核心 × 快消`
- category_resistance.summary：在红海-核心、快消的品类竞争态势下组织购买判断主路径。
- product_conversion_barrier.rule：`极低 × 高水位`

## 2. CandidateSet

- jtbd：物理安全与风险规避
- r_rule：R02_存量同类替换
- p_rule：P02_价值证明
- task_domain：functional
- persuasion_route：先把客观风险说清，再证明当前商品如何通过更稳妥的路径降低真实受伤或事故风险。

### 2.1 候选 H 库（candidate_set.h_list，4 条）

| code | label | hook_tag |
| --- | --- | --- |
| H1 | H1 痛点/焦虑直击 | H1 |
| H5 | H5 反常识与悬念 | H5 |
| H6 | H6 场景/人群代入 | H6 |
| H7 | H7 明星/权威同款 | H7 |

### 2.2 Core E-list（candidate_set.effect_list，2 条）

| code | label | effect_tag | completion_capabilities |
| --- | --- | --- | --- |
| E1 | E1 效果测评 | E1 | functional_proof_complete |
| E6 | E6 成分/参数科普 | E6 | functional_proof_complete, identity_binding_complete |

### 2.3 Core C-list（candidate_set.cta_list，1 条）

| code | label | cta_tag | close_strength | fallback_priority |
| --- | --- | --- | --- | --- |
| C4 | C4 人群/场景总结 | C4 | passive_close | C3, C1, C2 |

## 3. product_ec_skeletons（EC 主链，共 2 条）

| # | effect_tag | cta_tag | effect_label | cta_label | cta_resolution |
| --- | --- | --- | --- | --- | --- |
| 1 | E1 | C4 | E1 效果测评 | C4 人群/场景总结 | direct |
| 2 | E6 | C4 | E6 成分/参数科普 | C4 人群/场景总结 | direct |

## 4. product_hecs（HEC variants，共 8 条）

| # | variant_id | hook_tag | effect_tag | cta_tag | risk_flags |
| --- | --- | --- | --- | --- | --- |
| 1 | runben_repellent_24p9-v1 | H1 | E1 | C4 |  |
| 2 | runben_repellent_24p9-v2 | H1 | E6 | C4 |  |
| 3 | runben_repellent_24p9-v3 | H5 | E1 | C4 |  |
| 4 | runben_repellent_24p9-v4 | H5 | E6 | C4 |  |
| 5 | runben_repellent_24p9-v5 | H6 | E1 | C4 |  |
| 6 | runben_repellent_24p9-v6 | H6 | E6 | C4 |  |
| 7 | runben_repellent_24p9-v7 | H7 | E1 | C4 |  |
| 8 | runben_repellent_24p9-v8 | H7 | E6 | C4 |  |
