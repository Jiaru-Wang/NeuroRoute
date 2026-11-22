# 正式执行协议批准与执行边界

本记录汇总所有者在本次对话中的明确确认及委托决定，不由程序自动推断批准。
数据、增强、区域、预算分别见DATA_PROTOCOL_APPROVAL.md、
DEVELOPMENT_PROTOCOL_APPROVAL.md及FINAL_DESIGN_PARTIAL_APPROVAL.json。
统计方案经所有者明确要求“给出专业推荐然后你决定确认并且执行”，助手选择并记录：
logs/protocol/statistics-approved-y71cftan。其余服务器记录路径见前述JSON。

## 执行配置

- configs/protocol.approved.json：已批准主协议、融合、六测试条件及主模型机制计划。
- configs/ablations.approved.json：单因素消融；3个随机区域+K/λ/η各2个非基准设置。
- configs/statistics.approved.json：六项主比较，C3+C4下四类Macro-F1。
- configs/data.approved.json：此前批准的独立数据准备。preprocessing必须与主协议一致。

保留review文件未批准状态作为历史候选，正式命令使用approved文件。
批准配置不自动启动作业，外层测试必须在训练/checkpoint清单锁定后另行获授权。

## 统计定义

差值方向为candidate-reference。每被试平均42/123/3407三个种子，再分析9个差值。
六比较同属一个Holm校正族：B1-B0、Spatial-EGS-B1、Spatial-EGS-matched_mlp、
Spatial-EGS-mask_moe、shared_hm-shared_h、Spatial-EGS-shared_hm；最后一项为核心比较。
双侧精确符号翻转枚举512种组合；bootstrap按被试配对重采样10000次，seed42，
95%百分位区间（逐比较未作同时区间校正，不冒称家族置信覆盖）。
报告原始分数差及百分点、配对Cohen dz；差值标准差为0时dz标null，不伪造无穷效应。
九被试小样本、差值对称性假设、LOSO训练集重叠限制必须报告。

消融与机制为次要探索性分析，不能混用主比较程序的confirmatory标签。
随机区域需先每被试平均训练种子，再平均三个分组，另保留每分组结果。
目前paired_subject_statistics实现主比较；随机区域聚合/探索性报告尚需后续独立实现
和回归，在分析阶段之前完成，不能将三个分组当27名被试或冒称现已完成该分析。

## 作业与机制范围

主比较216个训练作业。单因素消融9个新增配置×27作业=243，合计459。
三个轴中的基准值复用主配置，计划manifest记录reused_main_config。
三个随机分组保持6/6/4/6，种子101/202/303，全部报告，不挑最差分组。
机制仅主比较路由模型的锁定后六条件；消融生成配置的机制审批置false，
避免把主配置K=2的固定专家编号误用于K=1/3。不是跳过失败测试。
完整输入捐赠含额外信息，不是部署主基线；计算路径干预不证明脑区生理因果。

## 修复及证据有效性

审查发现旧消融生成器会重跑三个基准设置，产出12个而非9个新增配置。
先用合成测试复现，再修复为复用基准；没有已执行的消融结果需要作废。
统计新增百分点输出，不改变检验公式。没有真实预测用于开发统计测试。
数据预处理及模型源码不改，9f4336a真实缓存与九折统计报告继续作为该版本证据。
后续GPU验收需在当前代码版本重新生成；不得篡改旧报告的提交信息。
