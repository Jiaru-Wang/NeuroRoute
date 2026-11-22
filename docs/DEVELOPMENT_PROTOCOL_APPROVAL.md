# 开发协议局部批准（2026-11-22）

依据本次对话逐项授权记录；精确数值快照见 DEVELOPMENT_PROTOCOL_APPROVAL.json。
数据协议继续遵循 DATA_PROTOCOL_APPROVAL.md。整份实验协议仍未冻结，不能启动训练。

## 已批准

- D04：pdf_four_branch，完整/C3单缺/C4单缺/其他各25%；其他分支在排除
  C3/C4的20电极中，等概率选缺1或2个，给定数量均匀抽样。
  每Trial每epoch一个视图，view_id=0；身份键含种子、折、epoch、会话、原Trial ID。
  B0只用完整输入；其他增强模型共享掩码。不会产生C3或C4与其他电极的联合缺失，
  也不会产生C3+C4及其超集；不宣称覆盖全部合法缺失集合。
- D05成员：front=Fz,FC3,FC1,FCz,FC2,FC4；left=C5,C3,C1,CP3,CP1,P1；
  mid=Cz,CPz,Pz,POz；right=C2,C4,C6,CP2,CP4,P2。
  此为新增几何提案经所有者批准，不是PDF已固定的生理功能脑区。
  随机区域保持6/6/4/6并重新训练；重复数、分组种子和额外预算仍待定。
- D06：完整/C3单缺/C4单缺各自在完整验证分区计算固定4类Macro-F1，
  然后三者等权平均。严格提高才更新最佳checkpoint，同分留更早epoch。
- D07：batch16、最多50epochs、patience10、min_delta0、AdamW、lr0.0003、
  weight_decay0.01；有Router模型的importance系数0.01，无Router模型不添加。
  种子42/123/3407；保留尾batch；本轮不增加超参数搜索。
  相同上限与早停规则不代表实际更新次数或GPU时间相同，须分别报告。

## 授权和实验边界

增强由所有者委托助手决定；区域与开发预算分别获得明确“确认”。
以上为预先固定的本轮方案，不是经性能验证的最优值。
先单折单种子工程验收，再推进获批完整实验；工程验收不减少正式九折三种子范围。
普通MoE融合、额外最终缺失模式、机制、消融预算与统计方案仍待审。
不得仅因本记录存在就把候选配置approved设为true，不自动开放外层测试。

## 对应已有实测

用户提供b127778服务器GPU套件49项通过（含CPU测试），失败/错误/跳过均0：
logs/checks/gpu-unit-7riZFxY6/report.json。
真实官方权重严格加载及合成GPU冻结/遮蔽检查通过：
logs/checks/official-backbone-0ka_lwgj/report.json。
这些为用户回传证据；本地没有读取服务器报告原文件，不冒称正式真实EEG训练通过。
本次只新增文档及批准快照，不改变运行源码、候选配置数值或uv.lock。
旧报告保留原提交归属；未来训练入口如要求当前提交匹配，应在最终代码冻结后
重跑相应验收，不得手改旧报告的source_commit。
