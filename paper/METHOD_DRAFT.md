# NeuroRoute 方法与实验协议草稿

**状态：方法草稿；具体协议待所有者批准，未有新项目真实实验结果。**

暂定标题：NeuroRoute: Spatial Missingness-Conditioned Mixture of Experts on Pretrained EEG Representations for Robust Motor Imagery Decoding。

## 研究动机

本研究考察运动想象解码中空间电极缺失与跨被试变化共同存在时，显式缺失条件化是否改善预训练表征的下游适应。空间覆盖是可用性mask的确定性摘要，不构成额外观测。创新性和有效性需要既有文献比较及严格实验支持。

## 方法与实现

输入、冻结CBraMod、可用电极池化、专家、Router、Top-K、共享融合、Importance损失的完整公式及每个变量定义见 `docs/IMPLEMENTATION_CONTRACTS.md`，代码对应 `models.py`、`missingness.py`、`engine.py`。正式稿必须与获批配置及实际执行代码一致。

## 数据与协议

目标BCI Competition IV Dataset2a；同被试T/E同分区，22EEG、250Hz、cue后4秒、重采样200Hz并分4patch。事件和官方标签逐Trial对齐；伪迹、滤波、单位、标准化、区域成员与缺失分布需先批准。确定性7/1/1是本代码支持的候选，不应写成嵌套验证。

开发选择只使用完整、仅C3、仅C4；C3+C4及超集不进入训练或开发选择。最终模式清单及机制干预必须事先冻结。所有模型共同预算和选择规则；B0完整训练，其余合法增强。参数匹配使用PDF6.5的候选结构，实际计数待GPU测试。

## 可复现性与局限

代码commit、配置哈希、官方源码及权重SHA、数据清单、训练统计来源、环境、每Trial掩码、最佳epoch和预测链保存在运行产物中。预训练数据与评价数据的重叠审计仍待完成。主结果不能超越9被试数据、共同训练集折重叠、区域定义候选和目标缺失配置的适用范围。

## 参考来源

- 官方CBraMod源码及许可证：https://github.com/wjq-learning/CBraMod 。正式论文引用需核对论文元数据后补齐。
- 数据集官方说明：https://www.bbci.de/competition/iv/desc_2a.pdf 。
- 原始课程PDF保持原样存放于 `docs/source/`；它不是外部文献有效性的替代证明。

Abstract、Results、Discussion、Conclusion中涉及真实效果的段落待实际实验及科研审查后撰写，不填入教学数字。
