# NeuroRoute

面向授权 NVIDIA GPU 服务器的科研实现：冻结 CBraMod、合法缺失增强、八类对照、Spatial-EGS、LOSO、路由机制分析及被试级统计。

**当前交付为待 GPU 验收的研究代码，不是已完成实验。** 原始六讲 PDF 原样保存；本地只做静态检查，没有把 Mac CPU/MPS 用作模型实验替代设备。原来已删除项目的任何测试报告都不作为本项目验收依据。

## 从这里开始

- [完整开发提示词](docs/DEVELOPMENT_PROMPT.md)
- [六讲原文 PDF](docs/source/NeuroRoute-原文全文-f4190653c2424fca9808543259b90567%282%29.pdf)
- [六讲—公式—代码—测试—论文映射](docs/THEORY_IMPLEMENTATION_MATRIX.md)
- [明天的分步运行手册](docs/RUNBOOK_ZH.md)
- [待确认科研选择](docs/PROTOCOL_DECISIONS.md)与[完整候选配置](configs/protocol.review.json)
- [实现公式和张量契约](docs/IMPLEMENTATION_CONTRACTS.md)
- [实际验证状态](docs/TEST_STATUS.md)、[来源](docs/SOURCES_AND_STATUS.md)

明天在服务器项目根目录，首先只执行：

```bash
python3 scripts/preflight.py
```

这是标准库只读检查，不安装环境、不分配模型显存、不启动训练、不终止进程。将输出交回审核，确认设备与隔离环境后，再逐步执行手册。**不要一次启动全部实验。**

## 已编写的功能

| 功能 | 入口或实现 |
|---|---|
| GDF/官方标签审计、四秒窗口、22 EEG 缓存、训练折统计 | `inspect-gdf`、`prepare`；`data.py` |
| 固定官方源码、严格权重加载、原生 Mask、冻结池化 | `models.py`、`vendor/` |
| B0/B1、匹配 MLP、Vanilla/Mask MoE、Shared(h)/Shared(h,M)/Spatial-EGS | `models.py` |
| 合法掩码重放、逐 epoch 身份记录、公平数据顺序 | `missingness.py`、`engine.py` |
| GPU 合成测试、真实数据冒烟、早停与 checkpoint 恢复 | `gpu-tests`、`smoke`、`train` |
| 单独锁定外层测试入口、原始预测和路由保存 | `lock-test`、`evaluate` |
| 随机区域重新训练配置、K/lambda/eta 消融清单 | `ablations`、`plan` |
| 路由指标、受控干预、专家相似度与跨种子编号匹配 | `analyze`、`mechanisms`、`align-experts` |
| 9 被试内先平均种子、配对统计与实际训练曲线 | `statistics`、`plot-training` |
| 参数、激活参数、主要线性层 MACs、真实 GPU 延迟/显存 | `benchmark` |

Python 包位于 `src/neuroroute/`。紧密相关功能合并为模块，避免建立没有实现的空目录。CLI 完整参数可用 `uv run --locked neuroroute --help` 查看。

## 运行边界

1. `configs/*.review.json` 是**待审候选**，`approved=false` 会被训练入口拒绝。真实 GDF 通道名未看过，保留空值等待实际审计；其余候选值及来源均明示。不能只改一个开关就把未确认协议视为批准。
2. `uv.lock` 固定依赖；Linux 的 torch 来自官方 `cu128` 索引。安装后仍必须核对驱动兼容性、实际 wheel 和 GPU 输出。模型训练/评价入口明确拒绝 CPU/MPS 回退。
3. 文件解析、滤波、统计和绘图属于 CPU 数据处理；这不改变模型实验的 GPU 设备约束。
4. 官方 checkpoint、GDF、MAT 和真实实验结果不包含在 Git 中。由所有者按授权来源取得并记录 SHA-256。预训练数据重叠审计尚未完成。
5. 此实现提供 PDF 的确定性 7/1/1 候选 LOSO，不声称实现了嵌套交叉验证。协议批准前不进行正式训练。
6. 默认不自动续训。保存优化器用于恢复验证，但没有恢复完整随机数/调度状态，不能冒称断点训练精确续跑。
7. 主方法和统计流程都可能在明天的真实 GPU 验收中暴露需要修复的问题；有错误就留在该步修复并更新代码版本，禁止用未验收结果写论文。

## 来源与权限

仓库目标为私有 `Jiaru-Wang/NeuroRoute`。官方 CBraMod 两个源码文件保留 MIT 许可证；只把绝对导入改成包内相对导入，原始与适配后哈希保存在 `vendor/UPSTREAM.json`。项目代码不包含 GitHub 私钥、Token、教师服务器密码或其他用户数据。

PDF SHA-256：`2e31ad2c289c94c57739ec8df4027508aca7857b71df6a78825c8a49c1dc7887`。
