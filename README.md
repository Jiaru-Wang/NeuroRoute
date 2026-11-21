# NeuroRoute

NeuroRoute: Spatial Missingness-Conditioned Mixture of Experts on Pretrained EEG Representations for Robust Motor Imagery Decoding

本仓库用于基于冻结 CBraMod 的 EEG 缺失电极鲁棒运动想象研究。研究假设尚未得到新实验验证。

## 交付顺序

1. [完整代码生成与验收提示词](docs/DEVELOPMENT_PROMPT.md)：可审查开发指令。
2. [六讲理论对应表](docs/THEORY_IMPLEMENTATION_MATRIX.md)：原始章节、公式、拟实现模块、测试与论文映射。
3. [待确认协议](docs/PROTOCOL_DECISIONS.md)：区分明确要求、PDF 示例与未冻结设置。
4. [开发阶段与验收](docs/EXECUTION_PLAN.md)：按模块提交代码和测试。
5. [来源和验证状态](docs/SOURCES_AND_STATUS.md)：源码版本与实际完成的检查。

## 当前状态

本次提交是阶段 A 的开发契约，**尚未包含完整项目实现**。代码、数据管线、单元测试、集成测试、GPU 验收、正式实验均不能据此标记完成。下一批实现以前置协议确认和官方 API 审查为依据。

此前已删除的旧项目、环境、结果和测试报告，不作为本仓库的当前验收证据。不得把教学数字或旧日志复制成新实验结果。

## 设备与数据边界

实验代码面向授权 NVIDIA GPU。训练、真实模型集成验收和延迟/显存测量要求 CUDA 可用；禁止自动 CPU 回退、缩小专家数量或减少折数来适配开发电脑。本地仅进行文档、源码与静态检查。服务器执行由项目所有者后续启动。

不提交 EEG 数据、官方权重、环境凭据、服务器连接信息或真实试次输出。服务器工作仅限所有者个人目录，不修改公共环境，不停止他人进程。

仓库归属：Jiaru-Wang/NeuroRoute。保持私有。
