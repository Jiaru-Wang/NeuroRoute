# 来源与实际验证状态

## 原始材料

- 所有者于 2026-11-21 在当前对话提供的完整 NeuroRoute Master Prompt。
- 同日补充：按模块批量开发，关键待定项需确认；代码上传 GitHub，所有者之后在授权服务器运行；不得因为 Mac 只有 CPU 而修改 GPU 研究方案。
- 六讲 PDF：NeuroRoute-原文全文-f4190653c2424fca9808543259b90567(2).pdf，413页。
- PDF SHA-256：2e31ad2c289c94c57739ec8df4027508aca7857b71df6a78825c8a49c1dc7887。
- 原始 PDF 已按所有者明确指示原样复制到 docs/source/，复制前后逐字节SHA一致；Master Prompt的派生执行要求保存在DEVELOPMENT_PROMPT.md，不上传服务器凭据。

## 官方依据与本次实际核查

| 来源 | URL | 本次核查范围与限制 |
|---|---|---|
| CBraMod作者仓库 | https://github.com/wjq-learning/CBraMod | 实际克隆并读取 models/cbramod.py、下游 bciciv2a 示例、预处理脚本及许可证 |
| 固定源码 | https://github.com/wjq-learning/CBraMod/tree/b9e961003214326972c567eff390e75b0287e32a | 本次克隆HEAD，未经运行验收；后续适配不能猜测当前接口 |
| 模型文件 | https://github.com/wjq-learning/CBraMod/blob/b9e961003214326972c567eff390e75b0287e32a/models/cbramod.py | forward(x,mask)、遮蔽先于卷积和FFT、proj_out、默认参数已读 |
| 官方预处理 | https://github.com/wjq-learning/CBraMod/blob/b9e961003214326972c567eff390e75b0287e32a/preprocessing/preprocessing_bciciv2a.py | MAT输入、CAR、因果Butterworth、resample、固定被试分区；与我们的GDF/严格LOSO方案需显式区分 |
| 数据集官方说明 | https://www.bbci.de/competition/iv/desc_2a.pdf | 9被试、4类、T/E、22EEG/3EOG、250Hz、trial时序、A04T校准例外；真实文件未读取 |
| MNE读取GDF | https://mne.tools/stable/generated/mne.io.read_raw_gdf.html | 官方API页面已访问；最终依赖版本与真实输入仍待验收 |
| MNE事件 | https://mne.tools/stable/generated/mne.events_from_annotations.html | 官方API页面已访问；必须显式映射事件，不能假定自动编号等于类别 |
| PyTorch加载 | https://docs.pytorch.org/docs/stable/generated/torch.load.html | 官方页面当前重定向至2.15；不能把stable文档版本当成服务器已安装版本 |

以上核查不构成完整官方论文复现、源码全量审查或数据重叠审计。预训练 checkpoint 尚未下载和核验。TimeKD/Time-MOE/FSMoE 的独立全文与引用仍需后续逐一核实。

## 已执行的工程检查

- SSH向GitHub认证明确返回 Jiaru-Wang。
- SSH读取目标仓库引用成功，初始无引用；克隆成功并提示空仓库。
- 匿名仓库网页HTTP 404，SSH可访问，符合私有仓库的访问表现；匿名API返回403，未将其当作私有性证明。
- 新项目独立于本机其他代码仓库。未修改或运行老师服务器代码。
- 本地只读SSH连接老师服务器曾返回认证失败；未建立可执行的远程会话，不声称已控制VS Code终端。

## 当前验收等级

| 项目 | 状态 |
|---|---|
| 章节定位与要求整理 | 初稿已写，完整理论冻结尚未验收 |
| GPU模型代码 | 八模型及训练/评价/机制实现已编写；尚未运行 |
| Python单元/数值测试 | 未运行 |
| 模型集成/梯度测试 | 未运行 |
| 真实EEG读取与训练 | 未运行 |
| 授权GPU L4 | 未运行 |
| 科研结果审查 | 未开始 |

静态检查及实际命令记录见TEST_STATUS.md；静态通过不证明模型数值或真实实验正确。uv.lock已解析，安装与GPU兼容性尚未验收。
