# 服务器逐步运行手册

此文件是完整参考，不是要求一次执行全部命令。每阶段把真实输出交回审核，出错停在本阶段。所有下述命令都在 **VS Code 远程终端、你自己的项目根目录** 执行。没有 sudo、驱动修改、公共环境安装、kill 或自动后台训练。

## 0. 把代码放入个人目录

仓库是私有 `Jiaru-Wang/NeuroRoute`，开发分支 `codex/research-contracts`。若导师服务器的共享账号没有你自己的安全 GitHub 认证，不要上传私钥或Token到共享账号。可在自己的 Mac 上获取仓库，然后通过 VS Code 文件上传把项目或 Git bundle 传到个人目录。

如使用随交付提供的 bundle，先将它上传到个人目录；在个人目录内执行 `git clone --branch codex/research-contracts NeuroRoute.bundle NeuroRoute`。bundle保留Git历史，运行脚本需要Git commit和干净工作树；GitHub ZIP不能直接满足这个要求。上传后可保留bundle作为源代码副本，不自动删除。

## 1. 明天的第一步：只读环境检查

```bash
python3 scripts/preflight.py
```

预期结束标记 `PREFLIGHT_INSPECTION_ONLY`。请发回完整输出。脚本报告主机、Python、磁盘、uv、Git、GPU UUID、显存和计算进程。它不导入PyTorch模型，不占用模型显存。终端里某GPU显示空闲也不等于已经获得使用授权；选导师允许的实例，核对当时状态，不结束他人进程。

后续步骤在本步验收后进行。不要照抄历史UUID，不因某GPU忙而自动切换到未授权GPU或CPU。

## 2. 用户目录中的隔离依赖

仅当服务器已有可用 `uv` 且环境检查通过，再执行这一组：

```bash
export UV_CACHE_DIR="$PWD/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$PWD/.python"
uv sync --locked --python 3.11
uv run --locked neuroroute doctor
```

`uv`在项目创建`.venv`，Python及缓存位于个人项目内。若uv不存在、网络不可达、空间不足、驱动不支持CUDA wheel，发回错误，不改公共环境。Linux torch固定官方CUDA12.8 wheel；`uv.lock`是已解析的依赖锁，不代表已在服务器安装成功。

CPU负责读取文件、事件处理、信号重采样和统计；所有神经网络前向/反向、集成、推理成本实验使用授权NVIDIA GPU。GPU不可用会报错，没有CPU/MPS模型回退。

## 3. 检查实际数据与冻结科研协议

数据文件不在GitHub中。以官方授权来源取得原始文件及独立标签，并记录来源/校验值；通用 `fetch`入口只接受公开HTTPS和已经核实的SHA-256，不猜测下载链接，不处理需登录的网盘页面。

当前读取器预期：`data/raw/gdf/A01T.gdf` 至 `A09E.gdf`，以及 `data/raw/labels/A01T.mat` 至 `A09E.mat`，标签键 `classlabel`、288条、1..4。如果官方提供的实际标签集合/结构不同，先反馈，不改名或拼造标签以通过检查。

先看一个真实GDF的元数据：

```bash
uv run --locked neuroroute inspect-gdf data/raw/gdf/A01T.gdf
```

然后与助手审查 `configs/protocol.review.json` 和 `docs/PROTOCOL_DECISIONS.md`：

- 核实实际22 EEG及3 EOG名称；按canonical order映射，不能只凭文件里的序号猜位置。
- 批准原始参考、伪迹策略、滤波、重采样、幅值缩放、训练折标准化。
- 批准确定性7/1/1、四区域成员、增强概率、开发分数、共同预算与普通MoE融合。
- 在开始正式比较前批准最终缺失模式、机制干预、消融网格与统计主比较。

候选配置完整列出了可建议的数值，但它们不是自动批准的协议。四区域成员、额外测试模式、lambda/eta/K消融网格和统计检验尤其需要明确决定。把获批配置保存为`configs/local.approved.json`，填写批准说明并将`approved`设为true；真实通道名也必须填入。`local*.json`不自动上传，可包含机器本地路径；运行会保存精确快照和哈希。

```bash
uv run --locked neuroroute check-config --config configs/local.approved.json
```

这个命令通过只代表配置合同满足，不代表数据/模型已经验收。修改任何研究选择需新快照及对应实验范围审查。

## 4. 数据缓存与严格标签对齐

```bash
uv run --locked neuroroute prepare --config configs/local.approved.json --raw-dir data/raw/gdf --labels-dir data/raw/labels --output data/cache/protocol_v1
```

会新建缓存目录，拒绝覆盖；先核对完整288个Trial与官方标签，再按获批伪迹策略筛选。保留原始trial_id、事件采样点、伪迹标记、类计数、输入/输出SHA及预处理依赖版本。原始数据准备会读取全部18个session做确定性独立处理，不拟合跨被试统计；训练/验证加载器只读取各自角色的EEG缓存。

`manifest.json`以及18个`*_audit.json`需要审核。若失败，保留部分产物排查，下一次使用新的输出目录，不覆写失败证据。

## 5. 授权GPU上的合成测试

先依据新的preflight输出选择导师允许的GPU/MIG UUID，并在终端明确设置。示意变量中的尖括号必须替换成实际值，不可原样执行：

```bash
export CUDA_VISIBLE_DEVICES='<授权且已检查的GPU或MIG UUID>'
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
uv run --locked neuroroute gpu-tests --output logs/checks/gpu_units_001.json
```

入口要求一个可见CUDA设备，不允许用skip把无GPU算通过。测试包含官方架构合成权重的Mask/冻结检查、专家/Router梯度、稀疏/密集数值比较、checkpoint恢复、合成Trial/标签/LOSO、防泄漏、指标手算、错误路径与统计。这里用随机官方架构权重的测试**不代表真实预训练权重验收**。

发回完整测试输出与JSON；报告失败或缺测试时禁止训练。

## 6. 真实官方权重与GPU冒烟

先从固定官方CBraMod仓库README核实checkpoint来源，并记录实际文件SHA-256。不得把未核实哈希、随机权重或其他结构checkpoint当官方预训练权重。预训练数据重叠审计单独记录。路径和哈希由实际文件确定后设置：

```bash
export NR_BACKBONE='pretrained/已核实的官方权重.pth'
export NR_BACKBONE_SHA256='<已经核实的64位SHA256>'
uv run --locked neuroroute smoke --config configs/local.approved.json --cache data/cache/protocol_v1 --backbone "$NR_BACKBONE" --backbone-sha256 "$NR_BACKBONE_SHA256" --variant B0 --fold 9 --seed 42 --output logs/smoke
```

会检查当前Git工作树干净、源码版本、strict加载、冻结、隐藏值反事实、一次下游更新、checkpoint和优化器下一步恢复、完整合法验证分区；不读外层测试。结果保存在新时间戳目录，成功生成 `acceptance.json`。这属于流水线L4冒烟，不是论文有效性验证。B0的反事实测试使用单缺输入是接口检查，不是B0正式训练增强。

其他模型/配置/折进入训练前也要有匹配验收报告；新代码版本需要重新相关测试。命令的具体参数由该阶段审核后逐步给出。

## 7. 先一个Fold、一个种子训练验收

将 `NR_ACCEPTANCE` 改成上步实际生成的验收文件：

```bash
export NR_ACCEPTANCE='logs/smoke/<实际run目录>/acceptance.json'
uv run --locked neuroroute train --config configs/local.approved.json --cache data/cache/protocol_v1 --backbone "$NR_BACKBONE" --backbone-sha256 "$NR_BACKBONE_SHA256" --variant B0 --fold 9 --seed 42 --acceptance "$NR_ACCEPTANCE" --unit-report logs/checks/gpu_units_001.json --output results/development
```

会记录每epoch训练和三条件验证的原始Trial预测/掩码、损失、路由、模型选择分数、早停及最佳权重。只优化下游；主干和验证时参数必须不变。训练结束恢复最佳权重，并核对整套验证预测。

程序没有精确断点续训承诺。中断后保留失败日志；不要把仅恢复分类头/优化器叫完整训练续跑。需要长时间任务时，另行确认终端会话管理和资源授权。

## 8. 基线、消融与计算成本

`plan`仅写计划，不启动任何任务。默认八模型×9折×3种子为216个主比较训练作业，消融另计，不能为节省本地计算偷偷减少论文要求。

```bash
uv run --locked neuroroute plan --config configs/local.approved.json --output logs/plans/main_001.json
```

先批准消融候选并另存 `configs/local.ablations.json`，再用 `ablations --config ... --plan configs/local.ablations.json --output logs/plans/ablations_001`生成独立配置。随机区域保持组大小、使用固定种子，**必须重新训练**；不允许仅推理时打乱区域就替代空间先验消融。K变化后，固定专家干预IDs也要匹配K并重新批准机制计划。

成本入口需明确预算，示例：

```bash
uv run --locked neuroroute benchmark --config configs/local.approved.json --cache data/cache/protocol_v1 --backbone "$NR_BACKBONE" --backbone-sha256 "$NR_BACKBONE_SHA256" --summary 'results/development/<实际run>/summary.json' --warmup 20 --repeats 100 --batch-size 16 --output logs/benchmarks/001.json
```

这读取训练分区测真实GPU，不读外层测试。记录批延迟、吞吐、显存、完整/下游/密集参考路径。组间比较应使用相同硬件/批量/预算，空闲情况需记录；不能从Top-K直接宣称加速。

## 9. 冻结后才开放外层测试

所有模型和超参数只用开发验证确定。正式主比较、消融、模式和统计计划审核后，由所有者明确锁定允许的训练summary清单：

```bash
uv run --locked neuroroute lock-test --config configs/local.approved.json --summaries 'results/development/<实际run>/summary.json' --approval-note '<明确的测试开放批准记录>' --output logs/locks/final_001.json
```

该示例只列一个run，不代表完整216作业覆盖；正式锁定必须列入全部预先决定的运行。入口检查训练状态和checkpoint哈希，记录允许集合。之后每个run分别 `evaluate`：

```bash
uv run --locked neuroroute evaluate --config configs/local.approved.json --cache data/cache/protocol_v1 --backbone "$NR_BACKBONE" --backbone-sha256 "$NR_BACKBONE_SHA256" --summary 'results/development/<实际run>/summary.json' --lock logs/locks/final_001.json --output results/final
```

代码版本、配置、数据和权重不匹配会拒绝。生成带Trial身份的 `predictions.json` 和不含大预测列表的 `summary.json`。外层结果不回传早停或选参。锁文件是可审计的流程门禁，不能阻止人为改代码；科研诚信仍需所有者遵守。

## 10. 机制、图表与统计

- `mechanisms`与`evaluate`使用同样的参数组，但输出到新的机制目录；只适用于路由模型。机制计划必须在最终锁定之前写入同一获批配置。
- `analyze --predictions results/final/<run>/predictions.json --output results/analysis/<新目录>`生成真实指标/激活图与等数量缺失配对JS和切换率。
- `plot-training --run results/development/<run> --output results/analysis/<新文件>.png`从真实epoch日志绘图。
- `align-experts --profiles <JSON文件> --output <新JSON文件>`输入 `left/right`两个条件×专家矩阵，行条件必须事先对齐。优先在训练/开发profile上定义编号匹配，测试匹配只能明确标为探索性描述。
- `statistics --summaries <所有真实final summary路径> --plan configs/local.statistics.json --output results/analysis/<新文件>.json`要求每个比较模型完整9折×3种子，先被试内平均，缺失或重复会报错；候选计划文件需先批准。

同一模型不同配置的重训练消融，统计计划可增加 `groups` 字典：每个组名指定 `variant`和确切`config_sha256`，`comparisons`引用这些组名。由此区分真实空间组与多个随机区域配置，不能用文件名臆测组别。主基线默认要求同一配置；消融组必须在分析前明确登记配置差异与公平性。

## 回传与修复

每步回传：执行命令、完整终端输出、对应JSON报告/错误日志、Git commit。不要发送账号密码、Token或私钥。大量结果按本地文件传回，不自动上传原始EEG到GitHub。

错误时只修当前失败路径，新增回归测试、提交新commit，明确哪些旧结果作废和需要重跑。修复后依次重新验收；不把输出变成“OK”就算科学正确，也不依据保留测试反复试到好结果。
