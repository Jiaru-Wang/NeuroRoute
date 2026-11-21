# 六讲理论—实现—测试—论文映射

状态：来源定位与实现映射；表内核心文件已编写，测试未实际执行。章节标题据 PDF，页码为 PDF 物理页。已定位 16 个小节；尚不能声称已完成全部图形公式和引用的逐项审查。

## 章节清单

| 小节 | PDF起始页 | 原始标题 | 内容边界 |
|---|---:|---|---|
|1.1|10|为什么 EEG 电极缺失需要特殊的专家路由？|EEG位置与缺失数量的区别；研究动机|
|1.2|21|研究假设、问题数学定义与创新性边界|H1/H2/H3；可证伪性；空间编码不增加观测|
|1.3|43|严格实验协议、跨被试泛化与未见缺失模式划分|双重泛化；被试/模式隔离|
|2.1|64|CBraMod 特征提取与缺失电极适配|Patch、时频与时空注意力、冻结与池化|
|2.2|91|缺失电极表示补偿模块设计|原生Mask与SCFC/FSMoE MMB的边界|
|3.1|114|空间条件化 EGS 专家路由设计|E0、候选专家、空间Router、Top-K融合|
|3.2|142|专家训练策略、梯度传播与空间专门化验证|稀疏梯度、Importance/Load、塌缩|
|4.1|168|完整/缺失 EEG 视图的对比学习与训练损失设计|InfoNCE等辅助目标，第一阶段不自动启用|
|5.1|192|端到端前向传播与完整训练流程|端到端shape、数据流、训练更新|
|6.1|222|科研环境搭建与数据组织|项目、数据、源码、权重来源及可追溯性|
|6.2|239|BCI IV-2a 数据读取、预处理与严格 LOSO 数据加载|事件、标签、单位、预处理与折内统计|
|6.3|266|CBraMod 预训练权重加载、Patch Mask 与梯度验证|strict加载、反事实、冻结、梯度验收|
|6.4|291|冻结 CBraMod + 线性分类头的第一个 LOSO 基线实验|B0/B1、早停、指标、checkpoint|
|6.5|316|公平对照实验与核心创新验证|参数/激活/MACs、基线、空间消融|
|6.6|351|专家激活可视化、空间专门化与路由干预实验|指标、配对、干预、编号置换与因果边界|
|6.7|387|第一阶段实验结果判定、失败诊断与论文创新决策|真实证据层级、统计、失败诊断及论文决策|

## 映射总表

每一行的测试/实验是验收要求，不是成功记录。源码核对、数值单元测试、集成、真实GPU和论文结果审查分别记账。

| 编号 | 讲次/小节 | 理论点 | 公式或契约 | 模块 | 实际文件 | 单元/数值检查 | 实验验收 | 论文章节 | 当前状态 |
|---|---|---|---|---|---|---|---|---|---|
|1|1.1/1.2|EEG/MI、22 EEG与四类别|[B,22,1000]|数据schema|data.py|形状/类型/类别非法输入|真实原始数据审计|Introduction/Dataset|已编写，运行待验收|
|2|1.2|位置、mask与空间归纳偏置|a=f(M)，不增加观测|区域编码|contracts.py / models.py|区域非空/覆盖/名称映射|真实vs随机区域重训|Problem/Method|成员待确认|
|3|1.2/6.7|H1/H2/H3研究假设|位置效应、条件化收益、未见组合收益|研究规范|docs/HYPOTHESES.md|无需网络编码；证据检查|公平对照与主比较|Introduction/Discussion|理论待逐条复核|
|4|1.3/6.2|九折LOSO、T/E同行|train ∩ val ∩ test按两两互斥校验|划分|contracts.py|每折完整且两两互斥|真实Trial ID覆盖|Protocol|内层方案待确认|
|5|1.3/6.2|训练折拟合统计|mean/std仅训练样本|折内变换|data.py|验证/测试改变不影响训练统计|每fold统计来源|Preprocessing|标准化方案待确认|
|6|1.3/6.4|联合缺失及超集保留|not(~M_C3 & ~M_C4)|合法性和采样|missingness.py|禁止集合/边界/重放|每Trial掩码记录|Protocol|核心，分布待确认|
|7|6.2|GDF事件与标签|768,769..772,783,1023|GDF解析|data.py|错位标签/缺cue/跨界/伪迹|官方独立标签核对|Dataset|已编写，运行待验收|
|8|6.2|单位、顺序、四秒窗口|250Hz*4=1000；200Hz*4=800|预处理|data.py|端点/单位/通道/EOG排除|原始记录与缓存审计|Preprocessing|D02/D03待确认|
|9|2.1|时域CNN与频域FFT嵌入|[B,C,T,200]->[B,C,T,200]|官方主干|models.py|官方接口/遮蔽位置检查|真实权重GPU前向|Background/Method|源码已查，运行未验收|
|10|2.1|Criss-Cross时空注意力|空间/时间分支|官方固定依赖|vendor/|版本/接口核查|冻结权重GPU验收|Related Work|不重复发明主干|
|11|2.1/6.3|Patch Mask方向|native_mask=(~M).expand(B,22,4)|适配器|models.py|计数/方向/隐藏输入反事实|真实checkpoint+GPU|Method|已编写，运行待验收|
|12|2.1/6.3|可用电极池化|v=mean_t(H)，h=sum_c Mv/sum_c M|池化|models.py|手算/全缺拒绝/隐藏输出排除|GPU接口集成|Method|已编写，运行待验收|
|13|6.3|严格加载与冻结|strict加载后proj_out=Identity|权重适配|models.py|键错/形状错/hash错拒绝；eval保持|权重SHA、前向、buffer/参数不变|Reproducibility|权重未获取|
|14|2.2|静态token、上下文补偿、SCFC|原生mask≠波形恢复；补偿公式另审|候选扩展说明|docs/OPTIONAL_EXTENSIONS.md|第一阶段配置拒绝误开启|核心失败诊断后另设实验|Related Work|可选，不纳入最小模型|
|15|3.1|专家MLP|200->128->200|专家|models.py|shape/参数/数值|GPU优化步|Method|已编写，运行待验收|
|16|3.1|条件Router|u=[h;M;a]，226->128->7|路由|models.py|输入信息隔离/shape|GPU路由训练|Method|已编写，运行待验收|
|17|3.1|候选Top-K及归一化|w_e=p_e/sum_selected p|稀疏选择|models.py|K边界/权重和/手算|实际dispatch路径|Method|已编写，运行待验收|
|18|3.1|E0残差融合|z=h+eta E0+(1-eta)sum wE|Spatial-EGS|models.py|二维手算/eta/输出|GPU完整梯度|Method|eta=0.5已指定|
|19|3.2|Top-K梯度路径|离散index；选中权重可微|梯度测试|tests/test_gpu.py|Router/E0/选中专家/未选专家|GPU梯度与优化步|Method/Validation|核心，未运行|
|20|3.2/6.6|Importance/硬Load|I=mean(p)，F=mean(indicator)|诊断|metrics.py / analysis.py|sum(I)=1，sum(F)=K|全分区加权聚合|Analysis|p151公式已看图|
|21|3.2/6.6|可微均衡|CV²(I)，E0不参与|损失|models.py|均匀=0/集中>0/梯度|lambda消融与塌缩诊断|Loss/Analysis|lambda待确认|
|22|4.1|InfoNCE/视图对齐/补偿损失|原公式需图形复核|候选扩展|docs/OPTIONAL_EXTENSIONS.md|启用前另写数值/泄漏测试|第一阶段后独立诊断|Related Work|可选，不默认实现|
|23|Master Prompt|知识蒸馏、TimeKD、Time-MOE|原论文机制及成本边界|参考梳理|docs/OPTIONAL_EXTENSIONS.md|核对真实引用，无需第一阶段代码|必要时另立扩展实验|Related Work|相关全文与具体公式未核实|
|24|5.1/6.4|统一forward/backward/update|CE+lambda*L_imp|训练引擎|engine.py|梯度/冻结/优化器集合/错误路径|授权GPU一步和完整epoch|Implementation|已编写，运行待验收|
|25|6.1|环境、数据、源码、权重追溯|代码->配置->数据->checkpoint链|来源/日志|contracts.py / experiments.py|hash不符/覆盖拒绝|环境与运行manifest|Reproducibility|已编写，运行待验收|
|26|6.4|B0/B1|Linear(200,4)，804参数|线性基线|models.py|相同初始化/全可用回归|完整vs合法增强|Baselines|已编写，运行待验收|
|27|6.4|四类完整分区指标|ACC、Macro-F1、Balanced ACC|评价|metrics.py|手算/未预测类/尾batch/无效输入|全验证预测重算|Metrics|已编写，运行待验收|
|28|6.4|条件归约与早停|mean(F1_condition)候选|开发协议|engine.py|与pooled F1反例/同分早期/耐心|真实验证全覆盖|Protocol|D06待确认|
|29|6.4|checkpoint保存恢复|模型+优化器+metadata|checkpoint|checkpoints.py|metadata错误/无覆盖/恢复下一步|GPU恢复一致性|Reproducibility|已编写，运行待验收|
|30|6.5|匹配MLP与普通MoE|1102隐藏层；8候选Top-3|基线|models.py|精确参数计数|相同mask/数据/预算|Baselines|PDF候选结构已定位|
|31|6.5|Mask/Shared各输入消融|[h;0;0]/[h;M;0]/[h;M;a]|对照族|models.py|填零槽与参数一致|同协议训练|Ablations|已编写，运行待验收|
|32|6.5|真实与随机分组重训|a_g=mean(M_Rg)|实验配置生成|planning.py|固定随机组/组大小/可重放|多个随机区域重训|Ablations|D05/D09待确认|
|33|6.5|K、lambda、eta消融|独立变量及共同预算|实验清单|configs/ablations.review.json|矩阵无重复/不混外层选择|获批配置全运行|Ablations|候选范围待确认|
|34|6.5|参数与MACs|Linear MAC=in*out；按实际路径汇总|成本统计|models.py / benchmark.py|理论与module参数交叉核对|GPU延迟/显存/warmup/sync|Efficiency|禁止以Top-K代替测速|
|35|6.6|路由熵|-sum p log(p)/log(E)|诊断|metrics.py / analysis.py|均匀=1/onehot=0/数值稳定|按模式/被试报告|Analysis|已编写，运行待验收|
|36|6.6|等数量缺失配对|固定同一trial，改变缺失位置|配对分析|analysis.py|身份/标签/数量对齐|预先规定模式评价|Analysis|已编写，运行待验收|
|37|6.6|JS与Top-K切换|JS范围[0,ln2]；1-∣交集∣/K|敏感性|metrics.py / analysis.py|相同/不交/半交手算|真实路由分布比较|Analysis|图形公式待完整复核|
|38|6.6|固定路由、权重交换、E0去除|冻结模型，受控改变计算路径|干预|mechanisms.py|恒等干预/路径变化/状态不变|配对预测和性能差异|Mechanism|干预清单待冻结|
|39|6.6|专家相似度、跨seed置换|输出相似度；编号不可直接对应|专家比较|analysis.py / mechanisms.py|置换/零向量/形状边界|多seed敏感性分析|Mechanism|对齐策略待确认|
|40|6.7|统计与科研判断|每被试先平均seed，再n=9配对|统计|analysis.py|完整性/重复/缺seed/配对/手算|真实结果与区间效应量|Results/Discussion|D09待确认|
|41|6.7|相关、干预与因果边界|无公式；主张需要证据等级|论文规范|paper/RESULTS_TEMPLATE.md|无需模型编码；科研审查|不能从激活推断生理因果|Discussion/Limitations|理论说明|
|42|6.7/Master Prompt|真实图表和论文一致|commit到原始日志的可追溯链|报表|analysis.py|假设数据禁止混入/缺运行显式报错|从真实日志重建图表|Paper/Supplement|未有新实验结果|

所有包内实现路径以 src/neuroroute/ 为根；tests、scripts、configs、docs、paper、third_party 路径以仓库为根。候选扩展不因出现在课程就自动纳入第一阶段。逐项审查发现新增重要内容时更新矩阵，不假称此初稿已穷尽全部413页细节。

## 本次可定位的自动化测试

| 文件 | 对应行为 | 当前状态 |
|---|---|---|
|tests/test_contracts.py|LOSO、两种合法采样器、重放、全局随机状态、手算指标、早停、JS/切换率|未运行|
|tests/test_gdf_audit.py|768/Cue/独立标签、伪迹前对齐、错位、缺Cue、边界|未运行|
|tests/test_data.py|缓存哈希、分区只读、训练折统计、防泄漏|未运行|
|tests/test_gpu.py|八模型参数/输入隔离、稀疏密集梯度、官方架构Mask、冻结、checkpoint、训练/评价集成|未运行|
|tests/test_error_paths.py|重复、尾批丢失、错误被试、非法数值、异常恢复、外层拒绝|未运行|
|tests/test_statistics.py|9被试统计、精确p值、缺种子/重复拒绝、专家编号置换|未运行|

理论条目有实现或说明映射不等于全量研究验收。真实GDF/官方预训练权重的L4证据与统计主张审查仍为空。
