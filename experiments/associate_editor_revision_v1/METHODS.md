# 冻结方法 v1

本方法作为返修期追加内部验证；原开发数据已被反复使用，不能称为独立新队列、患者独立验证或前瞻确认试验。参数在新结果产生前确定，实施预检失败需记录修订版本，不允许看结果后改方案。

## A 分类评价与匹配先验对照

固定原5个similarity-grouped外折。运行5个配对训练seed，分类输入224×224，原DFUC2022生成器与已生成prior固定。这样检验原先验在更严格分类评价下的效果，不包含生成器重训不确定性。

三个主要增量比较：Full LPT−neutral Full LPT；RA-LPT−six-model output-only；strong RGB+prior−strong RGB。Full LPT/RA-LPT−ResNet50仅作完整系统参考。不重开Candidate13融合搜索。

### 每外折每seed

1. 划出外评价集E和外训练集T。对T用StratifiedGroupKFold的3折产生V0–V2。对每个T\Vj再做5折分组分层，取第0折作stop，其他4折作fit。划分seed独立于训练seed，所有模型和5个重复共享划分。
2. CNN在fit上训练，只在stop上选checkpoint，之后才预测Vj。Vj的标签不得用于选epoch或超参数。三个Vj预测合为覆盖T一次的inner-OOF。
3. 8个基模型为ResNet50、EfficientNetB0、LPT no-alignment/no-modulation/no-ROI/stats-only、ConvNeXtTiny、EfficientNetV2S。前6个产生RA-LPT及其output-only特征；4个RGB产生strong RGB两种stack特征。
4. Stacker在T的新OOF上按原有限网格选C、class_weight；用inner交叉预测选bias，之后在整个T的OOF上拟合Scaler及逻辑回归。网格平分时取既定枚举顺序首项。全部步骤仅在T；内部选择分数只用于选择，不作为独立验证指标报告。
5. 各基模型取三个内部最佳epoch的中位数，在全部T从ImageNet初始化重新拟合至固定epoch。refit无早停、不访问E，cosine仍沿原60-epoch时间轴运行。refit模型预测E；stacker以这些E特征给出最终输出。
6. Full LPT与neutral另用T内相同5折分组分层的第0折作stop，各fit一次选epoch，再各在全部T refit一次。配对臂使用相同初始权重、样本次序、增强随机流。模型各自选epoch，比较的是相同选择规则下的两个学习流程，而不是严格update-matched梯度实验。
7. E仅评价一次；所有外层分数推迟至全部预测及配置哈希封存后统一计算。E只允许在无训练的类别计数检查阶段读标签，不能读入训练/选择模块。

注意：这里保证最终outer评价对整个训练/选择流程隔离。T内再对固定inner-OOF表做stacker选择仍可存在内部依赖；不能称每层内部选择分数均无偏。若另要求meta选择层也完全独立，要进一步在每次meta-training内部重新生成CNN OOF，属于另一协议，不能无声扩大或冒称已经完成。

### Neutral定义

Full LPT全部架构、辅助头、alignment head和loss权重保留；将样本特异prior替换为浮点常量0.5图，增强后仍常量。ROI全幅原图；stats为[0.5,0.5,1.0,0.5]；alignment target同常量。不得继续读取真实prior决定ROI/其他特征；不能仅设alpha=0或喂全零mask。

此控制检验样本特异prior信息相对固定非特异prior的贡献，不验证其解剖正确性；常量alignment也改变正则化目标的内容，应保留这一解释限制。

### 数值与随机性

- AdamW、学习率、weight decay、模型变体loss/结构取protocol.yaml指定源配置；运行前展开并保存完整resolved配置。所有分类模型batch16、AMP FP16、4workers、无梯度累积、max60epochs、patience12，避免跨模型运行环境漂移。
- 这些设置是协调后的新协议，新旧绝对分数差不能精确等同于泄漏偏倚。
- epoch从1计数；checkpoint只在stop Macro-F1严格提高时更新，平分保留较早epoch。主要Macro-F1始终对四个固定类别求平均，undefined F1=0。
- 随机流分别控制初始化、样本顺序、几何/颜色增强和dropout；配对臂使用同seed规则，不能因prior分支改变全局RNG消费而改变增强。保存首epoch样本/增强哈希作配对检查。
- 所有算子确定性设置、AMP、库版本和GPU记录；无法使用确定性算子时显式记录，不静默降级。

## B 源域分割评价审计

使用原DFUC2022的5个外折，每个外训练部分内部做5折KFold，取第0折stop。若源域有已知重复/同病例关系则必须保持同组；无法取得身份信息时不称患者独立。使用5个seed，各fit/stop一次后只评价对应源域外折，共25次fit。

模型、损失和训练参数继承指定源配置：512×384，batch8，max100epochs，patience15。stop Dice选epoch，相等取较早者。外折只评价一次冻结checkpoint，同时报告512×384与224×224的Dice/IoU/precision/recall；阈值0.5，每图计算再汇总，沿用并核对原空mask处理，不为新结果改定义。

不按外折成绩选best-overall，不用B中新checkpoint替换A的原生成器。B修正的是源域评价流程，不能倒推原生成器未见过源域选择数据，也不能验证DFUC2021的mask准确性。

## C 统计和少数类

每seed合并5个outer预测后计算pooled Macro-F1和三个配对差，报告全部seed、均值和sample SD，不平均概率冒充单模型成绩。每fold×seed另报告四类precision/recall/F1、support、TP/FP/FN，precision分母为0单列NA。

5,000次paired crossed seed×group bootstrap：共同抽取5670个组（有放回），保留组内全部图片及重复次数，再共同抽取5个seed（有放回）。每seed各算Macro-F1差，再平均。输出percentile95%CI；原每seed group-only区间另列。不能把25个fold×seed当独立样本，也不能把图片×seed拼接为独立样本。

如报告p值，沿用明示的近似双侧bootstrap tail probability（plus-one、零差进入双尾、截断1），三个比较同族Holm调整；不得宣称精确检验或无条件确认性证据。CI未经同时性调整，应说明。缺失类bootstrap重复按固定四类F1=0处理，不反复重抽到满意。

区间反映已有训练重复和评价组抽样，仍不含对每个重抽训练集完整重训、历史人工开发、患者依赖及固定生成器的不确定性。非显著不等于等价。

## D 解释性

原主分析grouped ResNet50和Full LPT的对应外fold checkpoint，不用standard-CV或任意best-overall。优先解释各CNN，不把CNN的CAM称为整个RA-LPT解释。

定量纳入全部可分析5955图，空mask单列。每张非空mask预先生成最多30个不同、形状/面积不变、完整落在图内的合法平移位置，按固定seed无放回抽样；位置不足则全部枚举，明确n。全幅/不可移动mask没有有效随机对照，单列不可估，不能硬报30次。

两模型共享位置，使用ImageNet均值填充。Full LPT按原图坐标同步遮挡global与ROI；主分析保持原prior和ROI位置固定。计算prior遮挡分数下降减随机位置平均下降；bootstrap单位为图像相似组，不是placement。

在看任何CAM/预测前四类各随机抽5个不同group、每group一图，固定并保存20例清单；保留错误分类病例，不按分数筛选。定量报告不能只用这20例。若主结果转为A的新流程，最终相应图需改用新checkpoint，旧grouped解释仅列历史审计。

## 硬性验收与停止规则

1. image唯一、每图一个标签、5955图/5670组；group不得跨outer、inner或fit/stop边界。两条拟合路径共享来源时必须记录祖先集合。
2. 每个fit/stop都含四类，分类每类至少10个样本；不满足则在未训练前停止并修订分组方案，不随机尝试直至获得高分。打印每类图数和组数。
3. 每outer的inner-OOF恰好覆盖T一次；outer-test只由该outer范围生成的CNN及stacker预测。
4. 完整resolved config、原始输入/划分/代码/权重哈希封存后才能正式训练。失败重跑仅限相同配置与seed，保留失败记录。
5. neutral的真实prior读取和路径使用断言为零；架构参数量与Full完全一致；初始化和样本/增强流配对。
6. 输出OOF需含image、label、group、outer_fold、seed、model、prob_0–3、pred_label、checkpoint_hash。辅助概率、bias、选定参数另保存，不能丢失stack特征所需输出。
7. 运行总量：分类900次CNN fit、源域分割25次fit；另加小模型stacking和解释性推理。不运行高分辨率分类搜索、学习曲线、目标域人工标注或新生成器全量refit。
8. 不保证结果优于原稿；不得依据任何部分外层结果更改方法、停止差seed或新增有利对照。
