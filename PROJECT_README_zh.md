# 面向工业储能的15分钟滚动负荷预测 — 项目版 v0.1

整理日期：2026-09-19。原教学包资料检查标记：2026-09-18。
二者都不是数据截止日期、原作者提交日期或训练完成日期。

## 1. 现在怎么用

不重新创建环境，不重新下载仓库，不覆盖原来的 Notebook、src、README 或原数据。
把本压缩包里的文件直接解压到 `D:\python\test-LSTM`。
打开 `01_load15_my_project.ipynb`，内核选择 `Python (load15)`，先运行第1—8节完成数据准备与基线。
直接负荷模型使用独立 Notebook：`02_load15_direct_lstm.ipynb`（干净内核即可，不依赖第01本内存变量）。
配置与数据文件位于同一层目录。数据准备依赖见 `requirements_data.txt`；直接负荷 LSTM 依赖见 `requirements_lstm.txt`（已在 `load15` 安装 `tensorflow==2.16.2`、`keras==3.4.1`）。
第01本第9节原仓库残差模型检查默认关闭，不要用它做本项目正式训练入口。

## 2. 本项目的任务

目标应用是工业负荷预测辅助储能控制。
每个预测发布时刻，使用已获得的7天负荷，预测紧接着24小时的96点负荷；15分钟后更新输入再预测。
7天是一个样本的历史长度，不是全部训练数据长度；每15分钟预测也不代表每15分钟重训。

当前配置指向工业 15 分钟表（由 `industrial_load_1min.csv` 聚合），单位 kW，时区 `Asia/Shanghai`。
公开德国数据仍可用于对照流程，不能与工业站成绩混比。

## 3. 当前实现与后续计划

| 状态 | 内容 |
|---|---|
| 已写入本版代码 | 集中配置、数据检查、完整15分钟网格、672历史窗口、96点答案、时间划分 |
| 已写入本版代码 | 训练期标准化、昨天/上周固定时长基线、分钟级周期编码、5/4特征样本 |
| 已写入本版代码 | 工业1分钟→15分钟适配器；独立输出目录；测试集不评分 |
| **直接负荷模型已实现** | `load15_direct_dataset.py` + `load15_direct_lstm.py` + `02_load15_direct_lstm.ipynb`：`(672,5)+(96,4)→(96,)` 直接负荷，非残差 |
| **短试跑已完成** | 2026-09-21：训练46 / 验证13，batch=16，epochs=2；标准化空间 MSE 有日志；**非正式成绩**；正式训练尚未执行 |
| 试跑记录 | `my_outputs/direct_lstm_trial_20260921T081946_701356Z_0ee1efa4/` |
| 待开发 | 正式训练（须重新初始化模型与优化器，不继承试跑权重）、保存模型、多步评估、最终测试 |
| 待开发 | 节假日/调休/班次/电价特征、7天与14天公平比较 |
| 待开发 | 预测结果接口、风险识别、离线储能优化闭环与业务指标 |

程序已写不等于正式训练完成。两轮短试跑的 loss / val_loss **不能**写成正式预测成绩。
历史项目中的MAPE、R²或风险事件减少数据，应留在历史项目说明中；不能填入本次实验结果。

## 4. 配置改哪里

`my_load15_config.json` 是本版参数入口，Notebook会真正读取它。
本轮保持 `lookback_days=7`，跑通后可以在数据准备阶段改为14，并从头重跑。

| 参数 | 当前值（工业配置） | 说明 |
|---|---|---|
| dataset_kind | industrial_csv | 读取已聚合的工业15分钟表；公开演示为 smard_demo |
| target_column | load_kw | 直接负荷目标，不使用运营商预测 |
| data_file | industrial_load_15min.parquet | 由1分钟表聚合得到 |
| sampling_minutes | 15 | 当前版固定 |
| lookback_days | 7 | 对应历史 672 点 |
| horizon_hours | 24 | 对应输出 96 点 |
| origin_stride_steps | 1 | 每15分钟滚动一次 |
| feature_timezone | Asia/Shanghai | 时间索引仍用UTC |
| load_unit | kW | unit_verified=true |
| train_fraction / validation_fraction | 0.70 / 0.26 | 冻结分区；勿为提高成绩改划分 |

工业适配器见 `industrial_load_adapter.py`。工业站合理的停机零负荷不能按德国总负荷的非正值异常规则机械处理。

## 5. 特征与泄漏边界

历史输入：标准化负荷、time_sin、time_cos、dow_sin、dow_cos。
未来已知输入：上述4个确定性时间特征，不含未来真实负荷。
分钟级周期：一天进度 = (hour*60+minute)/1440；星期周期 = weekday/7。
周末不是工厂休息日的同义词。调休、班次、节假日及电价需来源匹配，且在预测时可用；未取得时不虚构。

标准化只拟合训练期；样本的全部96个答案必须落在同一分区。
验证输入可以使用预测时已发生的实际负荷；不能输入尚未发生的未来负荷。
缺失时间戳保留，含缺口的窗口跳过并记录，不插值编造未来答案。

## 6. 复现与改进必须分开

上游原任务：日前发布、预测实际负荷与电网运营商预测的差值。
本版任务：每15分钟滚动、直接预测未来24小时负荷。
两者的成绩不能不加说明直接比较。
上游模型图默认历史长度固定672；只改本版配置不能自动变更其模型输入。
原仓库源码、来源说明和许可证文件保留；新增实现与实验记录单独存放。

## 7. 实验结果怎么记录

第8节输出到 `my_outputs/<UTC时间戳_uuid>/`，每次另建目录。
`experiment.json` 包含代码整理日期、原资料检查标记、真实执行时刻、数据起止时间、配置、SHA256及验证状态。
`validation_baselines.csv` 是运行后真实生成的验证成绩；没有预填分数。
`one_sample.npz` 保存负荷单特征、5特征历史、4特征未来与答案示例。
同一目标时刻可能被不同发布时刻预测多次，指标按“发布时刻×预测步长”的预测对统计。
后续补充每个horizon的MAPE/R²、峰值与低负荷区间误差，以及储能仿真指标。

## 8. 储能接口（设计，尚未实现）

预测表建议保留：issue_time、target_interval_start、horizon_step、predicted_load、load_unit、model_version。
future_actual_load仅可在事后评估时合并，不用于生成当前预测。
当前时间戳按区间起点解释，生产接入前须核对，不能与区间结束时间混用。

在负荷已统一为kW且不另计光伏的示例下：
预测并网功率 = 预测负荷 + 基准充电功率 - 基准放电功率。
结合SOC、设备限值判断基准计划是否安全；安全时沿用，预测有风险才对功率做最小必要修正。
此处是业务设计，不是本版已完成的求解器。15分钟模型不能替代瞬时防逆流与超功率保护。

## 9. 怎样修改文档和代码

Notebook 的 Markdown 单元格用于业务说明、假设和结论，Code 单元格用于实际逻辑。
只改“7天”为“14天”的说明，不会改变计算；必须改配置并从头执行。
在JupyterLab里打开JSON或Markdown可通过文本编辑器修改。修改前保存副本，避免覆盖唯一可回退版本。
先改一类因素并保留基线，确认不出错后再加入下一项。

## 来源

- 原项目：https://github.com/connectashish028/german-day-ahead-forecast
- 原数据窗口：https://github.com/connectashish028/german-day-ahead-forecast/blob/main/src/loadforecast/models/dataset.py
- 原LSTM图：https://github.com/connectashish028/german-day-ahead-forecast/blob/main/src/loadforecast/models/lstm_plain.py
- 时间序列教程：https://www.tensorflow.org/tutorials/structured_data/time_series
- 数据泄漏：https://scikit-learn.org/stable/common_pitfalls.html
- JupyterLab Notebook：https://jupyterlab.readthedocs.io/en/stable/user/notebook.html

本版业务流程根据用户提供的《LSTM储能综合》说明转为学习项目需求，不等同于复原其历史生产代码。

## 2026-09-28 第一轮正式直接负荷 LSTM

数据覆盖 2024-08-01 17:45 至 2024-12-31 23:45（Asia/Shanghai）。固定 target period 后，有效样本为 train 1192、validation 1169、test 1617；test 本轮保持封存，未预测、未评分。

模型为 encoder LSTM(32) → decoder LSTM(32，encoder h/c 初始化) → TimeDistributed Dense(1)，接口 `(672,5)+(96,4)→(96,)`。训练使用 Adam、初始学习率 1e-3、标准化空间 MSE、batch 32、最多100 epochs、seed 42、EarlyStopping(patience=10) 与 ReduceLROnPlateau(patience=5, factor=0.5)。实际训练31 epochs，最佳为第21 epoch，最佳 val_loss=0.0282039233，EarlyStopping 已恢复最佳权重。

Validation 正式结果：Direct LSTM MAE 44.788800 kW、RMSE 72.686732 kW、MAPE 11.146815%、R² 0.970825。相同1169个 origin 上，昨天同期为 32.836406 / 63.081994 / 7.127217% / 0.978026，上周同期为 20.986387 / 28.179248 / 5.774038% / 0.995615（依次为 MAE、RMSE、MAPE、R²）。因此第一轮 LSTM 未超过两个朴素基线；这是 validation 结论，不是 test 成绩。

正式产物位于 `my_outputs/direct_lstm_formal_20260928T120117_444358Z/`。

## 2026-10-15 当前推荐预测器

当前推荐方案不是第一轮 Direct LSTM，而是严格因果的统计预测器：`未来96点 = 上周同期96点 + 0.40 × 最近4小时平均周残差`。系数0.40只由 train 选择，并通过带96步标签隔离的向前滚动验证；validation 上 MAE=19.287353 kW，优于同一1153个 origin 上的 Last-week MAE=21.058767 kW。

推理入口为 `predict_recent_residual.py --origin "YYYY-MM-DD HH:MM"`。它只读取 origin 之前的数据；最近4小时 residual 不完整时回退到纯 Last-week，上周同期96点不完整时明确失败。输出包含未来96点、baseline、修正量和完整来源时间元数据。test 仍未评分。

版本化配置保存在 `model_checkpoints/recent_residual_v1/model.json`。正式推理路径已对全部1169个 validation origin 逐条重放：1153次 correction、16次安全回退，回退率1.37%；所有输出均为96个有限值。包含回退的整体 validation MAE=19.239217 kW，对应 Last-week MAE=20.986387 kW。单次推理平均约4.73 ms（本机CPU、仅核心计算，不代表生产SLA）。

站点 `#4` 作为未参与 `#3` 参数选择的新站进行了零适配验证。冻结的 `#3 beta=0.40` 在 `#4` validation 的1792个共同origin上将 Last-week MAE从63.498017降至59.604651 kW；`#4` train-only MAE最优beta为0.30，对应MAE=59.379062 kW。两者差距较小，表明当前公式具备初步跨站泛化能力，但zero-shot MAPE从16.663276%升至17.166756%，尚不能据两个站点宣称普适。`station4_load15_config.json` 可让同一推理代码加载 `#4`，不需要复制模型实现。

在 `#3/#4` train 上按站点等权选择了共享 beta=0.35，冻结为 `model_checkpoints/recent_residual_global_v1/model.json`。选择目标是两站各自 MAE/Last-week MAE 的宏平均，validation 不参与选择。共享模型在 `#3` validation 的 MAE=19.405508 kW（Last-week 21.058767），在 `#4` validation 的 MAE=59.405385 kW（Last-week 63.498017）；两站 validation 宏平均 MAE 改善7.15%。第三个电站保持完全封存，等待最终一次零适配测试。
### 尺度无关诊断（2026-10-15）

`diagnose_scale_independent_transfer.py` 对 #3/#4 做了 train-only 选参和双向整站 validation。
对公式 `last_week + beta * recent_residual`，若用同一个站内标量将 residual 归一化并在输出端还原，标量会代数抵消；四种训练期尺度的逐点预测最大差异约为 `2.27e-13 kW`。因此没有创建名义不同但预测完全相同的 `global_v2`。

本轮只要求预测器实际依赖的 target 96 点、上周同期 96 点、最近 4 小时残差两端完整，不要求 Residual LSTM 的全部 672 点历史连续。结果目录为 `my_outputs/scale_independent_transfer_20261015T113530_630644Z`。#3/#4 test 未打开，第三电站未发现或读取。
### 非线性相对残差候选

`select_global_relative_correction_v2.py` 搜索公式：

`last_week + beta * station_scale * clip(recent_residual / station_scale, -cap, cap)`。

其中 `station_scale` 只由各站训练期负荷得到。#3/#4 等权 train-only 搜索选择 `beta=0.35, cap=1.0`，但该截断在两个 validation 上均未激活，结果与 `global_v1` 相同。因此此实验没有晋级为 `global_v2`。结果保存在 `my_outputs/global_relative_correction_v2_20261015T114604_446581Z`。
### Horizon 衰减诊断

`select_global_horizon_decay_v2.py` 在 train 上搜索
`last_week[h] + beta * exp(-h/tau) * recent_residual_mean_4h`，并把 `tau=∞` 的常数修正作为候选之一。两站等权训练目标最终选择 `beta=0.35, tau=∞`，即原有 `global_v1`；validation 结果也完全相同。因此数据不支持为当前预测器增加 horizon 衰减参数。完整结果位于 `my_outputs/global_horizon_decay_v2_20261015T114756_881623Z`。
### 第三电站盲测预注册

第三电站解封前已建立 `THIRD_STATION_BLIND_TEST_PROTOCOL_zh.md`，并冻结 `evaluate_sealed_station_global_v1.py`。通过标准预先定义为 global_v1 的 MAE 严格优于 Last-week 且 RMSE 不劣于 Last-week。模型、评估器和协议 SHA-256 记录在 `model_checkpoints/recent_residual_global_v1/blind_test_freeze_manifest.json`。评估器已用 #3 validation 做管线校验，但第三电站仍未发现或读取。
### #1 未见电站一次性盲测结果

冻结的 `global_v1` 在 #1 的2024年11月完成一次零样本盲测，共2090个可用 origin。Last-week MAE/RMSE 为21.619076/28.926073 kW，global_v1 为20.978116/27.907445 kW，分别改善2.9648%和3.5215%，按预注册标准通过。MAPE由6.674489%轻微升至6.677963%，因此不能表述为所有指标均改善。该零样本结果永久保留，后续若使用 #1 调参只能另记为站点适配实验。
