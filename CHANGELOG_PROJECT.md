# 项目改动记录

## 0.4.0 — 2026-10-15

1. 使用4个带96步标签隔离的 train-only 向前滚动 fold 验证 recent-residual correction。
2. 常数 beta=0.40 在冻结 validation 上优于分段 horizon beta，保留为当前推荐方案。
3. 新增 `load15_recent_residual_predictor.py` 和 `predict_recent_residual.py`：按 origin 输出未来96点预测。
4. 增加严格因果检查、未来数据扰动不变测试、recent residual 缺失回退和 Last-week 缺失硬失败。
5. 未训练新模型，test 未预测、未评分。
6. 新增版本化配置 `model_checkpoints/recent_residual_v1/model.json`，推理时校验 horizon、lag、时区和单位契约。
7. 使用正式推理函数重放全部1169个 validation origin：1153次 correction、16次回退；整体 MAE=19.239217 kW。
8. 新增站点 `#4` 保守质量处理和零适配验证：重复分钟全部失效、负值失效；原始文件不修改。
9. `#3 beta=0.40` 零适配到 `#4` 后 MAE=59.604651 kW，优于 `#4` Last-week 63.498017 kW；新增 `station4_load15_config.json` 复用同一推理入口。
10. 仅使用 `#3/#4` train、按站点等权选择共享 beta=0.35，并冻结 `recent_residual_global_v1`；validation 未参与参数选择。
11. 共享 beta 在两站 validation 上均改善 MAE/RMSE/MAPE/R²；第三电站未发现、未读取，继续封存。

## 0.3.0 — 2026-09-28

第一轮正式直接负荷 LSTM 训练与 validation 评估。

1. 五个 `#3` 分钟文件先合并后聚合，固定 target period；有效样本 train 1192 / validation 1169 / test 1617。
2. 新增 `refresh_industrial_aug_dec.py`，生成三个带精确历史 context 的业务 CSV 与数据质量报告。
3. `my_load15_config.json` 与 `load15_direct_dataset.py` 已切换至 target-based split；scaler 只拟合训练数据。
4. 新增 `train_direct_lstm_formal.py`；基础 encoder/decoder LSTM(32)，batch=32，最多100 epochs，EarlyStopping 与 ReduceLROnPlateau。
5. 实际训练31 epochs，最佳第21 epoch，最佳 val_loss=0.0282039233；最佳权重保存为 `best_model.keras`。
6. Validation：LSTM MAE=44.788800 kW、RMSE=72.686732 kW、MAPE=11.146815%、R²=0.970825，未超过昨天同期和上周同期基线。
7. 保存完整训练历史、逐 horizon、逐 origin 日期及两类示例预测。test 未传入 predict/evaluate，仍未评分。

## 0.2.0 — 2026-09-21

直接负荷 LSTM 第一版与短试跑（非正式）。

1. 新增 `load15_direct_dataset.py`：独立组装 `(672,5)+(96,4)→(96,)`，复用既定切分/缺失/标准化规则。
2. 新增 `load15_direct_lstm.py`：编码器–解码器 LSTM(32)，Adam + 标准化空间 MSE；预测直接负荷而非残差。
3. 新增 `02_load15_direct_lstm.ipynb`（由 `_write_02_notebook.py` 生成）：干净内核可独立运行。
4. 在 `load15` 环境安装 `tensorflow==2.16.2`、`keras==3.4.1`；numpy/pandas 未更换；`pip check` 通过。
5. 短试跑已完成：46 训练 / 13 验证，batch=16，epochs=2；日志见 `my_outputs/direct_lstm_trial_20260921T081946_701356Z_0ee1efa4/`。
6. **正式训练尚未执行**；后续须重新初始化模型与优化器，不继承本次试跑权重。
7. 未运行第01本第9节；未改工业聚合流程；未把两轮试跑写成正式成绩。

## 0.1.0 — 2026-09-19

基于2026-09-18教学包另存项目版，不改动原仓库或旧教学文件。

1. 明确资料检查日期、项目整理日期、实际执行日期和数据截止日期的区别。
2. 增加实际参与运算的JSON集中配置及参数校验。
3. 参数化训练/验证划分比例，并保留目标窗口不跨界的规则。
4. 增加含分钟的当地时间周期特征，分离历史5特征和未来已知4特征。
5. 保存项目版特征顺序、配置、数据指纹及训练状态。
6. 每次执行生成独立输出目录，不覆盖原lesson01结果。
7. 文档分开当前实现、真实数据待验证、LSTM和储能闭环待开发事项。

没有新增：正式训练好的模型、正式测试指标、节假日电价数据、储能控制器。
## 2026-10-15 - 尺度无关与整站迁移诊断

- 新增 `diagnose_scale_independent_transfer.py`，仅打开 #3/#4 的 train 和 validation 文件。
- 使用训练期站内中位绝对负荷、平均绝对负荷、标准差和 IQR 检查残差归一化。
- 数值断言确认：对当前线性 correction，除以站内标量再乘回会严格抵消，不能形成不同于 `global_v1` 的模型。
- 完成 #3→#4、#4→#3 的整站迁移；未训练模型，未打开 test，未发现或读取第三电站。
- 本诊断采用预测器原生可用性口径，不要求 LSTM 的完整 672 点历史连续，因此 origin 数多于早期 LSTM 公平子集。
## 2026-10-15 - 相对残差非线性候选（未晋级）

- 新增 `select_global_relative_correction_v2.py`。
- 候选公式按各站 train-only 中位绝对负荷归一化近期残差，并搜索相对截断阈值与 beta。
- #3/#4 联合 train-only 搜索选择 `beta=0.35, cap=1.0`；该阈值在 validation 未触发截断，预测与 `global_v1` 相同。
- 因候选没有形成稳定、实质性的提升，未创建或冻结 `global_v2`，继续保留 `global_v1`。
- #3/#4 test 未打开，第三电站未发现或读取。
## 2026-10-15 - Horizon 衰减候选（未晋级）

- 新增 `select_global_horizon_decay_v2.py`，比较指数衰减 correction 与常数 correction。
- #3/#4 等权 train-only 搜索选择 `beta=0.35, tau=∞`，即现有 `global_v1`。
- #3→#4 整站迁移同样选择不衰减；#4 单站选择的 24 小时衰减迁移到 #3 后略逊于 `global_v1`。
- 保存完整 1～96 horizon 指标；未创建 `global_v2`，未打开 test，未读取第三电站。
## 2026-10-15 - 第三电站盲测预注册

- 新增 `THIRD_STATION_BLIND_TEST_PROTOCOL_zh.md` 和无调参的一次性评估程序 `evaluate_sealed_station_global_v1.py`。
- 预先固定主指标 MAE、安全指标 RMSE，以及“两者均优于 Last-week”通过标准。
- 使用 #3 validation 完成管线校验，复现 2168 origins、Last-week MAE 20.018066、global_v1 MAE 19.169980。
- 新增冻结哈希清单 `model_checkpoints/recent_residual_global_v1/blind_test_freeze_manifest.json`。
- 第三电站仍未发现或读取；#3/#4 test 未打开。
## 2026-10-15 - 第三电站 #1 一次性零样本盲测

- 按预注册协议固定 `global_v1 beta=0.35`，以 #1 的 2024-11 为盲测期，未进行 #1 参数搜索。
- 严格15/15聚合得到 11712 点，完整11300点；盲测共同可用 origin 为2090。
- Last-week：MAE 21.619076、RMSE 28.926073；global_v1：MAE 20.978116、RMSE 27.907445。
- MAE 改善2.9648%，RMSE改善3.5215%，按预注册标准通过；MAPE轻微变差，已保留。
- #1-1 经月份识别发现为12月，仅检查过结构和首尾，未聚合、评分或用于参数。
