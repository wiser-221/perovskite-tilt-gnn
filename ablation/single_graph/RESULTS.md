# Single-graph 消融实验结果

## 实验目的

检验双图差分是否优于普通单图绝对形成能回归。Single-graph 只输入候选结构，直接预测 CHGNet 代理形成能；不输入同组成参考结构，也不使用 `delta_head` 或 `anchor_head`。模型规模和五个 seed 保持与基础 AngleGNN 一致。

标签为 CHGNet 代理标签，不是新计算的 DFT 真值。训练、验证、测试候选数为 800/200/400；测试集不用于训练、早停或 checkpoint 选择，但 Full 模型此前已在同一封存测试划分上评估，因此这里是配对消融比较。

## 结果

| 模型 | 验证绝对形成能 MAE (eV/atom) | 封存测试绝对形成能 MAE (eV/atom) | 测试集平均 seed 分歧 (eV/atom) |
|---|---:|---:|---:|
| Full 双图差分 | 0.003547 | 0.006109 | 0.009898 |
| Single-graph | 0.007013 | 0.009274 | 0.013203 |

在相同测试划分上，Single-graph 的绝对形成能 MAE 比 Full 双图模型高约 **0.003165 eV/atom（约51.8%）**，seed 分歧也更高。这支持双图差分结构对当前角度插值任务有实际价值。不过 Single-graph 没有单独的 ΔE 输出，因此不应把它与双图 ΔE MAE 直接等价比较。

五个 seed 均训练 80 轮；最佳验证 epoch 分别为：42→79、123→77、2026→79、3407→76、7777→75。每个模型有 886,657 个参数。

## 训练与显卡记录

- GPU：NVIDIA GeForce RTX 5060，BF16 混合精度。
- 共 400 个 epoch，逐 epoch 计时累计约 **381秒（6分21秒）**，在一小时限制内完成。
- PyTorch 记录峰值分配显存约 **1,848 MiB**。
- 外部 GPU 采样期间利用率约 **70–85%**，整卡显存约 **4.5 GiB / 8.15 GiB**，温度约 **63–66°C**。

## 文件

- `train_single_graph.py`：训练与测试评估脚本。
- `single_graph_models.pt`：五个 seed 的最佳 checkpoint。
- `training_history.csv`：400 条逐 epoch 训练记录。
- `single_graph_results.json`：机器可读集成结果。

前两项实验分别位于 [`../no_anchor/`](../no_anchor/) 和 [`../generic_angle/`](../generic_angle/)。
