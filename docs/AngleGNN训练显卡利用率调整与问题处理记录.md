# AngleGNN 训练显卡利用率调整与问题处理记录

## 1. 文档范围

本文记录五随机种子 AngleGNN 集成训练过程中，为提高 RTX 5060 利用率、控制 8 GB 显存占用并保持电脑可用性所做的调整，以及训练期间出现的问题和解决方法。

当前训练状态：

- 已完成并测试：seed 42、123、2026；
- 暂停中：seed 3407；
- 尚未开始：seed 7777；
- 训练模型仅为 `AngleGNN(angle_scope="typed")`，没有训练组成基线、距离 GNN 或无角度消融模型。

本文数据来自：

- `outputs/angle_ensemble/gpu_monitor_high_throughput.csv`；
- `outputs/angle_ensemble/gpu_monitor.csv`；
- 三组批量基准测试；
- 各 seed 的 `history.csv`、检查点和测试结果。

## 2. 硬件与软件环境

| 项目 | 配置 |
|---|---|
| GPU | NVIDIA GeForce RTX 5060 |
| 显存 | 8,151 MiB |
| 系统内存 | 7.7 GiB |
| Swap | 2.0 GiB |
| 驱动 | NVIDIA 591.86（`nvidia-smi` 显示接口版本 590.57） |
| 驱动支持 CUDA | 13.1 |
| PyTorch | 2.8.0+cu128 |
| Python | 3.12.14 |
| 混合精度 | BF16 |

系统默认 Python 3.14 没有 PyTorch、NumPy 和 pymatgen，而且不适合作为本次稳定训练环境。因此使用 uv 安装 Python 3.12，并创建项目隔离环境 `.venv`。CUDA 12.8 版 PyTorch 能正确识别 RTX 5060，计算能力为 12.0。

## 3. 训练前的数据瓶颈

原始数据含 59,708 个晶体结构，保存在 7,464 个压缩 JSON 分片中。直接在每个 epoch 解压、解析 CIF 并构图会使 GPU 等待 CPU，无法维持较高利用率。

初始抽样结果：

- CIF 构图约 39.8 个结构/秒；
- 平均每个结构约 22.4 个节点；
- 平均约 358.4 条有向边；
- 平均约 627.2 个三体角。

按照该速度，每个 epoch 重新构图不可接受。因此增加了分片图缓存：

1. 每个原始压缩分片只解析一次；
2. 将节点、边、距离、三体角和角类型保存为 PyTorch 分片；
3. 训练时直接读取已经构建好的图；
4. 以分片为单位随机排序，使相邻样本尽量命中同一个缓存分片；
5. 缓存支持断点续建，已完成分片不会重复处理。

最终生成 7,464 个图缓存分片，约占 1.8 GB。该缓存只保留在本地，不提交 Git。

## 4. GPU 批量基准与第一次优化

正式训练前用 seed 42 分别测试了三个批量。所有基准均采用 BF16，模型有 886,657 个参数。

| 配置 | DataLoader workers | 单 epoch 时间 | GPU 利用率均值 | GPU 利用率峰值 | 总显存峰值 | PyTorch 分配峰值 |
|---|---:|---:|---:|---:|---:|---:|
| batch 24 | 2 | 88.0 s | 24.4% | 33% | 2,657 MiB | 338 MiB |
| batch 96 | 4 | 36.3 s | 55.2% | 76% | 3,899 MiB | 1,130 MiB |
| batch 192 | 4 | 33.4 s | 66.8% | 89% | 5,328 MiB | 2,121 MiB |

结论：

- batch 24 计算粒度太小，CPU 拼图和主机到 GPU 传输占比过高；
- batch 96 已显著改善利用率；
- batch 192 在 8 GB 显存上仍安全，吞吐和 GPU 利用率最好；
- batch 192 相比 batch 96 的 epoch 时间只再缩短约 8%，说明此时 CPU 拼图和小矩阵操作仍占有一定比例。

第一次正式配置因此采用：

```text
batch_size = 192
eval_batch_size = 256
workers = 4
gradient_accumulation = 1
amp = bf16
fused AdamW = true
pin_memory = true
non_blocking transfer = true
```

同时启用了：

- `torch.set_float32_matmul_precision("high")`；
- `torch.backends.cudnn.benchmark = True`；
- BF16 自动混合精度；
- fused AdamW；
- `optimizer.zero_grad(set_to_none=True)`；
- 固定内存和异步传输；
- 分片局部性采样；
- 5 秒一次的 `nvidia-smi` GPU 监控。

## 5. 高吞吐训练表现

seed 42 和 seed 123 使用高吞吐配置完成训练：

| Seed | 训练轮数 | 平均每轮时间 | 最佳轮数 | 验证 MAE | 测试 MAE |
|---:|---:|---:|---:|---:|---:|
| 42 | 100 | 30.85 s | 93 | 0.017119 | 0.016449 |
| 123 | 100 | 31.15 s | 100 | 0.018055 | 0.017023 |

PyTorch 训练峰值：

- seed 42：分配峰值 2,187 MiB，保留峰值 3,236 MiB；
- seed 123：分配峰值 2,207 MiB，保留峰值 3,306 MiB。

两组训练没有出现 CUDA OOM、NaN 或显存逐轮增长，说明单训练任务下 batch 192 是安全的。

## 6. 学习率调度与 GPU 效率之外的收敛策略

训练使用：

```text
初始学习率 = 3e-4
ReduceLROnPlateau factor = 0.5
early stopping patience = 12
min_delta = 1e-5
gradient clip = 5.0
target normalization = 仅使用训练集统计量
```

多个 seed 都出现了验证 MAE 平台期。降低学习率后，验证 MAE 会再次明显下降。例如 seed 2026 在第 64 轮附近处于约 0.0185 的平台，学习率降低后第 65 轮达到 0.01755，之后继续细化至 0.01730。

`min_delta=1e-5` 用于防止把十万分位以下的浮点波动当成有效改善。最佳检查点只在验证 MAE 有实质改善时覆盖。

## 7. 电脑严重卡顿事件

### 7.1 表现

训练过程中电脑出现明显卡顿，诊断时观察到：

- GPU 显存约 7.4 GB / 8.15 GB；
- 系统内存只剩约 80–120 MB 空闲；
- 2 GB Swap 几乎全部占满；
- GPU 利用率在两个训练任务之间竞争；
- 同时存在两个 AngleGNN 主训练进程；
- 两个主进程分别启动了多组 DataLoader 子进程。

### 7.2 根因

交互界面中断后，原训练进程没有实际退出，仍在后台运行。恢复任务时又启动了一套训练，形成两套训练同时访问同一 GPU、同一输出目录和同一批图缓存。

高吞吐训练的 `workers=4` 会分别为训练集、验证集等 DataLoader 建立常驻进程。在 7.7 GiB 内存的机器上，每个子进程都会持有 Python、PyTorch、pymatgen 和文件缓存。两套任务叠加后，内存和 Swap 被耗尽，电脑卡顿的主要原因实际上是系统内存压力，而不仅是 GPU 显存占用。

`gpu_monitor_high_throughput.csv` 的整体统计为：

| 指标 | 均值 | 中位数 | 最大值 |
|---|---:|---:|---:|
| GPU 利用率 | 59.6% | 76% | 100% |
| 总显存使用 | 7,400 MiB | 7,396 MiB | 7,724 MiB |
| 温度 | 58.6°C | 59°C | 66°C |
| 功率 | 71.9 W | 73.8 W | 104.9 W |

注意：该文件后半段包含重复训练事故，因此 7.4 GB 平均显存不能代表正常的单任务 batch 192 配置。正常单任务基准峰值约为 5.3 GB。

### 7.3 处理

采取了以下措施：

1. 用 `nvidia-smi` 和进程树确认两个主训练 PID 及其子进程；
2. 普通 `SIGTERM` 未能及时终止进程后，对精确 PID 使用 `SIGKILL`；
3. 确认 GPU 利用率回落到约 1%，显存回落到桌面基础占用；
4. 确认系统可用内存恢复到约 6.2 GB；
5. 保留已经完整完成的 seed 42 和 seed 123；
6. 将两进程同时写过的 seed 2026 中间目录移入中断备份，不作为正式模型；
7. 从头训练 seed 2026，保证训练历史和检查点一致。

## 8. 从最高吞吐切换为低干扰训练

用户仍需在训练期间正常使用电脑，因此最终没有继续追求最高 GPU 利用率，而是采用资源更保守的配置：

```text
batch_size = 96
eval_batch_size = 128
workers = 0
gradient_accumulation = 1
amp = bf16
```

并对唯一训练进程执行：

```text
nice = 10
I/O class = idle
```

这表示前台应用在 CPU 和磁盘访问上优先于训练。

低干扰监控统计：

| 指标 | 均值 | 中位数 | 最大值 |
|---|---:|---:|---:|
| GPU 利用率 | 42.8% | 48% | 59% |
| 总显存使用 | 3,976 MiB | 3,969 MiB | 4,298 MiB |
| 温度 | 57.0°C | 57°C | 61°C |
| 功率 | 68.2 W | 74.9 W | 78.4 W |

切换后的代价与收益：

| 项目 | 高吞吐配置 | 低干扰配置 |
|---|---:|---:|
| batch size | 192 | 96 |
| workers | 4 | 0 |
| 单 epoch 典型时间 | 31 s | 56–58 s |
| 总显存典型占用 | 约 5.3 GB 以下 | 约 4.0 GB |
| 系统可用内存 | 多进程时压力较大 | 约 4.4 GB |
| 桌面响应 | 训练优先 | 前台优先 |

低干扰模式让 epoch 时间增加约 80%，但消除了常驻 DataLoader 子进程，避免再次耗尽内存和 Swap。

## 9. 低干扰配置下完成的 seed 2026

seed 2026 使用低干扰配置重新训练：

| 项目 | 结果 |
|---|---:|
| 训练轮数 | 88，触发早停 |
| 平均每轮时间 | 56.65 s |
| 最佳轮数 | 76 |
| 验证 MAE | 0.017300 eV/atom |
| 验证 RMSE | 0.038857 eV/atom |
| 测试 MAE | 0.016780 eV/atom |
| 测试 RMSE | 0.037635 eV/atom |
| 测试 R² | 0.889745 |
| 测试 PR-AUC | 0.933671 |
| PyTorch 分配峰值 | 1,157 MiB |
| PyTorch 保留峰值 | 1,842 MiB |

该结果与高吞吐配置下的 seed 42、123 相近，说明降低 batch 和 workers 没有造成明显精度退化。

## 10. GPU 监控实现

训练程序启动后台监控线程，每 5 秒执行一次：

```text
nvidia-smi --query-gpu=
index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw
--format=csv,noheader,nounits
```

记录字段：

```text
timestamp
GPU index
GPU name
GPU utilization (%)
used memory (MiB)
total memory (MiB)
temperature (°C)
power draw (W)
```

每个 epoch 还通过 PyTorch 记录：

- `torch.cuda.max_memory_allocated()`；
- `torch.cuda.max_memory_reserved()`；
- epoch 时间；
- 学习率；
- 训练损失；
- 验证 MAE、RMSE、R² 和 PR-AUC。

系统总显存和 PyTorch 分配显存不同：前者包含桌面、驱动、其他 CUDA 上下文和 PyTorch 保留池；后者只统计当前 PyTorch 进程实际分配。因此两类数据均需保留。

## 11. 后续继续训练时的推荐策略

考虑到这台电脑只有 7.7 GiB 系统内存，继续 seed 3407 和 7777 时推荐保持低干扰配置：

```text
batch_size = 96
eval_batch_size = 128
workers = 0
amp = bf16
nice = 10
ionice = idle
```

启动前必须检查：

```bash
pgrep -af train_angle_ensemble.py
nvidia-smi
free -h
```

只有确认不存在旧训练进程后才能启动新任务。不得仅根据交互界面是否中断判断后台进程是否已经结束。

如果电脑无需同时承担交互工作，可以恢复 batch 192，但仍建议把 workers 从 4 降到 1 或 2，并观察可用内存，不应再同时运行两个训练任务。

## 12. 总结

本次训练经历了三个阶段：

```text
直接构图导致潜在 CPU 瓶颈
→ 建立分片图缓存
→ 批量 24/96/192 基准并选择高吞吐配置
→ seed 42、123 高吞吐训练
→ 交互中断后误启动重复任务，内存与 Swap 耗尽
→ 精确终止重复进程并隔离受污染输出
→ 改为 batch 96、workers 0、低 CPU/I/O 优先级
→ seed 2026 稳定完成
```

最终经验是：RTX 5060 的显存并不是唯一限制。对于含 pymatgen 和大量小图拼接的任务，系统内存、DataLoader 进程数、文件缓存以及是否存在残留训练进程同样决定电脑是否流畅。最高 GPU 利用率和整机可用性之间需要明确取舍；本项目后续应优先采用低干扰配置完成剩余模型。
