# GraphData 图输入构建详解

`graph_data.py` 负责把 CIF 晶体结构和 CSV 属性转换成 AngleGNN 可以读取的张量。它不执行模型训练，也不会把所有信息强行合成一个大矩阵，而是分别保存节点、边、三体角和晶体级信息。

## 1. 整体流程

```text
CIF 晶体结构
  ├─ 元素信息                → 节点张量
  ├─ 5 Å 内的周期邻居关系    → 边张量
  └─ 同一中心的两条边组合    → 三体角张量

CSV 属性
  ├─ 元素、氧化态、容忍因子  → 组成特征
  └─ decomposition energy    → 预测标签
```

设一个晶体包含：

- `N` 个节点（原子）；
- `E` 条有向边；
- `T` 个三体角。

## 2. 节点信息

### 2.1 原子序数 `z`

每个原子用原子序数表示。例如：

```text
原子：Ba  Ti  O  O  O
z  = [56, 22, 8, 8, 8]
```

尺寸为：

```text
z.shape = [N]
```

`z` 进入 AngleGNN 后才通过嵌入层从 `[N]` 转换成节点特征矩阵 `[N, 128]`。

### 2.2 位点角色 `role`

程序将原子所属位置编码为：

```text
O 位 = 0
A 位 = 1
B 位 = 2
```

例如：

```text
role = [1, 2, 0, 0, 0]
```

尺寸为：

```text
role.shape = [N]
```

`role` 当前主要用于判断三体角类型，并没有直接拼入 AngleGNN 的节点特征。

## 3. 边信息

### 3.1 搜索周期性邻居

程序以每个原子为中心：

```text
搜索半径 cutoff = 5.0 Å
每个中心最多保留 max_neighbors = 16 个邻居
```

搜索考虑周期性边界，因此邻居可以来自相邻晶胞。

### 3.2 边索引 `edge_index`

每条边表示：

```text
中心原子 ← 邻居原子
```

假设存在：

```text
边0：Ti(1) ← O(2)
边1：Ti(1) ← O(3)
边2：O(2)  ← Ti(1)
边3：O(3)  ← Ti(1)
```

那么：

```text
edge_index =
[
  [1, 1, 2, 3],   ← 中心原子
  [2, 3, 1, 1]    ← 邻居原子
]
```

尺寸为：

```text
edge_index.shape = [2, E]
```

每一列对应一条有向边。

### 3.3 距离 `distance`

每条边对应一个原子间距离：

```text
distance = [2.00, 2.05, 2.00, 2.05]
distance.shape = [E]
```

它与 `edge_index` 的列严格对应：

| 边编号 | 中心 | 邻居 | 距离 |
|---:|---:|---:|---:|
| 0 | 1 | 2 | 2.00 Å |
| 1 | 1 | 3 | 2.05 Å |
| 2 | 2 | 1 | 2.00 Å |
| 3 | 3 | 1 | 2.05 Å |

### 3.4 周期边向量 `edge_vector`

程序计算从中心原子指向邻居原子的三维笛卡尔向量：

```text
edge_vector =
[
  [ 2.00,  0.00, 0.00],
  [ 0.00,  2.05, 0.00],
  [-2.00,  0.00, 0.00],
  [ 0.00, -2.05, 0.00]
]
```

尺寸为：

```text
edge_vector.shape = [E, 3]
```

每一行都是 `[Δx, Δy, Δz]`。其计算过程为：

```text
(邻居分数坐标 + 周期镜像 - 中心分数坐标) × 晶格矩阵
```

## 4. 三体角信息

### 4.1 从两条边组成角度

指向同一个中心原子的两条边可以组成一个三体角。例如：

```text
边0：Ti ← O₁
边1：Ti ← O₂

组成：O₁—Ti—O₂
```

每个中心最多选择距离最近的8条边参与角度组合。

### 4.2 三体边索引 `triplet_edge_index`

三体角通过组成它的两条边的编号表示，而不是直接记录三个原子编号。

例如角0由边0和边1组成：

```text
triplet_edge_index =
[
  [0],
  [1]
]

triplet_edge_index.shape = [2, T]
```

如果三个角分别由 `(0,1)`、`(0,4)`、`(1,4)` 三对边构成：

```text
triplet_edge_index =
[
  [0, 0, 1],
  [1, 4, 4]
]
```

### 4.3 角度余弦 `triplet_cosine`

程序使用两条边向量计算：

```text
cosθ = (vj · vk) / (|vj||vk|)
```

假设三个角为 `180°、90°、120°`：

```text
triplet_cosine = [-1.0, 0.0, -0.5]
triplet_cosine.shape = [T]
```

保存 `cosθ` 而不是原始角度，可以保持整体旋转不变性，并将数值固定在 `[-1, 1]`。

### 4.4 三体类型 `triplet_type`

每个角根据中心和邻居的位点角色编码为：

```text
0 = OTHER
1 = B—O—B
2 = O—B—O
3 = A—O—B
```

例如：

```text
triplet_type = [2, 1, 3]
triplet_type.shape = [T]
```

三个三体张量按位置对应：

```text
角编号                  0       1       2
                       ────────────────────
triplet_edge_index    (0,1)   (0,4)   (1,4)
triplet_cosine         -1.0     0.0    -0.5
triplet_type              2       1       3
```

## 5. 晶体级信息

### 5.1 组成特征 `composition_features`

程序按 `a1、a2、b1、b2` 的顺序，为每个位点保存：

```text
元素原子序数 + 氧化态
```

然后加入 Goldschmidt `t` 和 Bartel `τ`：

```text
[
 a1原子序数, a1氧化态,
 a2原子序数, a2氧化态,
 b1原子序数, b1氧化态,
 b2原子序数, b2氧化态,
 Goldschmidt_t,
 Bartel_tau
]
```

尺寸固定为：

```text
composition_features.shape = [10]
```

该向量用于组成基线模型，当前 AngleGNN 前向传播不使用它。

### 5.2 预测标签 `target`

每个晶体对应一个分解能标签：

```text
target = decomposition_energy_per_atom
```

单个晶体的 `target` 是一个标量。

## 6. 单个晶体的最终数据形式

`structure_to_graph()` 最终返回一个张量字典：

```text
graph = {
    z:                    [N]
    role:                 [N]

    edge_index:           [2, E]
    distance:             [E]
    edge_vector:          [E, 3]

    triplet_edge_index:   [2, T]
    triplet_cosine:       [T]
    triplet_type:         [T]

    composition_features: [10]
    target:               标量
    material_id:          字符串
}
```

这些张量的索引关系为：

```text
节点 ← edge_index ← 边 ← triplet_edge_index ← 三体角
```

- `edge_index` 将边连接到节点；
- `triplet_edge_index` 将三体角连接到边。

## 7. 多个晶体怎样拼成 batch

假设：

```text
晶体A：N₁ = 3，E₁ = 4，T₁ = 1
晶体B：N₂ = 2，E₂ = 2，T₂ = 1
```

### 7.1 节点向量首尾拼接

```text
晶体A z = [56, 22, 8]
晶体B z = [38, 40]

batch z = [56, 22, 8, 38, 40]
```

尺寸为：

```text
[N₁ + N₂] = [5]
```

`role` 使用相同方式拼接。

### 7.2 拼接边时增加节点偏移量

晶体A的边：

```text
edge_index_A =
[
  [0, 1, 1, 2],
  [1, 0, 2, 1]
]
```

晶体B原本使用自己的局部节点编号：

```text
edge_index_B =
[
  [0, 1],
  [1, 0]
]
```

晶体A占用了全局节点 `0、1、2`，因此晶体B的节点编号全部加3：

```text
edge_index_B + 3 =
[
  [3, 4],
  [4, 3]
]
```

最后按列拼接：

```text
batch edge_index =
[
  [0, 1, 1, 2, 3, 4],
  [1, 0, 2, 1, 4, 3]
]

shape = [2, E₁+E₂] = [2, 6]
```

### 7.3 拼接三体角时增加边偏移量

晶体A有4条边，所以晶体B的三体角边编号全部加4：

```text
晶体A triplet_edge_index =
[
  [0],
  [1]
]

晶体B原始 triplet_edge_index =
[
  [0],
  [1]
]

晶体B加偏移后 =
[
  [4],
  [5]
]
```

拼接结果：

```text
batch triplet_edge_index =
[
  [0, 4],
  [1, 5]
]

shape = [2, T₁+T₂] = [2, 2]
```

### 7.4 `batch` 记录节点所属晶体

```text
晶体A有3个节点 → [0, 0, 0]
晶体B有2个节点 → [1, 1]

batch = [0, 0, 0, 1, 1]
batch.shape = [N总]
```

模型最后根据这个向量，把节点特征重新汇总成晶体特征。

### 7.5 晶体级特征按行堆叠

组成特征：

```text
composition_features =
[
  [晶体A的10个数],
  [晶体B的10个数]
]

shape = [B, 10]
```

标签：

```text
target = [target_A, target_B]
target.shape = [B]
```

## 8. 最终送入模型的 batch

```text
z                    [N总]
role                 [N总]

edge_index           [2, E总]
distance             [E总]
edge_vector          [E总, 3]

triplet_edge_index   [2, T总]
triplet_cosine       [T总]
triplet_type         [T总]

batch                [N总]
composition_features [B, 10]
target               [B]
```

其中：

```text
N总 = batch中所有晶体的节点数之和
E总 = batch中所有晶体的边数之和
T总 = batch中所有晶体的三体角数之和
B   = batch中的晶体数量
```

最终转换主线为：

```text
原子元素
  → z [N]

原子坐标 + 周期晶格
  → edge_index [2,E]
  → distance [E]
  → edge_vector [E,3]

同一中心的两条边
  → triplet_edge_index [2,T]
  → triplet_cosine [T]
  → triplet_type [T]

多个晶体
  → 节点、边、三体角分别拼接
  → 修正节点索引和边索引
  → 得到一个 batch
```

真正的特征矩阵拼接和逐层维度变化发生在 `models.py` 中；`graph_data.py` 只负责构建这些彼此关联、长度不同的基础张量。
