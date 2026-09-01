# AngleGNN 模型结构与矩阵变化详解

本文只讲解 `models.py` 中的 `AngleGNN`：GraphData 的输入怎样进入网络，以及节点、边、三体角和晶体矩阵在每一层如何变化。

## 1. 模型实际使用的 GraphData

```text
z                    [N]       原子序数
edge_index           [2, E]    原子连接关系
distance             [E]       每条边的距离
triplet_edge_index   [2, T]    哪两条边组成三体角
triplet_cosine       [T]       三体角余弦
triplet_type         [T]       三体角类型
batch                [N]       节点所属晶体编号
```

以下数据不直接进入 AngleGNN：

```text
role                 已用于 GraphData 构建 triplet_type
edge_vector          已用于 GraphData 计算 triplet_cosine
composition_features 只供组成基线模型使用
target               真实标签，用于计算损失
```

## 2. 符号与默认尺寸

```text
N = batch 中的总节点数
E = batch 中的总有向边数
T = batch 中的总三体角数
B = batch 中的晶体数

H = hidden_dim = 128
R = radial_dim = 32
A = angle_dim = 16
L = layers = 4
```

整体结构：

```text
GraphData
   ↓
节点嵌入 + 距离展开
   ↓
初始边特征
   ↓
角度基 + 角类型嵌入
   ↓
AngleLayer × 4
   ↓
晶体平均池化
   ↓
输出头
   ↓
预测分解能
```

## 3. 节点嵌入

原子序数通过可训练的元素嵌入表：

```text
z [N]
  ↓ Embedding(119,128)
X⁰ [N,128]
```

`X⁰` 每一行对应一个原子，每个原子由128个模型自动学习的隐藏特征表示。

## 4. 距离展开

每条边的一个距离标量通过32个高斯径向基展开：

```text
distance [E]
      ↓ GaussianBasis(0, cutoff, 32)
R [E,32]
```

`R` 的每一行是一条边的32维距离表示。

## 5. 初始边特征

`edge_index` 的第一行是中心节点，第二行是邻居节点。模型根据它取出：

```text
X⁰center   [E,128]
X⁰neighbor [E,128]
R          [E,32]
```

沿最后一个维度拼接：

```text
Q⁰ = [X⁰center, X⁰neighbor, R]

[E,128] + [E,128] + [E,32]
                  ↓
               [E,288]
```

每一行的形式是：

```text
中心节点128维 | 邻居节点128维 | 距离32维
```

然后进入边嵌入 MLP：

```text
[E,288]
   ↓ Linear(288 → 128)
   ↓ SiLU
   ↓ Linear(128 → 128)
F⁰ [E,128]
```

此时每条边已经同时包含中心元素、邻居元素和原子间距离信息。

## 6. 角度和角类型表示

### 6.1 角度余弦展开

```text
triplet_cosine [T]
         ↓ GaussianBasis(-1,1,16)
angle_basis [T,16]
```

### 6.2 三体角类型嵌入

角类型编码：

```text
0 = OTHER
1 = B—O—B
2 = O—B—O
3 = A—O—B
```

经过可训练嵌入表：

```text
triplet_type [T]
       ↓ Embedding(4,128)
type_features [T,128]
```

其中：

```text
angle_basis  表示角度具体有多大
type_features 表示角由什么化学位点组成
```

## 7. AngleLayer 的输入

每个 AngleLayer 接收：

```text
节点特征              X [N,128]
边特征                F [E,128]
edge_index              [2,E]
triplet_edge_index      [2,T]
angle_basis             [T,16]
type_features           [T,128]
triplet_mask            [T]
```

当前完整模型使用 `angle_scope="typed"`，所有三体角都参与计算，并使用角类型嵌入。

每层的内部路线是：

```text
三体角 → 更新边 → 更新节点
```

## 8. 选择组成角的两条边

例如：

```text
triplet_edge_index =
[
  [0, 2, 4]
  [1, 3, 5]
]
```

表示：

```text
角0 = 边0 + 边1
角1 = 边2 + 边3
角2 = 边4 + 边5
```

模型根据索引从边矩阵中取出：

```text
Fj = F[edge_j] [T,128]
Fk = F[edge_k] [T,128]
```

`triplet_edge_index` 不作为普通数值拼接；它负责选择组成每个角的两条边。

## 9. 三体角矩阵拼接

每个三体角使用四部分信息：

```text
Fj + Fk          [T,128]  两条边的共同信息
|Fj - Fk|        [T,128]  两条边的差异信息
angle_basis      [T,16]   角度大小
type_features    [T,128]  角度化学类型
```

拼接结果：

```text
Qtriplet = [Fj+Fk, |Fj-Fk|, angle_basis, type_features]

[T,128] + [T,128] + [T,16] + [T,128]
                            ↓
                         [T,400]
```

因为：

```text
128 + 128 + 16 + 128 = 400
```

每一行的形式是：

```text
边和128维 | 边差128维 | 角度16维 | 类型128维
```

## 10. 三体消息 MLP

三体矩阵进入：

```text
[T,400]
   ↓ Linear(400 → 128)
   ↓ SiLU
   ↓ Linear(128 → 128)
Mtriplet [T,128]
```

每个三体角由此生成一个128维消息。

## 11. 三体消息聚合到边

一个三体角由两条边组成，因此同一个三体消息会传给对应的两条边。一个边收到多个角度消息时，程序对消息求平均：

```text
三体消息 [T,128]
       ↓ 根据 triplet_edge_index 分组并求平均
边消息 Medge [E,128]
```

## 12. 更新边特征

当前边特征和边消息拼接：

```text
[F, Medge]

[E,128] + [E,128]
         ↓
       [E,256]
```

进入边更新 MLP：

```text
[E,256]
   ↓ Linear(256 → 128)
   ↓ SiLU
   ↓ Linear(128 → 128)
ΔF [E,128]
```

然后执行残差连接和归一化：

```text
Fnew = LayerNorm(F + ΔF)

[E,128] → [E,128]
```

更新后的边特征已经融合周围三体角信息。

## 13. 边消息聚合到节点

更新后的边特征先通过节点消息 MLP：

```text
Fnew [E,128]
      ↓ MLP(128 → 128 → 128)
边节点消息 [E,128]
```

再根据 `edge_index[0]` 将边消息聚合到中心节点：

```text
边节点消息 [E,128]
         ↓ 按中心节点分组并求平均
Mnode [N,128]
```

## 14. 更新节点特征

当前节点特征和节点消息拼接：

```text
[X, Mnode]

[N,128] + [N,128]
         ↓
       [N,256]
```

进入节点更新 MLP：

```text
[N,256]
   ↓ Linear(256 → 128)
   ↓ SiLU
   ↓ Linear(128 → 128)
ΔX [N,128]
```

执行残差连接和归一化：

```text
Xnew = LayerNorm(X + ΔX)

[N,128] → [N,128]
```

更新后的节点特征包含元素、邻居、距离、三体角和角类型信息。

## 15. 四层 AngleLayer

模型连续执行四层：

```text
AngleLayer 1：X⁰ [N,128], F⁰ [E,128] → X¹ [N,128], F¹ [E,128]
AngleLayer 2：X¹ [N,128], F¹ [E,128] → X² [N,128], F² [E,128]
AngleLayer 3：X² [N,128], F² [E,128] → X³ [N,128], F³ [E,128]
AngleLayer 4：X³ [N,128], F³ [E,128] → X⁴ [N,128], F⁴ [E,128]
```

各层尺寸保持不变，但信息传播范围逐层扩大。

## 16. 晶体平均池化

经过四层后：

```text
X⁴ [N,128]
```

内部的 `batch [N]` 记录每个节点属于哪个晶体。例如：

```text
batch = [0,0,0,1,1]
```

模型将同一晶体的节点特征求平均：

```text
X⁴ [N,128]
    ↓ 按 batch 分组平均
C [B,128]
```

`C` 的每一行代表一个完整晶体。

## 17. 输出头

```text
C [B,128]
   ↓ Linear(128 → 128)
   ↓ SiLU
  [B,128]
   ↓ Linear(128 → 1)
  [B,1]
   ↓ squeeze
预测分解能 [B]
```

训练时将预测分解能 `[B]` 与真实标签 `target [B]` 比较并计算损失。

## 18. 完整矩阵变化主线

```text
z [N]
  ↓ 原子嵌入
X⁰ [N,128]

distance [E]
  ↓ 32维径向基
R [E,32]

[X中心, X邻居, R]
[E,128+128+32]
  ↓ 拼接
[E,288]
  ↓ 边嵌入 MLP
F⁰ [E,128]

triplet_cosine [T]
  ↓ 16维角度基
A [T,16]

triplet_type [T]
  ↓ 类型嵌入
P [T,128]

[Fj+Fk, |Fj-Fk|, A, P]
[T,128+128+16+128]
  ↓ 拼接
[T,400]
  ↓ 三体 MLP
[T,128]
  ↓ 聚合到边
[E,128]

[原边, 边消息]
[E,128+128]
  ↓ 拼接
[E,256]
  ↓ 边更新 MLP + 残差 + LayerNorm
F¹ [E,128]

边特征 [E,128]
  ↓ 聚合到中心节点
节点消息 [N,128]

[原节点, 节点消息]
[N,128+128]
  ↓ 拼接
[N,256]
  ↓ 节点更新 MLP + 残差 + LayerNorm
X¹ [N,128]

以上 AngleLayer 重复4次
  ↓
X⁴ [N,128]

按晶体平均池化
X⁴ [N,128] → C [B,128]

输出头
C [B,128] → [B,1] → 预测分解能 [B]
```

核心消息传递路线：

```text
节点 + 距离
      ↓
     边
      ↓
两条边 + 角度 + 角类型
      ↓
  三体角消息
      ↓
   更新边
      ↓
   更新节点
      ↓
  晶体级预测
```

AngleGNN 在每一层执行一次“三体角 → 边 → 节点”，连续四层，使角度和局部旋转环境贯穿整个晶体表征过程。
