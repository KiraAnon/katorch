# KAcustom

核心层库。**所有模型模块都依赖它。**

## 文件

| 文件 | 内容 |
|---|---|
| `layers.py` | `Lin` / `Norm` / `Gelu` / `Silu` / `Rotary` / `Attention` / `MLP` / `AdaLN` / `GatedBranch` |
| `blocks.py` | `QwenBlock` / `AnimaBlock`（用 layers 拼装） |
| `bal_norm.py` | **`BalancedNorm`** —— 残差坐标 + mean-abs 预条件，解决 fp16 溢出 |
| `blockbuild.py` | 底层 nn.Module 封装 |

## BalancedNorm（核心创新）

**问题**：DiT 残差流最大值达 **65 万**，fp16 上限只有 65504，MDLA 又没有 fp32。

**解法**：
```
x_source = d * x_bal,   d = g * r   (d 逐通道，2 的幂)
y = r * x_bal = x_source / g
Norm(x_source) == Norm(y)      (g > 0 标量，Norm 尺度不变)

mean-abs 预条件:
  s = mean(|x|) + sqrt(eps)
  a = 1/s
  z = a*x,  e = a*sqrt(eps)
  v = mean(z^2) + e^2
  out = z * rsqrt(v)
```

**关键性质**（Cauchy-Schwarz）：`mean(z^2)` 恒在 `[0.5, dim]`，**与输入幅度无关**。

**实测**：28 层累积误差 **0.001%**。

## 约束

- 每个 Norm 必须有自己的 `d` 和 `g`（残差跨 5 个数量级）
- `d_in` 用**块输出**边界的量级（不是输入）
- 层间用 `post_scale = d_i / d_{i+1}` 转换（2 的幂，精确）
