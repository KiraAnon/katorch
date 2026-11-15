# 编译器

torch.fx 图 → Neuron 描述 → MDLA 可执行。

## 文件

| 文件 | 作用 |
|---|---|
| `lowering.py` | **核心**：遍历 fx 图，转成 `OPD/OP/IO` 文本 |
| `shape_infer.py` | 形状推导 + 图校验 + opcode 定义 |
| `host.c` | C 运行时：加载 adapter、编译、执行、缓存 |
| `host` | 编译产物（`clang -O2 -o host host.c -ldl`） |

## 图格式

```
OPD <id> <type> <nd> <dims...> <scale> <zp> [w <权重文件> | a <标量>]
OP <opcode> <n_in> <in...> <n_out> <out...>
IO <n_in> <in...> <n_out> <out...>
END
```

**type**：0=FLOAT32标量 1=INT32标量 3=TENSOR_FLOAT32 4=TENSOR_INT32 8=TENSOR_FLOAT16

**结尾标记**：`i`=输入 `o`=输出 `w <path>`=权重文件 `a <int>`=立即数

## host 命令

| 命令 | 作用 |
|---|---|
| `build <path>` | 加载图 |
| `compile` | 编译（写缓存） |
| `run <spec>` | 执行 |
| `chain <spec>` | 多图串联（内存传递） |
| `cachedir <path>` | 设缓存目录 |
| `cachekey <v>` | 设缓存 key |
| `relax <0/1>` | fp32 是否降级 fp16 |
| `nodev <0/1>` | 是否不指定设备 |
| `use <slot>` | 切换模型槽 |
| `poolclear` | 清空内存池 |
| `ping` | 心跳 |
| `quit` | 退出 |

## 关键数字

- `MAXOP=65536`（操作数上限，改这个能支持更多层）
- `MAXSLOT=16`（模型槽）
- `MAXIO=64`（输入输出数）
- 实测 1 层 DiT = 284 算子 / 519 operand / 图 30KB
- **编译产物**：1 层 = 132MB `.dla`
