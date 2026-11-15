# 编译器

torch.fx 图 → Neuron 描述 → MDLA 可执行。

## 文件

| 文件 | 作用 |
|---|---|
| `lowering.py` | **核心**：遍历 fx 图，转成 `OPD/OP/IO` 文本 |
| `shape_infer.py` | 形状推导 + 图校验 + opcode 定义 |
| `host.c` | C 运行时：加载 adapter、编译、执行、缓存、blob 部署 |
| `host` | 编译产物（`clang -O2 -o host_new host.c -ldl`），不入库 |

对外只依赖设备 `/vendor/lib64` 里的 Neuron adapter，其余全部 `dlopen`。

## 图格式

```
OPD <id> <type> <nd> <dims...> <scale> <zp> [w <权重文件> | a <标量>]
OP <opcode> <n_in> <in...> <n_out> <out...>
IO <n_in> <in...> <n_out> <out...>
END
```

**type**：0=FLOAT32标量 1=INT32标量 3=TENSOR_FLOAT32 4=TENSOR_INT32
7=QUANT16_SYMM 8=TENSOR_FLOAT16 11=QUANT8_SYMM_PER_CHANNEL

> 实际运算只支持 fp16（type 8）。type 3 在 `relax 1`（默认）下会被 adapter 降成 fp16，
> `relax 0` 时编译失败——硬件没有 fp32 通路。type 7/11 是 W8A16 量化路径用的，
> 见 docs/operators.md。

**结尾标记**：`i`=输入 `o`=输出 `w <path>`=权重文件 `a <int>`=立即数

## host 命令（24 条）

启动时先吐一行 banner，**必须先读掉**，否则后续回复全部错位：

```json
{"ok":true,"devices":["mtk-gpu","mtk-dsp","mtk-mdla"],"ready":true}
```

### 构建与执行

| 命令 | 作用 |
|---|---|
| `{"cmd":"build","path":...}` | 加载图文本 |
| `{"cmd":"compile"}` | 编译（可写缓存） |
| `{"cmd":"run","spec":...}` | 执行一张图，输入输出走文件 |
| `{"cmd":"chain","spec":...}` | 多图串联，中间结果走内存 |

### blob 部署（生产路径）

| 命令 | 作用 |
|---|---|
| `{"cmd":"export","path":...}` | 导出编译好的 blob |
| `{"cmd":"restore","path":...}` | 从 blob 恢复（仍需同构图先 build） |
| `{"cmd":"bload","list":...}` | 批量加载 blob 列表（一行一个路径） |
| `{"cmd":"map","idx":N}` | 映射第 N 个 blob 供执行 |
| `{"cmd":"unmap"}` | 解除映射 |
| `{"cmd":"rpush","path":...}` | 把 blob 直接恢复进新 slot，**跳过 build 与 compile** |
| `{"cmd":"pop"}` | 释放当前 slot（回收 IOVA） |

多块模型的正确姿势是 `bload` → 逐块 `map` / `run` / `unmap`。每块常量必须常驻，
所以块数受 IOVA 配额约束（详见 docs/performance.md）。

### 槽与配置

| 命令 | 作用 |
|---|---|
| `{"cmd":"create"}` | 手动建一个空 model |
| `{"cmd":"use","slot":N}` | 切换模型槽 |
| `{"cmd":"cachedir","path":...}` | 设编译缓存目录 |
| `{"cmd":"cachekey","v":...}` | 设缓存 key |
| `{"cmd":"relax","v":0|1}` | fp32 是否降级 fp16（默认 1） |
| `{"cmd":"nodev","v":0|1}` | 是否不指定设备 |
| `{"cmd":"poolclear"}` | 清空内存池 |

### 诊断

| 命令 | 作用 |
|---|---|
| `{"cmd":"ops"}` | 列出已注册算子 |
| `{"cmd":"sup"}` | 查询当前 model 在 `mtk-mdla` 上哪些算子**不支持** |
| `{"cmd":"dump","dir":...}` | 把所有输出写到 `<dir>/o<n>.bin`（标定用） |
| `{"cmd":"dbg","path":...}` | 开启 Neuron 调试报告（`setReportPath` + `createForDebug`） |

### 杂项

| 命令 | 作用 |
|---|---|
| `{"cmd":"ping"}` | 心跳 |
| `{"cmd":"quit"}` | 退出 |

## 关键数字

- `MAXOP=65536`（操作数上限，改这个能支持更多层）
- `MAXSLOT=16`（模型槽）
- `MAXIO=64`（输入输出数）
- `MAXBLOB`（bload 的 blob 数上限）
- 实测 1 层 DiT = 284 算子 / 519 operand / 图 30KB
- **编译产物**：1 层 = 132MB `.dla`
