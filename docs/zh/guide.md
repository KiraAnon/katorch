# 格式、协议与移植（中文）

接 [README](../../README.md#五分钟跑通) 之后要用的参考。
英文版：[../guide.md](../guide.md)

---

## 一、你面对的是四层

```
你的 PyTorch 模型
        │  lowering.py          trace → 图文本
        ▼
   OPD / OP / IO  纯文本
        │  host.c              build → compile → run
        ▼
   NeuronAdapter（厂商库，从 /vendor/lib64 dlopen）
        │
        ▼
   MDLA 5.0
```

如果你的模型不是 transformer，你绝大部分时间会在第三层（`lowering.py` + `host.c`）。
仓库里那四个模型包（`KAcustom` / `KAllm` / `KAdit` / `KAvae`）是**写好的样例**，不是框架。

### 设备前提

```sh
ls -l /dev/mtk_mdla* /dev/apusys* 2>/dev/null
find /vendor -name "libneuron*" 2>/dev/null
```

**必须 root。** 普通 Termux 权限拿不到 NPU 设备节点。

### 编宿主

```sh
pkg install clang
clang -O2 -o host_new host.c -ldl
```

`host.c` 里所有厂商相关的都用 `dlopen`，所以编译时不需要链任何厂商库。
如果设备上的 Neuron 库不在 `/data/data/com.termux/files/usr/lib/`，改文件顶部的 `LIBDIR`。

---

## 二、图文本格式

一行一条指令，没有续行。

```
OPD <id> <type> <nd> <dim1> ... <dim_nd> <scale> <zp> [<tail>]
OP  <opcode> <n_in> <in_id...> <n_out> <out_id...>
IO  <n_in> <in_id...> <n_out> <out_id...>
END
```

| 指令 | 含义 |
|---|---|
| `OPD` | 声明一个 operand：编号、类型、维数、各维大小、量化 scale、零点、尾部标记 |
| `OP` | 声明一个算子：opcode、输入 operand 编号、输出 operand 编号 |
| `IO` | 图的对外接口：哪些 operand 是输入、哪些是输出 |
| `END` | 结束 |

### 一个完整的图

```text
OPD 0 8 2 256 512 0 0 i
OPD 1 8 2 256 512 0 0 w b.bin
OPD 2 8 2 256 512 0 0 o
OPD 3 4 1 1 0 0 a 0
OP 0 3 0 1 3 1 2
IO 1 0 1 2
END
```

读法：operand 0 是 fp16 `[256,512]` 的图输入；operand 1 是从 `b.bin` 读的 fp16 `[256,512]`
常量；operand 2 是输出；operand 3 是 INT32 立即数 0。然后 `ADD` 吃 operand `0 1 3`，
产出 operand `2`。

这个图算的是 `a + 0`。它是能跑通整条路径的最小东西，所以是仓库里的第一个样例。

### 类型码

| 值 | 类型 | 用途 |
|---|---|---|
| `0` | FLOAT32 标量 | 少用 |
| `1` | **INT32 标量** | CONV 的 pad / stride / 激活码，`nd=0` |
| `3` | TENSOR_FLOAT32 | 会被 `relax 1` 降成 fp16 |
| `4` | **TENSOR_INT32** | 激活码、shape 数组、轴索引 |
| `8` | **TENSOR_FLOAT16** | 主力类型 |

### 尾部标记

| 标记 | 含义 |
|---|---|
| `i` | 图输入 |
| `o` | 图输出（中间结果也可以标） |
| `w <path>` | 从文件读常量 |
| `a <int>` | 立即数（配 type 1 或 4） |

### 三个最容易写错的算子

**FC（9）必须 4 个输入。**

```
OP 9 4 <input> <weight> <bias> <activation_code> 1 <output>
```

`activation_code` **必须是 type 4**。写成 type 1 不会报错——算子会把第一个输入直接透传出去。
图能跑、返回 rc=0、每个数都是错的。

**CONV（3）7 个输入，参数是 0 维 INT32 标量。**

```
OP 3 7 <in> <w> <b> <pad> <stride> <act> <dilation> 1 <out>
```

pad / stride / act / dilation 用 `scalar0()` 生成（type 1 + `nd=0`）。写 type 4 不被接受。

**RESHAPE（22）的目标 shape 放在一个 INT32 常量文件里。**

```python
np.array([1, 1024, 2048], np.int32).tofile("s.bin")
# OPD <id> 4 1 3 0 0 w s.bin
```

**MEAN / REDUCE_SUM / REDUCE_MAX 是 3 个输入**，不是 2 个：数据、轴常量、激活码。
少给一个直接 `-516 UNMAPPABLE`，而且不告诉你是少了参数。

---

## 三、host 协议

`host_new` 从 stdin 一行读一个 JSON，往 stdout 一行写一个回复。

### 3.0 启动 banner（最容易栽的一步）

启动时它先吐一行，之后才接受命令：

```json
{"ok":true,"devices":["mtk-gpu","mtk-dsp","mtk-mdla"],"ready":true}
```

**必须先把它读掉。** 漏读的话后面每一条回复都错位一格，你会去追一个根本不存在的 bug。
banner 还顺便告诉你这台设备有哪些加速器。

### 3.1 命令表

| 命令 | 参数 | 作用 | 返回 |
|---|---|---|---|
| `ping` | — | 心跳 | `{"ok":true,"pong":1}` |
| `build` | `path` | 加载图文本 | `{"ok":true,"ops":N,"slot":N}` |
| `compile` | — | 编译成可执行网络 | `{"ok":true,"rc":0}` |
| `run` | `spec` | 执行一张图 | `{"ok":true,"rc":0}` |
| `chain` | `spec` | 多图串联，结果**内存直传** | `{"ok":true,"rc":0}` |
| `create` | — | 手动建空 model | `{"ok":true,"rc":0}` |
| `export` | `path` | 导出编译好的 blob | `{"ok":true,"bytes":N}` |
| `restore` | `path` | 从 blob 恢复，跳过 build + compile | — |
| `use` | `slot` | 切换模型槽 | — |
| `pop` | — | 释放当前槽（回收 IOVA） | `{"ok":true,"freed_slot":N,...}` |
| `bload` | `list` | 批量加载 blob（一行一个路径） | `{"ok":true,"n":N}` |
| `map` | `idx` | 映射第 idx 个 blob | `{"ok":true,"nin":N,"nout":N}` |
| `unmap` | — | 解除映射 | — |
| `cachedir` | `path` | 设编译缓存目录 | — |
| `cachekey` | `v` | 设缓存 key | — |
| `relax` | `0`/`1` | fp32 是否降级 fp16 | — |
| `nodev` | `0`/`1` | 是否不指定设备 | — |
| `poolclear` | — | 清空内存池 | — |
| `ops` | — | 列出已注册算子 | — |
| `dump` | `dir` | 导出中间张量 | — |
| `quit` | — | 退出 | — |

### 3.2 两条执行路径

**开发期：build + compile。**

```json
{"cmd":"build","path":"/path/to/g.txt"}
{"cmd":"compile"}
{"cmd":"run","spec":"..."}
```

每次都重新解析 + 编译。一层 DiT 约 1.8 s，大图几分钟。

**生产期：bload + map。**

```json
{"cmd":"bload","list":"/path/to/blobs.txt"}
{"cmd":"map","idx":0}
{"cmd":"run","spec":"..."}
```

直接内存映射预编译好的 blob，**毫秒级**。这才是部署姿势。

### 3.3 `run` 的 spec

```
<n_in> <slot> <path> <slot> <path> ... <n_out> <path> [<path> ...]
```

### 3.4 错误码

| 码 | 含义 | 怎么办 |
|---|---|---|
| `-101` | 图文件打不开 | 检查路径 |
| `-102` | `M_create` 失败 | 设备没初始化 |
| `-104` … `-106` | 常量文件读取失败 | 检查 `w` 指向的文件 |
| `-200-rc` | `M_addOp` 失败（OPD） | 多半是 operand 声明 |
| `-300-rc` | `M_setVal` 失败（常量） | 检查大小与类型 |
| `-310-rc` | 立即数设置失败 | 检查 `a` 的写法 |
| `-400-rc` | `M_addOpn` 失败（OP） | 多半是算子不支持 |
| **`-314`** | **`M_addOp` 返回 114** | **常量总量超了 IOVA 配额** |
| `-507` | 编译失败 | 算子不支持或 shape 不合法 |
| `-516` | **UNMAPPABLE** | 这个算子**组合**映射不出来 |
| `-599` | 没有编译好的模型 | 先 `compile` |
| `-601` … `-603` | 输入文件问题 | 检查 spec 路径 |

两个要记住的：

* **`-314` = 「常量装不下」。** 常量张量必须常驻，直到编译产物消失，共用约 1.35 GB 的
  设备级配额。这不是单模型限制，是一个池子——所以模型必须切块。
* **`-516` 是你要长期打交道的那个。** 它是"这个算子组合表达不出来"的唯一信号，
  而且不告诉你是哪个算子。

---

## 四、用 lowering.py 从 PyTorch 建图

```python
import sys
sys.path.insert(0, "/path/to/npt/编译器")
from lowering import Lowering

import torch, torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 256)
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

m = MyModel().eval()

low = Lowering("/tmp/mymodel")
x = low.input((1, 256))            # 声明输入形状
low.lower_module(m, [x])           # symbolic_trace + 走图
low.validate()                     # 出货前先做 shape 校验
open("/tmp/mymodel/g.txt", "w").write(low.graph_text())
```

`lower_module` 内部就是 `fx.symbolic_trace(mod)` → 逐节点转换成 NPU 算子 → 把返回值标成
图输出。

### 它认得的 PyTorch 写法

算术（`add`/`sub`/`mul`/`div`）、激活（`relu`/`silu`/`tanh`/`sigmoid`/`exp`/`log`/`sqrt`）、
归约（`mean`/`sum`/`softmax`）、`matmul`/`bmm`、`transpose`/`permute`/`reshape`/`view`、
以及 `nn.Linear`/`nn.Conv2d`/`nn.RMSNorm`。

其余一律抛 `Unsupported`。这是故意的：trace 是一份**契约**，悄悄回退只会得到一个错的图。

### 手动建图

trace 挡路时就手写：

```python
low = Lowering("/tmp/out")
x = low.input((1024, 2048))
y = low.linear(x, some_linear_module)         # 直接出 FC（自动处理 4 输入）
z = low.rmsnorm_seq(y, weight)                # 完整 RMSNorm 序列
out = low.op(18, [z, x, low.act0()])          # 裸 MUL
low.output(out)
```

| 方法 | 生成 |
|---|---|
| `linear(x, mod)` | FC（含 4 输入形式） |
| `rmsnorm(x, mod)` / `rmsnorm_seq(x, w)` | 完整 RMSNorm 序列 |
| `silu(x)` | sigmoid × x |
| `conv2d(x, w, b, pad, stride)` | CONV_2D（标量类型已处理） |
| `transpose(x, d0, d1)` | TRANSPOSE（3 维自动过 4 维桥） |
| `reshape(x, shape)` | RESHAPE + INT32 shape 常量 |
| `matmul(a, b)` | 按操作数类型选 FC 或 BATCH_MATMUL |
| `softmax5(x, axis)` | 手展开的 softmax |
| `scalar(v)` / `scalar0(v)` | INT32，nd=1 / nd=0 |
| `const(arr)` / `const_i32(arr)` | fp16 / int32 常量文件 |

---

## 五、移植自己的模型

现实路线是增量式的。别想第一天就把整个网络降下来。

```
第 1 步  拿出最重的那块（通常是 decoder），跑 lowering.py，看哪里抛 Unsupported
第 2 步  把这些算子改写成能表达的写法，先在 PC 上用 torch 验证数学等价
第 3 步  建图、上机跑单块、和 torch 对拍（误差应 < 1%），不对就停下来先修
第 4 步  再加下一块，用 chain 串联
第 5 步  最后迁 vocoder（卷积是这颗芯片的强项）
```

### 从 NNAPI 迁过来要特别注意

| | NNAPI | KAtorch |
|---|---|---|
| 抽象层 | 高——切图、回退、融合都是别人决定 | 低——你自己决定一切 |
| 动态 shape | 部分支持 | **完全不支持** |
| 子图划分 | 自动 | **手动，一张图一个 blob** |
| 算子融合 | 运行时决定 | 编译期决定，你控制不了 |
| 精度 | 自动混合 | **你自己选 fp16 / INT16** |
| 错误信息 | 一般 | **一个数字，别的没有** |

**最大的差异：NNAPI 会把不支持的算子悄悄丢给 CPU，MDLA 不会。**
一个算子不支持，整张图编不出来。所以移植的真正工作，是找出你模型里 MDLA 表达不了的那些
算子，然后手工改写。

### 移植 TTS 模型

语音模型撞的墙和扩散模型完全不同。

| | DiT | TTS（VITS / FastSpeech 类） |
|---|---|---|
| 输入 | latent `[1,16,64,64]` | 音素 `[1,T]` |
| 结构 | 28 层 transformer | conv + LSTM/transformer + conv |
| 输出 | latent | 波形 `[1, T×hop]` |
| 关键算子 | FC、BATCH_MATMUL | **CONV_1D**、转置卷积、LSTM |

**① 没有 CONV_1D。** 只有 CONV_2D，必须改写：

```python
nn.Conv1d(256, 256, kernel_size=5, padding=2)
# 改成
nn.Conv2d(256, 256, kernel_size=(1, 5), padding=(0, 2))
# 输入从 [B,C,T] reshape 成 [B,C,1,T]
```

这不是可选项，lowering 会直接拒绝一维写法。

**② 动态序列长度是死穴。** `T` 被编进 blob 里。你只能：编几档固定长度（64/128/256/512），
往最近的档位 padding；或者干脆拒绝变长输入。这件事要在动手设计之前定下来。

**③ 转置卷积在这颗芯片上从没测过。** 常见退路是拆成 stride-1 卷积 + 像素重排。

**④ LSTM 大概率不支持。** 按 transformer block 或者 Conformer 式的 conv + GLU 来规划替代。

**模型分块时用 `chain`，不要反复 `run`。** 中间张量走内存而不是文件——5 段的模型能省掉
4 次几十 MB 的写 + 读。

---

## 六、出问题了怎么查

### 编译失败（`-507` / `-516`）

**二分。** 图是纯文本，删行重跑。这就是全部的调试策略，而它之所以有效，是因为产物可读。

按出现频率排序的三个原因：

1. 算子组合映射不出来（某层 norm 写法特殊）。
2. INT16 图里用了坏掉的归约。
3. SHAPE_ONLY 算子（RESHAPE/TRANSPOSE）的输入输出 scale 不一致。

### 能跑但结果不对

**先查 FC 的激活码。** 如果声明成 type 1 而不是 type 4，算子会把第一个输入透传。
症状非常明显：输出和第一个输入逐位相同。

第二可能：norm 里的 fp16 累加溢出。表现为某一层输出全是 0 或 NaN，而它前面的层看起来正常。

### 比预期慢

**先测带宽再优化任何东西。** 搭一张 32 个 MUL 串联、`[1024,2048]` 的图计时。如果拿到
8–9 GB/s，你就在 MDLA 的稳态上限上，图是带宽受限的——融合和量化都救不了
（见 [performance.md](performance.md)）。

### 设备重启了

```sh
su -c "dmesg -T | grep -iE 'oom|mdla|apusys' | tail -20"
```

看到 `system_is_deadloomld_on_oomory` 就是内存耗尽，表现为 kernel panic。常见原因是同时跑了两套
模型——比如 ComfyUI 后端没停，又在外面起了个 profiling 脚本自己加载一份权重。
要么先停后端，要么把计时逻辑写进**已经持有模型的那个进程**里。

---

## 七、坑清单

1. **只有 fp16。** fp32 图 + `relax 0` 会以 `-516` 编译失败。
2. **不能有 0 维 fp16 标量。** 用 Python float 生成 nd=1 的常量。
3. **`S` 必须是 2 的幂。** 16/64/256 可以，36/100/144 不行。
4. **CONV 的 pad/stride 必须是 type 1 且 nd=0。**
5. **FC 的激活码必须是 type 4**，否则算子静默透传。
6. **`LOGISTIC` 和 `SOFTMAX` 的输出 scale 必须是 `1/32768`。**
7. **SHAPE_ONLY 算子要求输入输出共享 scale。**
8. **命中缓存时无法导出 blob**——`getCompiledNetworkSize` 返回 `-7`。必须冷编译再导出。
9. **同一进程里重复编译最终会 `-514 RESOURCE_EXHAUSTED`。** 重启宿主。
10. **不要用含 `main.py --cpu` 字样的 adb shell 命令行去 `pkill -f "main.py --cpu"`**——
    那条命令行本身就匹配，你会把自己杀掉。用设备端脚本。
