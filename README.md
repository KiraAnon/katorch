# KAtorch

> **中文** · [English](README.en.md)

**在 MediaTek MDLA NPU 上跑 PyTorch 模型 —— 不依赖 NNAPI，不依赖厂商 SDK。**

KAtorch 是一套针对 MediaTek Dimensity 芯片内置 NPU 的编译与运行栈，分两部分：

* **静态图编译器（Python）**：用 `torch.fx` 抓图，降级成一套很小的纯文本图格式
  （`OPD` / `OP` / `IO`），再由设备上的 Neuron adapter 编译成逐层的 `.dla` blob。
* **宿主进程（C，约 550 行，只用 `dlopen`）**：加载图 → 编译 → 执行，在 stdin/stdout 上讲
  一套逐行 JSON 协议。

运行时唯一的外部依赖，是设备 `/vendor/lib64` 里本来就有的 `libneuron_adapter`。
MediaTek 的东西既没有被链接，也没有被拷贝，更没有被打补丁。

```text
  PyTorch 模块               图文本                   .dla blob
  ────────────  lowering.py  ──────────   host.c +    ──────────
  nn.Module  ──────────────►  OPD/OP/IO  ──────────►  NeuronModel
                                                          │
                                 .bin 张量  ◄──────────────┘
```

---

## 为什么要自己造

NNAPI 是访问 Android NPU 的正规途径。但在 MediaTek 上，它给你的是：自动切图、不支持的算子
静默回退 CPU、有时候能用的动态 shape、以及数值出错时完全看不出是哪一层。

KAtorch 站在反面：图你自己搭、层你自己切、精度你自己选，**并且没有回退**。一个算子不支持，
编译就带着错误码失败。代价是能力面很小——实测可用的算子只有 20 个（另有 4 个已定义未测）；回报是这套栈真的可以调，
因为中间产物是一个纯文本文件，能读、能 diff、能二分。

它不是通用 NPU 运行时。它只是把一个 28 层 DiT 和一个 Wan-VAE 解码器在手机上跑成 fp16 所
需要的东西。之所以公开，是因为这颗芯片的这些硬指标几乎没有公开记录。

---

## 现状

**只在一块设备上做过端到端验证**，其余一切按未验证对待。

| | |
|---|---|
| SoC | MediaTek MT6897 / Dimensity 8300 |
| NPU | MDLA 5.0（APUSys） |
| Neuron | 7.x |
| 系统 | Android 15 / Termux / Python 3.14 / torch 2.15（CPU only） |
| 权限 | **必须 root**（要访问 `/dev/mtk_mdla*`、`/dev/apusys*`） |

**能跑通的**：fp16 建图 → 编译 → 执行；20 个实测算子（+4 已定义）；多 blob 的 `bload`/`map`；编译缓存；
blob 的 `export`/`restore`；内存内 `chain` 串联；28 层 DiT（14 块）与 4 段 VAE，都和
PyTorch 参考实现逐层对过数值。

**不支持的**：动态 shape（图在编译期定死）、NPU 上的 fp32、LSTM/GRU、gather/scatter，
以及任何这套算子表达式写不出来的东西。INT8/INT16 量化能打通，但在**这颗芯片上净收益为负**，
见[docs/zh/performance.md](docs/zh/performance.md#4-量化)。

---

## 实测数字（MT6897）

| 项目 | 数值 |
|---|---|
| 实测可用算子 | **20 个**（+4 已定义未测） |
| CONV_2D 吞吐 | 700 – 1200 GFLOP/s |
| FC（1024×2048 → 2048） | 8.6 ms（约 1 TFLOP/s） |
| MDLA 稳态访存带宽 | **8.7 GB/s** |
| 单块 blob 编译，冷 / 缓存命中 | 9.3 s / 0.25 s |
| DiT 单步去噪（14 块） | 11.0 s |
| DiT 8 步 | 90 s |
| VAE 解码 512×512，NPU / CPU | **1.3 s / 37 s** |
| 端到端 512×512 出图（稳定态） | **96 s** |
| 单层对 PyTorch 的误差 | 0.14 % |
| Wan VAE 四段整体误差 | 0.078 % |

DiT 的单步已经跑到**带宽下限的 78%**，不是算力受限：每步搬运 74 GB，按 8.7 GB/s 算是 8.5 s，
实测 10.8 s。这也正是量化、算子融合、`map`/`unmap` 调优都被测过并放弃的原因——
见[docs/zh/performance.md](docs/zh/performance.md)。

---

## 五分钟跑通

前提：设备已 root，Termux 里装了 `clang`，且 `/vendor/lib64` 下有
`libneuron_adapter_mgvi.so`。

```sh
# 1. 编宿主
pkg install clang
git clone https://github.com/KiraAnon/katorch.git && cd katorch
clang -O2 -o host_new 编译器/host.c -ldl

# 2. 跑最小的图（一个 ADD，不需要任何模型权重）
python examples/minimal_add/run.py --host ./host_new
# → {"ok":true,"ops":1,"slot":0}
# → {"ok":true,"rc":0}
# → max|out - in| = 0.0        （a + 0 == a，逐位精确）
```

再往下读：

| 文档 | 内容 |
|---|---|
| [docs/zh/guide.md](docs/zh/guide.md) | 图文本格式、host 全部命令协议、移植指南、排障 |
| [docs/zh/operators.md](docs/zh/operators.md) | 哪些算子真能用、哪些会静默出错 |
| [docs/zh/performance.md](docs/zh/performance.md) | 性能实测，以及"为什么快不起来" |

---

## 目录

**部分目录名是中文的**，因为它们在设备上（`~/npt/编译器` 等）就是这个路径，改名会让
电脑和手机两边对不上。

```
编译器/        lowering.py（torch.fx → 图文本）、shape_infer.py、host.c
KAcustom/      公共层：Lin / Norm / BalancedNorm / Attention / MLP / AdaLN
KAllm/         Qwen3 文本编码器 + 6 层 cross-attn adapter
KAdit/         Anima DiT：block、RoPE、patch embed、final layer、28 层分块
KAvae/         Wan2.1 VAE decoder（4 段）
katorch_nodes/ ComfyUI 节点包（13 个节点）
pcw/           DiT / VAE 的建图脚本
tools/         基准与分析脚本（带宽、算子成本、图统计）
测试/         算子能力表生成、数值对拍
校准/          残差坐标校准表
docs/          格式、协议、算子、性能文档
examples/      最小可运行样例
```

**四个模型包不能直接从 checkout 里 import**：它们是被部署到设备 `~/npt/` 下的扁平模块，
彼此通过 `KA_HOME` / `KA_WORK` 找对方。

模型权重、编译好的 `.dla`、厂商二进制、以及安卓壳都**不在**这个仓库里，见 [NOTICE](NOTICE)。

---

## 三条塑造了整个设计的硬约束

1. **静态图。** 每个 shape 在编译期冻死。换分辨率 = 重新编译，不是换参数。
2. **只有 fp16。** MDLA 上没有 fp32 通路。任何超过 65504 的东西都是一个**不会报错**的正确性
   缺陷 —— `BalancedNorm` 之所以存在，就是因为 DiT 的残差流峰值到 65 万。
3. **全局 1.35 GB 的 IOVA 配额。** 常量张量必须在编译产物的整个生命周期里常驻。一层 DiT 带着
   579 MB 常量，这就是模型被切成 14 块、以及合并块会直接 `-314` 失败的原因。

---

## 环境的其他约束（验证过）

* **token 数 S 必须是 2 的幂**：16 / 64 / 256 可以，36 / 100 / 144 编译失败。
* **不能有 0 维 fp16 标量**，要用 Python float 生成 nd=1 的常量。
* **CONV 的 pad/stride 必须是 INT32 标量**（type=1, nd=0），写成 type=4 会失败。
* **FC 必须 4 个输入**，且 activation_code 必须是 type 4；写成 type 1 会**静默直通**。

完整清单在 [docs/zh/guide.md](docs/zh/guide.md#七坑清单)。

---

## 环境变量

| 变量 | 默认 | 作用 |
|---|---|---|
| `KA_HOME` | `/data/data/com.termux/files/home` | Termux 家目录 |
| `KA_WORK` | `$KA_HOME/npt` | 工作目录（图、缓存、中间张量） |
| `KA_MODELS` | `$KA_HOME/ComfyUI/models` | 模型目录 |

换机器只需要改这三个。

---

## 许可证

Apache-2.0，见 [LICENSE](LICENSE) 与 [NOTICE](NOTICE)。

本仓库不分发任何模型权重、厂商二进制或厂商头文件。本项目与 MediaTek 没有隶属关系。
