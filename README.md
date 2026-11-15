# KAtorch

**Run PyTorch models on the MediaTek MDLA NPU — without NNAPI, and without a vendor SDK.**

KAtorch is a two-part toolchain for the NPU that ships inside MediaTek Dimensity SoCs:

* **A static-graph compiler (Python).** `torch.fx` is traced, lowered to a small plain-text
  graph format (`OPD` / `OP` / `IO`), and compiled by the on-device Neuron adapter into
  per-layer `.dla` blobs.
* **A host process (C, ~550 lines, `dlopen` only).** It loads a graph, compiles it, runs it,
  and speaks a line-oriented JSON protocol on stdin/stdout.

The only runtime dependency is the `libneuron_adapter` already present in `/vendor/lib64`
on the device. Nothing from MediaTek is linked, copied, or patched.

```text
  PyTorch module                graph text                 .dla blob
  ──────────────   lowering.py  ──────────   host.c +      ──────────
  nn.Module  ───────────────►   OPD/OP/IO  ──────────►    NeuronModel
                                                            │
                                   .bin tensors  ◄──────────┘
```

---

## Why this exists

NNAPI is the sanctioned way to reach an Android NPU. On MediaTek hardware it gets you a
partitioned graph, a silent CPU fallback for anything unsupported, dynamic shapes that
sometimes work, and no visibility into where a numerical error came from.

KAtorch takes the opposite position: you build the graph, you split the layers, you pick
the precision, and there is no fallback. One unsupported op fails the compile with a code.
The result is a much smaller surface — 20 measured operators, plus 4 defined but never isolated — and a stack you can actually
debug, because the intermediate artifact is a text file you can read, diff, and bisect.

It is not a general-purpose NPU runtime. It is what was needed to get a 28-layer diffusion
transformer and a Wan-VAE decoder running fp16 on a phone, and it is published because the
measured limits of that hardware are almost undocumented.

---

## Status

Validated end-to-end on one device; treat everything else as unverified.

| | |
|---|---|
| SoC | MediaTek MT6897 / Dimensity 8300 |
| NPU | MDLA 5.0 (APUSys) |
| Neuron | 7.x |
| OS / userland | Android 15, Termux, Python 3.14, torch 2.15 (CPU) |
| Access | root required (`/dev/mtk_mdla*`, `/dev/apusys*`) |

**Working:** fp16 graph build → compile → run; 20 measured operators (+4 defined); multi-blob models with
`bload`/`map`; compile cache; blob `export`/`restore`; in-memory chaining; a 28-layer DiT
(14 blobs) and a 4-segment VAE, both numerically verified against the PyTorch reference.

**Not supported:** dynamic shapes (the graph is fixed at compile time), fp32 on the NPU,
LSTM/GRU, gather/scatter, anything you cannot express with the operator set. INT8/INT16
quantization works but is **not worth it** on this silicon — see [Performance](docs/performance.md#quantization).

---

## Measured numbers (MT6897)

| Item | Value |
|---|---|
| Operators measured working | **20** (+4 defined, untested) |
| CONV_2D throughput | 700 – 1200 GFLOP/s |
| FC (1024×2048 → 2048) | 8.6 ms (~1 TFLOP/s) |
| MDLA steady-state memory bandwidth | **8.7 GB/s** |
| Blob compile, cold / cached | 9.3 s / 0.25 s |
| DiT, one denoise step (14 blobs) | 11.0 s |
| DiT, 8 steps | 90 s |
| VAE decode 512×512, NPU / CPU | **1.3 s / 37 s** |
| End-to-end 512×512 image, stable state | **96 s** |
| Single-layer numerical error vs. PyTorch | 0.14 % |
| Wan VAE, 4 segments | 0.078 % |

The DiT step is **78 % of the bandwidth-limited floor**, not compute-limited: 74 GB of
tensor traffic per step at 8.7 GB/s predicts 8.5 s against the measured 10.8 s. That is why
quantization, operator fusion and `map`/`unmap` tuning were all measured and abandoned —
see [docs/performance.md](docs/performance.md#4-quantization).

---

## Quick start

You need root, Termux with `clang`, and a device whose `/vendor/lib64` contains
`libneuron_adapter_mgvi.so`.

```sh
# 1. build the host
pkg install clang
git clone https://github.com/KiraAnon/katorch.git && cd katorch
clang -O2 -o host_new 编译器/host.c -ldl

# 2. run the smallest possible graph
python examples/minimal_add/run.py --host ./host_new
# → {"ok":true,"ops":1,"slot":0}
# → {"ok":true,"rc":0}
# → max|out - in| = 0.0        (a + 0 == a, bit exact)
```

That example needs no model weights and no compilation step beyond the two-operand graph.
From there:

* [docs/guide.md](docs/guide.md) — graph format, the full host protocol, porting, diagnosis.
* [docs/operators.md](docs/operators.md) — what actually compiles, and what silently misbehaves.
* [docs/performance.md](docs/performance.md) — measured numbers and why it cannot go faster.

---

## Repository layout

Some directory names are Chinese, because they are also the names used on the
deployed device (`~/npt/编译器` and friends). English gloss is in parentheses.

```
编译器/     (compiler)   lowering.py (torch.fx → graph text), shape_infer.py, host.c
KAcustom/    (common)     shared layers: Lin, Norm, BalancedNorm, Attention, MLP, AdaLN
KAllm/       (llm)        Qwen3 text encoder + the 6-layer cross-attention adapter
KAdit/       (dit)        Anima DiT: blocks, RoPE, patch embed, final layer, 28-layer chain
KAvae/       (vae)        Wan2.1 VAE decoder in 4 segments
katorch_nodes/             ComfyUI node pack (13 nodes) wrapping the above
pcw/                       graph-building scripts for the DiT/VAE experiments
tools/                     benchmarks and analyzers (bandwidth, operator cost, graph stats)
测试/        (tests)      operator table generation, numeric verification scripts
校准/        (calibration) residual-coordinate calibration tables
docs/                      format, protocol, operator and performance references
examples/                  minimal runnable graphs
```

The four model packages are not importable from a checkout: they are deployed onto the
device as flat modules under `~/npt/`, and locate each other through `KA_HOME` / `KA_WORK`.

Model checkpoints, compiled `.dla` blobs, vendor binaries and the Android shell are
**not** part of this repository. See [NOTICE](NOTICE).

---

## The three constraints that shape everything

1. **Static graph.** Every shape is frozen at compile time. A new resolution is a new
   compile, not a new parameter.
2. **fp16 only.** There is no fp32 path on the MDLA. Anything that overflows 65504 is a
   correctness bug you will not be told about — `BalancedNorm` exists because the DiT
   residual stream peaks at 650,000.
3. **A global 1.35 GB IOVA budget.** Constant tensors must stay resident for the lifetime
   of a compiled model. One DiT layer carries 579 MB of constants, which is why the model is
   split into 14 blobs and why merging them fails with `-314`.

---

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

No model weights, vendor binaries, or vendor headers are distributed here. KAtorch is not
affiliated with MediaTek.
