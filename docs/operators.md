# Operator support

Everything below was measured on MT6897 / MDLA 5.0 / Neuron 7.x, fp16, by building each
operator as a one-or-two-operation graph and comparing against a torch fp32 reference
(tolerance: relative error < 1 %).

"Works" means: `build` succeeds, `compile` returns rc=0, `run` returns rc=0, and the
numbers match. Anything else is listed at the bottom.

> **Read the status column literally.** A row marked *verified* has a stored measurement
> behind it. A row marked *defined, untested* is implemented in `shape_infer.py` and has
> been seen to compile in a larger graph, but was never isolated and checked numerically.

---

## 1. Operators verified working

| opcode | name | inputs | rel. error | notes |
|---|---|---|---|---|
| 0 | **ADD** | 2, same shape or broadcast | 0.05 % | scalar and broadcast both work |
| 36 | **SUB** | 2, same shape | 0.04 % | |
| 18 | **MUL** | 2, same shape or broadcast | 0.03 % | third input is an activation code |
| 9 | **FC** | input + weight + bias + act | 0.04 % | **exactly 4 inputs**; weight must be a constant |
| 28 | **TANH** | 1 | 0.04 % | |
| 14 | **LOGISTIC** (sigmoid) | 1 | 0.03 % | output scale must be `1/32768` |
| 49 | **EXP** | 1 | 0.017 % | |
| 88 | **SQRT** | 1 | 0.017 % | |
| 83 | **RSQRT** | 1 | 0.022 % | |
| 38 | **ABS** | 1 | 0.02 % | must be reached as `torch.abs()` (call_function) |
| 67 | **NEG** | 1 | 0.02 % | |
| 65 | **MAXIMUM** | 2, same shape | 0.02 % | used to build relu / abs |
| 66 | **MINIMUM** | 2, same shape | 0.02 % | |
| 31 | **MEAN** | input + axis + act | 0.029 % median | **3 inputs**; see §4 |
| 80 | **REDUCE_SUM** | input + axis + act | 0.03 % | **3 inputs** |
| 77 | **REDUCE_MAX** | input + axis + act | 0.018 % | **3 inputs** |
| 37 | **TRANSPOSE** | arbitrary | 0.02 % | 3-D needs a 4-D bridge |
| 22 | **RESHAPE** | same element count | 0.02 % | **eliminated at compile time — free** |
| 2 | **CONCAT** | 2, same shape | 0.02 % | last axis only |
| 25 | **SOFTMAX** | 1 (+ axis) | 0.05 % | expands to 5 operators internally |
| 3 | **CONV_2D** | NHWC fp16 + OHWC weight | 2e-4 | weight must be a constant |
| 1 | **AVGPOOL** | — | *defined, untested* | shape rules only |
| 17 | **MAXPOOL** | — | *defined, untested* | shape rules only |
| 45 | **CAST** | — | — | see §5, cannot cross INT16↔FP16 |
| 102 | **BATCH_MATMUL** | 4 inputs | — | `lowering` reaches it via `matmul` |

**20 measured, 4 defined but never isolated.**

The unary transcendental row is worth calling out, because an earlier write-up reported
`MEAN` at 117 % error and made the math path look broken. That number is real but it
belongs to the **INT16 quantization route**, not to fp16. In fp16 the same operators are
accurate. The large *relative* error on `MEAN` appears where the row mean is already near
zero — absolute error stays under 8e-5 (median 1.1e-5). If you never touch INT16, the
note does not apply to you.

---

## 2. Composite operators

Built by `lowering.py` out of the primitives above.

| name | composition | rel. error | note |
|---|---|---|---|
| **SILU** | `x * sigmoid(x)` | 0.06 % | 2 ops |
| **RELU** | `MAXIMUM(x, 0)` | 0.02 % | needs a 1-D zero constant |
| **GELU (tanh)** | polynomial + tanh | **24.99 %** | **unusable** — `x³` overflows fp16 |
| **GELU (sigmoid)** | `x * sigmoid(1.702x)` | — | the only GELU that survives fp16 in practice |

---

## 3. Reported as unavailable

| operator | status | reason | workaround |
|---|---|---|---|
| **DIV** (30) | ⚠️ **conflicting records** | 2026-11-14 sweep: `build` OK, `compile` → -516 UNMAPPABLE. 2026-11-15 porting notes list it as usable. No log survives from the second claim. | until re-measured, avoid it: multiply by a CPU-precomputed reciprocal |
| **LOG** (60) | ⚠️ **conflicting records** | same as above | keep on CPU, or piecewise-linear |
| **SIN** (85) | ⚠️ **conflicting records** | same as above | keep on CPU; Taylor is too coarse at fp16 |
| **SLICE** | not supported | no such operator | RESHAPE + CONCAT |
| **GATHER** | not supported | no such operator | keep on CPU |
| **FLOOR** | not supported | — | CAST truncation |
| **LOG_SOFTMAX** | not supported | — | hand-expand it |

The DIV/LOG/SIN conflict is a documentation defect, not a hardware mystery. To settle it,
run the sweep yourself — `测试/optable.py` builds every case in that table and prints
`OK` / `BAD` / `COMPILE` / `RUN` per operator. If your run says OK for all three, fix this
table and send the log.

**Treat the three as unavailable until then.** `lowering.py` will happily emit them
(`div`, `log`, `sin` all have lowering branches); the failure, if it comes, comes from
`compile`.

---

## 4. The three-input reduction trap

`MEAN`, `REDUCE_SUM` and `REDUCE_MAX` are **not** two-input operators. Give them two and
you get `-516 UNMAPPABLE` with no hint about which argument is missing.

```
OP 31 3 <input> <axis_const> <act_code> 1 <output>
```

* `axis_const` — **type 4** (TENSOR_INT32), `a -1` for the last axis or an explicit axis
* `act_code` — **type 4**, `a 0`

Complete graph, `[512,256]` → row means → `[512,1]`:

```
OPD 0 8 2 512 256 0 0 i
OPD 1 4 1 1 0 0 a -1
OPD 2 4 1 1 0 0 a 0
OPD 3 8 2 512 1 0 0 o
OP 31 3 0 1 2 1 3
IO 1 0 1 3
END
```

---

## 5. Hard rules

### 5.1 Argument typing

```c
// CONV_2D pad / stride must be an INT32 scalar written as nd=0
self._opd(oid, 1, (), "a %d" % k)     // type=1, nd=0
// type=4 (TENSOR_INT32) here compiles but is not what the adapter wants
```

FC's `activation_code` is the opposite case: it **must** be type 4. Type 1 does not error —
it makes the operator pass its first input straight through, so your graph runs, produces
wrong numbers, and reports rc=0.

### 5.2 Broadcasting

| pattern | supported | note |
|---|---|---|
| `[S,D] + [S,D]` | ✅ | same shape |
| `[S,D] + [1,D]` | ✅ | one-sided |
| `[S,D] + [S,1]` | ✅ | one-sided |
| `[S,1] * [1,D]` | ❌ | no two-way broadcast |

The broadcast side must be a graph input or a constant — never a computed value.

### 5.3 Constant types

| type | meaning | bytes |
|---|---|---|
| 0 | FLOAT32 scalar | 4 |
| 1 | INT32 scalar | 4 |
| 3 | TENSOR_FLOAT32 | 4 |
| 4 | TENSOR_INT32 | 4 |
| 8 | TENSOR_FLOAT16 | 2 |

A **0-dim fp16 scalar** (nd=0, type=8) fails to compile. Emit a Python float as an
nd=1 constant instead.

### 5.4 Sequence length

`S` must be a power of two. `16 / 64 / 256` compile; `36 / 100 / 144` return
`-516 UNMAPPABLE`. This is not a size limit — it is a layout rule.

### 5.5 fp16 accumulation overflow

```
sum(x²) over 2048 dims  >  65504  →  inf  →  rsqrt(inf) = 0  →  the whole layer is zeros
```

No error, no warning. Pre-scale K so that `K > sqrt(var * D / 65504)`, or use the
`BalancedNorm` construction in `KAcustom/`.

### 5.6 CONV_2D layout

| item | format |
|---|---|
| input | NHWC `[N,H,W,C]` |
| weight | OHWC `[O,H,W,C]` — permute torch's OIHW |
| output | NHWC |
| weight | must be a constant operand, never a computed value |

---

## 6. Throughput

| operator | shape | throughput |
|---|---|---|
| CONV_2D 96→96 k3 @1024² | 173.9 GFLOP | **701.8 GFLOP/s** |
| CONV_2D 384→384 k3 @256² | 173.9 GFLOP | **1189.8 GFLOP/s** |
| FC 2048→2048 @S=256 | — | ~117 GFLOP/s |
| BATCH_MATMUL | 0.27 GFLOP | 3.0 ms → 89 GFLOP/s |

**CONV is the MDLA's dedicated path** — 6–10× faster than expressing the same work as FC.
If your model has convolutions, that is where the hardware pays off.

---

## 7. Reproducing this table

```sh
export LD_PRELOAD=/data/data/com.termux/files/usr/lib/libpython3.14.so
python3 测试/optable.py
```

It prints one line per operator and writes the graphs it built into its work directory, so
a `COMPILE` failure can be bisected by deleting lines from a text file.


---

## 8. Quantization: what this silicon actually accepts

Measured on MT6897, one operator at a time, each in a fresh host process.

| path | weight form | compiles | speed | note |
|---|---|---|---|---|
| **FC** | INT8 per-channel, W8A16 (type 11 + QCH) | ✅ | **1.14x** | 8.6 -> 7.6 ms on 1024x2048x2048 |
| **CONV_2D** | INT8 per-channel + QCH (type 11) | ❌ `-516` | — | |
| **CONV_2D** | INT8 per-channel, no QCH (type 11) | ❌ `-516` | — | |
| **CONV_2D** | INT8 asymmetric per-tensor (type 5) | ❌ `-516` | — | |
| **CONV_2D** | INT16 per-tensor (type 7) | ✅ | **1.08x** | 20.49 -> 18.89 ms, 128x128x256->256 |

**The convolution path does not accept INT8 at all** — three different encodings, all
`compile rc=-516`. INT16 compiles but is only 8 % faster, against roughly half the
numeric precision.

This is why this project runs fp16 end to end and why the INT8 experiment was abandoned
rather than continued. On other phone NPUs (Hexagon, Da Vinci, ANE) INT8 is genuinely a
2-4x path; **on this MDLA generation it is not**, and the convolution path refuses it
outright.

Reproduce:

```sh
python tools/probe_conv_weight_types.py     # the five weight forms above
python tools/bench_conv_int16.py            # fp16 vs INT16 timing
python tools/bench_conv_int8.py             # the earlier INT8 attempt (fails to compile)
```
