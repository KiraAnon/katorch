# Performance

All numbers below are measured on MT6897 / Dimensity 8300, MDLA 5.0, fp16, Android 15.

The short version: **this NPU is bandwidth-limited, and it is slower than you expect.**
Everything that looked like an optimisation opportunity was measured and turned out to be
noise against a 8.7 GB/s wall. If you are porting a model, read §4 before you spend a day
on quantisation.

---

## 1. End to end: 512×512, 8 steps, 96 seconds

| stage | time |
|---|---|
| DiT sampling, 8 steps | 95 s |
| VAE decode | 1.3 s |
| everything else | 0.7 s |
| **stable-state total** | **96 s** |
| cold start, first image | 107 s |

Cold start costs an extra ~14 s of one-time work: encoder load 3.1 s, text encoding 5.5 s,
NPU host load 2.75 s, plus overhead. The second image does not pay it.

> An earlier note in this project claimed "text encoding ≈ 25 s". It was wrong. Measured
> log lines say `文本编码完成 3.0s (257 chars)` and `2.5s (155 chars)`, and ComfyUI caches
> node output, so re-running the same prompt skips it entirely. Text encoding was never
> a bottleneck.

### 1.1 Where one sampling step goes

| item | ms | share |
|---|---|---|
| 14 blobs, `run_block` | 10817.9 | 95.5 % |
| write next block's input | 163.3 | 1.4 % |
| read output + rescale | 209.1 | 1.8 % |
| final layer | 41.3 | 0.4 % |
| patchify | 25.1 | 0.2 % |
| patch embed | 23.3 | 0.2 % |
| normalise + write | 15.4 | 0.1 % |
| unpatchify | 11.4 | 0.1 % |
| time-embedding transfer | 10.4 | 0.1 % |
| read patch-embed output | 4.5 | 0.04 % |
| **total** | **11328.9** | |

Two things this table settles immediately:

* `map`/`unmap` overhead is **3.3 %** of a step. Optimising it cannot pay for itself.
* `patchify` + `unpatchify` together are **0.3 %**. Do not touch them.

---

## 2. The ceiling is memory bandwidth

| measurement | result |
|---|---|
| 32 chained MUL over `[1024,2048]` | 46.4 ms → **8.7 GB/s** |
| one MUL over `[4096,4096]` | 37.2 ms → 2.7 GB/s |

Large tensors behave *worse*, not better. This is the MDLA's own access bandwidth, an order
of magnitude below the phone's LPDDR5 (50+ GB/s) — it is not a DDR limit and there is
nothing to tune about it in your graph.

Now the accounting for one DiT step:

```
per-block traffic   5303 MB  × 14 blocks  =  74 GB / step
74 GB ÷ 8.7 GB/s                          =  8.5 s    ← floor
measured                                    10.8 s    ← 79 % of the floor
```

**The implementation is already at 79 % of a hard physical limit.** That is why the answer
to "can we make it faster" is "change the strategy, not the code".

### 2.1 Per-operator cost, measured by differencing

| operator | size | time | implied |
|---|---|---|---|
| MUL | 4.2 MB | 1.39 ms | 9.1 GB/s |
| **RESHAPE** | — | **0** | eliminated at compile time |
| BATCH_MATMUL | 0.27 GFLOP | 3.0 ms | 89 GFLOP/s |
| FC (large) | 1024×2048×2048 | 8.6 ms | ~1 TFLOP/s |
| fixed per-operator overhead | — | ~0.6 ms | derived from the totals |

RESHAPE costing literally zero surprised us too — 16 RESHAPEs in a graph account for 4 ms
total. There are ~190 of them in the DiT graph (~33 % of all operators) and they are free.

---

## 3. The VAE story: 37 s → 1.3 s

The decoder was the second-largest cost after sampling, and it is a pure convolution stack —
exactly what the MDLA is built for. Splitting it into four segments moved it to the NPU:

| segment | content | ops | blob | time |
|---|---|---|---|---|
| 1 | conv2(1×1) + conv1 + middle + 3×Res | 209 | 28 MB | 0.17 s |
| | CPU upsample ×2 | | | 0.04 s |
| 2 | resample conv + 3×Res | 103 | 16 MB | 0.23 s |
| | CPU upsample ×2 | | | 0.05 s |
| 3 | resample conv + 3×Res | 98 | 5 MB | 0.34 s |
| | CPU upsample ×2 | | | 0.07 s |
| 4 | resample conv + 3×Res + head | 113 | 1 MB | 0.42 s |

**37 s → 1.3 s, a 28× speedup**, with no visible difference: against the torch CPU
reference, the maximum 8-bit channel difference is 2 levels and the fraction of pixels
differing by more than that is 0.00 %.

**The nearest-2× upsample cannot be done on the NPU.** The natural implementation is a
reshape plus a 5-D CONCAT, and MDLA's CONCAT only works along the last axis; every variant
(`hw = 8/16/24/32/48/64`) returned `rc=-516`. Moving the upsample to the CPU costs 0.04–0.07 s
of `np.repeat`, which is a rounding error against the win.

---

## 4. Quantization

**Measured verdict: INT8 is 1.14× faster on this silicon, and the surrounding work makes it
a net loss. Do not do it.**

The decisive experiment was 20 lines long:

| variant | FC 1024×2048 @ 2048×2048 |
|---|---|
| fp16 | 8.6 ms |
| W8A16 | 7.6 ms |
| **speedup** | **1.14×** |

Not 2×. Meanwhile, avoiding the parts of the INT16 path that are numerically broken costs
about 36 extra operators per block (DQ/Q island boundaries, reductions replaced by an FC
against a ones vector), which consumes the 14 % and then some:

```
W8A16 route:  17.3 s per step
fp16 route:   14.6 s per step
```

Full INT8 also collapses numerically — 22–78 % error per layer — so the only viable mixed
route is "INT16 with FP16 islands", at negative net value.

**What is genuinely broken in INT16** (and perfectly fine in fp16 — do not conflate these):

| operator | INT16 error |
|---|---|
| `MEAN` | 117 % |
| `REDUCE_MAX` | 63 % |
| `REDUCE_SUM` | **200 %, including a flipped sign** |

If you must quantise anyway: replace `sum(x)` with `FC(x, ones)`, which measured 0.0000 %
error. `max(x)` has no equivalent and has to stay inside an FP16 island.

**`CAST(45)` cannot cross INT16↔FP16** — it fails to compile. `DEQUANTIZE(6)` and
`QUANTIZE(72)` can, which is how the island boundaries are built. A full
`INT16 → DQ → MUL → Q → INT16` round trip measured 0.0184 % error.

---

## 5. Everything that was tried and rejected

Recorded so that nobody repeats it.

| idea | verdict | evidence |
|---|---|---|
| INT8 weights | ❌ | 8.6 ms → 7.6 ms, 1.14× |
| Full INT8 | ❌ | 22–78 % error per layer |
| Eliminate RESHAPEs | ❌ | already eliminated at compile time; zero cost |
| Reduce MUL/ADD count | ⚠️ | manual fusion saves at most 3.4 % (134 MB of 3935 MB) |
| Optimise `map`/`unmap` | ❌ | 3.3 % of a step |
| **Merge blocks (14 → fewer)** | ❌ | constants hit 1158 MB, budget is 1.35 GB, `build` returns `-314` |
| Reuse cross-attention K/V | ❌ | measured 0.65 % (context is only 512 tokens) |
| Operator fusion in general | ❌ | the MDLA has no fusion mechanism; the graph format has no fused/in-place marker |
| Cache text encoding | ❌ | encoding is 3 s and ComfyUI already caches nodes |

### The block-merge experiment, in detail

b00 and b01 were concatenated into one graph: operand ids renumbered, inputs redirected. The
graph itself was legal — 2050 operands, no duplicates, no dangling references, correct I/O —
and `build` still returned `-314` (`M_addOp` → 114).

```
b00 constants: 449 tensors /  579 MB
b01 constants: 449 tensors /  579 MB
merged:        898 tensors / 1158.5 MB   ← over the 1.35 GB IOVA budget
```

**The 14-way split is not an artefact of `map`/`unmap` — it is forced by the constant
budget.**

---

## 6. What is actually left

Only strategy changes, not implementation changes:

| option | estimate | cost |
|---|---|---|
| 384×384 | ~70 s | traffic drops to 56 %; recompose all 16 blobs, weaker conditioning |
| 6 steps instead of 8 | ~87 s | visible quality loss |
| **stay at 512×512 / 8 steps** | **96 s** | none — the current choice |

---

## 7. Reproducing the measurements

The scripts are in `tools/` and run on the device:

| script | what it measures |
|---|---|
| `bench_bandwidth.py` | steady-state access bandwidth |
| `bench_unary_prec.py` | fp16 accuracy of MEAN / SQRT / RSQRT / EXP |
| `bench_reshape.py`, `bench_reshape_mul.py` | whether RESHAPE really is free |
| `bench_op_overhead.py` | fixed per-operator cost |
| `bench_batmatmul.py` | BATCH_MATMUL throughput |
| `analyze_bandwidth.py`, `analyze_graph.py` | graph-level traffic accounting |
| `profile_step.py`, `profile_e2e.py` | step / end-to-end breakdown |

**Run these with the model backend stopped.** Instrumenting from a second process while a
ComfyUI backend holds the weights loaded is how this project crashed a phone into a kernel
panic (`system_is_deadloomld_on_oomory`). Put the timing inside the process that owns the
model, or stop the backend first.
