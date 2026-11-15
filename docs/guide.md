# KAtorch: format, protocol, and porting

This is the reference you need after the [quick start](../README.en.md#quick-start).
It covers the graph text format, the complete host command protocol, driving
`lowering.py` from your own PyTorch module, and how to diagnose the failures that
this hardware produces.

中文版：[zh/guide.md](zh/guide.md)

---

## 1. What you are working with

Four layers, and you choose how deep to go:

```
your PyTorch model
        │  lowering.py          trace → graph text
        ▼
   OPD / OP / IO  text
        │  host.c              build → compile → run
        ▼
   NeuronAdapter (vendor, dlopen'd from /vendor/lib64)
        │
        ▼
   MDLA 5.0
```

For a model that is not a transformer you will mostly live in layer three
(`lowering.py` + `host.c`). The four model packages in this repository
(`KAcustom` / `KAllm` / `KAdit` / `KAvae`) are worked examples, not a framework.

### Requirements on the device

```sh
# is there an NPU at all?
ls -l /dev/mtk_mdla* /dev/apusys* 2>/dev/null
find /vendor -name "libneuron*" 2>/dev/null
```

**Root is required.** An ordinary Termux session cannot open the NPU device nodes.

### Building the host

```sh
pkg install clang
clang -O2 -o host_new host.c -ldl
```

`host.c` uses `dlopen` for everything vendor-related, so there is nothing to link against.
If your device keeps its Neuron libraries somewhere other than
`/data/data/com.termux/files/usr/lib/`, change `LIBDIR` at the top of `host.c`.

---

## 2. The graph format

Plain text, one directive per line, no continuation lines.

```
OPD <id> <type> <nd> <dim1> ... <dim_nd> <scale> <zp> [<tail>]
OP  <opcode> <n_in> <in_id...> <n_out> <out_id...>
IO  <n_in> <in_id...> <n_out> <out_id...>
END
```

| directive | meaning |
|---|---|
| `OPD` | declare an operand: id, type, rank, dims, quant scale, zero point, and a tail |
| `OP` | declare one operation: opcode, its input operand ids, its output operand ids |
| `IO` | the graph's external interface — which operands are inputs, which are outputs |
| `END` | end of file |

### 2.1 A complete graph

```text
OPD 0 8 2 256 512 0 0 i
OPD 1 8 2 256 512 0 0 w b.bin
OPD 2 8 2 256 512 0 0 o
OPD 3 4 1 1 0 0 a 0
OP 0 3 0 1 3 1 2
IO 1 0 1 2
END
```

Read it as: operand 0 is an fp16 `[256,512]` graph input; operand 1 is an fp16 `[256,512]`
constant read from `b.bin`; operand 2 is the output; operand 3 is an INT32 immediate `0`.
Then `ADD` takes operands `0 1 3` and produces operand `2`.

That graph computes `a + 0`. It is the smallest thing that exercises the whole path, which
is why it is the first example in the repository.

### 2.2 Types

| value | type | use |
|---|---|---|
| `0` | FLOAT32 scalar | rare |
| `1` | **INT32 scalar** | CONV pad / stride / activation, `nd=0` |
| `3` | TENSOR_FLOAT32 | demoted to fp16 by `relax 1` |
| `4` | **TENSOR_INT32** | activation codes, shape arrays, axis indices |
| `8` | **TENSOR_FLOAT16** | the default working type |

### 2.3 Tails

| tail | meaning |
|---|---|
| `i` | graph input |
| `o` | graph output (also valid for intermediate results) |
| `w <path>` | constant weights, read from a file |
| `a <int>` | immediate scalar (pair with type 1 or 4) |

### 2.4 Operator-specific argument rules

Three operators are picky enough to deserve their own section. Get any of these wrong and
you will lose an afternoon.

**FC (9) takes exactly four inputs.**

```
OP 9 4 <input> <weight> <bias> <activation_code> 1 <output>
```

`activation_code` must be **type 4**. If you declare it as type 1, the operator does not
fail — it passes its first input straight through. Your graph runs, returns rc=0, and every
number is wrong.

**CONV (3) takes seven inputs, and the parameters are 0-dimension INT32 scalars.**

```
OP 3 7 <in> <w> <b> <pad> <stride> <act> <dilation> 1 <out>
```

Use `scalar0()` for pad / stride / act / dilation — that emits type 1 with `nd=0`. A type-4
tensor here will not be accepted.

**RESHAPE (22) takes its target shape as an INT32 constant file.**

```
OP 22 2 <in> <shape_const> 1 <out>
```

```python
np.array([1, 1024, 2048], np.int32).tofile("s.bin")
# OPD <id> 4 1 3 0 0 w s.bin
```

**MEAN / REDUCE_SUM / REDUCE_MAX take three inputs**, not two — data, axis constant,
activation code. See [operators.md §4](operators.md#4-the-three-input-reduction-trap).

### 2.5 Interface

```
IO <n_in> <id...> <n_out> <id...>
```

Conventionally placed after all `OPD` lines, but order does not matter to the parser.

---

## 3. Host protocol

`host_new` reads one JSON object per line on stdin and writes one line per command on
stdout.

### 3.0 The banner

On startup it prints exactly one line before accepting anything:

```json
{"ok":true,"devices":["mtk-gpu","mtk-dsp","mtk-mdla"],"ready":true}
```

**Read and discard it.** If you skip it, every subsequent response is off by one and you
will chase a bug that is not there. The banner also tells you which accelerators this
particular device exposes.

### 3.1 Commands

| command | arguments | effect | returns |
|---|---|---|---|
| `ping` | — | liveness | `{"ok":true,"pong":1}` |
| `build` | `path` | load a graph text file | `{"ok":true,"ops":N,"slot":N}` |
| `compile` | — | compile into an executable network | `{"ok":true,"rc":0}` |
| `run` | `spec` | execute one graph | `{"ok":true,"rc":0}` |
| `chain` | `spec` | execute several graphs, results passed **in memory** | `{"ok":true,"rc":0}` |
| `create` | — | create an empty model manually | `{"ok":true,"rc":0}` |
| `export` | `path` | write the compiled blob | `{"ok":true,"bytes":N}` |
| `restore` | `path` | load a blob, skipping build + compile | — |
| `use` | `slot` | switch model slot | — |
| `pop` | — | free the current slot (reclaims IOVA) | `{"ok":true,"freed_slot":N,...}` |
| `bload` | `list` | load a list of blobs, one path per line | `{"ok":true,"n":N}` |
| `map` | `idx` | map blob `idx` for execution | `{"ok":true,"nin":N,"nout":N}` |
| `unmap` | — | unmap | — |
| `cachedir` | `path` | set the compile-cache directory | — |
| `cachekey` | `v` | set the cache key | — |
| `relax` | `0`/`1` | demote fp32 to fp16 | — |
| `nodev` | `0`/`1` | do not bind an explicit device | — |
| `poolclear` | — | clear the memory pool | — |
| `ops` | — | list registered operators | — |
| `dump` | `dir` | dump intermediate tensors | — |
| `quit` | — | exit | — |

### 3.2 The two execution paths

**Development: build + compile.**

```json
{"cmd":"build","path":"/path/to/g.txt"}
{"cmd":"compile"}
{"cmd":"run","spec":"..."}
```

Parses the graph and compiles it every time. One DiT layer takes about 1.8 s to compile;
a large graph takes minutes.

**Production: bload + map.**

```json
{"cmd":"bload","list":"/path/to/blobs.txt"}
{"cmd":"map","idx":0}
{"cmd":"run","spec":"..."}
```

Memory-maps precompiled blobs — milliseconds. **This is what you ship.**

### 3.3 The `run` spec

```
<n_in> <slot> <path> <slot> <path> ... <n_out> <path> [<path> ...]
```

`n_in` inputs, each a pair of input index and file path, then `n_out` output file paths.

### 3.4 Error codes

| code | meaning | what to do |
|---|---|---|
| `-101` | graph file could not be opened | check the path |
| `-102` | `M_create` failed | device not initialised |
| `-104` … `-106` | constant file read failed | check what `w` points at |
| `-200-rc` | `M_addOp` failed (OPD) | usually the operand declaration |
| `-300-rc` | `M_setVal` failed (constant) | check the size and type |
| `-310-rc` | immediate-set failed | check the `a` tail |
| `-400-rc` | `M_addOpn` failed (OP) | usually an unsupported operator |
| **`-314`** | **`M_addOp` returned 114** | **total constants exceeded the IOVA budget** |
| `-507` | compilation failed | unsupported operator or invalid shape |
| `-516` | **UNMAPPABLE** | the operator *combination* cannot be mapped |
| `-599` | no compiled model | run `compile` first |
| `-601` … `-603` | input file problem | check the spec paths |

Two of these are worth memorising:

* **`-314` means "your constants do not fit".** Constant tensors must stay resident for as
  long as the compiled model exists, against a device-wide budget of about 1.35 GB. This is
  not a per-model limit — it is a pool, and it is why models have to be split.
* **`-516` is the one you will fight.** It is the only signal you get for "this operator
  combination is not expressible", and it does not tell you which operation caused it.
  See §6.

---

## 4. Building graphs from PyTorch with `lowering.py`

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
x = low.input((1, 256))            # declare the input shape
low.lower_module(m, [x])           # symbolic_trace + walk
low.validate()                     # shape check before you ship it
open("/tmp/mymodel/g.txt", "w").write(low.graph_text())
```

`lower_module` does `fx.symbolic_trace(mod)`, walks the nodes converting each to NPU
operators, and marks the return value as the graph output.

### What it understands

Arithmetic (`add` / `sub` / `mul` / `div`), activations (`relu`, `silu`, `tanh`, `sigmoid`,
`exp`, `log`, `sqrt`), reductions (`mean`, `sum`, `softmax`), `matmul` / `bmm`,
`transpose` / `permute` / `reshape` / `view`, and `nn.Linear` / `nn.Conv2d` / `nn.RMSNorm`.

Anything else raises `Unsupported`. That is deliberate: the trace is a *contract*, and a
silent fallback would hide a wrong graph.

### Lower-level construction

When the trace gets in the way, build the graph by hand:

```python
low = Lowering("/tmp/out")
x = low.input((1024, 2048))
y = low.linear(x, some_linear_module)         # emits FC with the 4-input form
z = low.rmsnorm_seq(y, weight)                # full RMSNorm sequence
out = low.op(18, [z, x, low.act0()])          # raw MUL
low.output(out)
```

| method | emits |
|---|---|
| `linear(x, mod)` | FC (handles the 4-input form) |
| `rmsnorm(x, mod)` / `rmsnorm_seq(x, w)` | the full RMSNorm sequence |
| `silu(x)` | sigmoid × x |
| `conv2d(x, w, b, pad, stride)` | CONV_2D with correct scalar typing |
| `transpose(x, d0, d1)` | TRANSPOSE, bridging 3-D through 4-D |
| `reshape(x, shape)` | RESHAPE with an INT32 shape constant |
| `matmul(a, b)` | FC or BATCH_MATMUL, chosen by operand kind |
| `softmax5(x, axis)` | hand-expanded softmax |
| `scalar(v)` / `scalar0(v)` | INT32 as nd=1 / nd=0 |
| `const(arr)` / `const_i32(arr)` | fp16 / int32 constant files |

---

## 5. Porting your own model

The realistic path is incremental. Do not try to lower the whole network on day one.

```
step 1  take the heaviest subgraph (usually a decoder)
        run lowering.py on it, see what raises Unsupported

step 2  rewrite those operations into expressible forms
        verify the rewrite is mathematically equivalent in torch on the PC

step 3  build the graph, run it on device, compare against torch
        expect < 1 % relative error; if not, stop and fix before continuing

step 4  add the next block, and chain them

step 5  port the vocoder last (convolutions are what this hardware is good at)
```

### If you are coming from NNAPI

| | NNAPI | KAtorch |
|---|---|---|
| abstraction | high — partition, fallback, fusion decided for you | low — you decide everything |
| dynamic shapes | partly supported | **not at all** |
| subgraph split | automatic | **manual, one blob per graph** |
| operator fusion | decided at runtime | decided at compile time, not by you |
| precision | automatic mixed | **you choose fp16 or INT16** |
| error messages | mediocre | **a numeric code, and nothing else** |

**The single biggest difference: NNAPI silently hands unsupported operators to the CPU.
MDLA does not.** One unsupported operator fails the whole graph. So the actual work of
porting is finding out which operators your model uses that this device cannot express,
and rewriting those by hand.

### If you are porting a TTS model

Speech models hit different walls than diffusion models.

| | DiT | TTS (VITS / FastSpeech class) |
|---|---|---|
| input | latent `[1,16,64,64]` | phonemes `[1,T]` |
| structure | 28 transformer layers | conv + LSTM/transformer + conv |
| output | latent | waveform `[1, T×hop]` |
| critical operators | FC, BATCH_MATMUL | **CONV_1D**, conv-transpose, LSTM |

**① There is no CONV_1D.** Only CONV_2D exists. Rewrite:

```python
nn.Conv1d(256, 256, kernel_size=5, padding=2)
# becomes
nn.Conv2d(256, 256, kernel_size=(1, 5), padding=(0, 2))
# input reshaped [B,C,T] → [B,C,1,T]
```

This is not optional — the lowering will refuse the 1-D form.

**② Dynamic sequence length is the hard stop.** `T` is baked into the blob. You can either
compile a few fixed sizes (64 / 128 / 256 / 512) and pad into the nearest bucket, or refuse
variable-length input. Decide this before you design anything else.

**③ Conv-transpose has never been measured** on this hardware. The usual fallback is to
express it as stride-1 convolutions plus a pixel shuffle.

**④ LSTM is almost certainly unsupported.** Plan on replacing it with a transformer block
or a conformal convolution + GLU stack.

**Use `chain`, not repeated `run`, when your model is split.** Intermediate tensors then
travel through memory instead of files — on a 5-segment model that removes four
write/read round trips of tens of megabytes each.

---

## 6. Diagnosing failures

### `-507` / `-516` at compile

**Bisect the graph.** It is a text file; delete lines and re-run. That is the entire
debugging strategy, and it works because the artifact is readable.

The three usual causes, in order of frequency:

1. An operator combination that cannot be mapped (a norm written in an unusual order).
2. An INT16 graph using one of the broken reductions.
3. Input and output scales disagree on a SHAPE_ONLY operator.

### It runs, but the numbers are wrong

**Check the FC activation code first.** If it was declared as type 1 instead of type 4, the
operator passes its first input through. The signature is unmistakable: the output is
bit-identical to the first input.

Second most likely: fp16 accumulation overflow inside a norm. You will see zeros or NaNs at
the output of one specific layer while everything before it looks fine.

### It is slower than expected

**Measure bandwidth before optimising anything.** Build a graph of 32 chained MULs over
`[1024,2048]` and time it. If you get 8–9 GB/s, you are at the MDLA's steady-state limit
and your graph is bandwidth-bound — no amount of fusion or quantisation will help
(see [performance.md](performance.md)).

### The device reboots

```sh
su -c "dmesg -T | grep -iE 'oom|mdla|apusys' | tail -20"
```

`system_is_deadloomld_on_oomory` means memory exhaustion, reported as a kernel panic. The
usual cause is running two model sets at once — for example leaving a ComfyUI backend up
while a profiling script loads its own copy of the weights. Either stop the backend first,
or put your instrumentation **inside** the process that already has the model loaded.

---

## 7. The pitfalls, in one place

1. **Logistics are fp16-only.** A fp32 graph with `relax 0` fails to compile with `-516`.
2. **No 0-dim fp16 scalars.** Emit a Python float as an `nd=1` constant.
3. **`S` must be a power of two.** 16/64/256 compile; 36/100/144 do not.
4. **CONV pad/stride must be type 1 with `nd=0`.**
5. **FC's activation code must be type 4**, or the operator silently passes through.
6. **`LOGISTIC` and `SOFTMAX` outputs need scale `1/32768`.**
7. **SHAPE_ONLY operators (RESHAPE, TRANSPOSE) require input and output to share a scale.**
8. **A cached compile cannot be exported** — `getCompiledNetworkSize` returns `-7`. Export
   from a cold compile.
9. **Repeated compiles in one process eventually return `-514 RESOURCE_EXHAUSTED`.**
   Restart the host.
10. **Do not `pkill -f "main.py --cpu"` from an `adb shell` command line that contains that
    same string** — you will kill your own shell. Use a device-side script.

Chinese, with the full narrative and the measured evidence behind each one:
[zh/guide.md](zh/guide.md).
