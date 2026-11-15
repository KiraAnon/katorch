# Documentation

Everything here is measured on MT6897 / MDLA 5.0 unless stated otherwise. Where a claim
is uncertain, the document says so instead of smoothing it over.

| Document | 内容 |
|---|---|
| [guide.md](guide.md) · [zh/guide.md](zh/guide.md) | Graph text format, the complete host command protocol, driving `lowering.py`, porting your own model, diagnosing `-516` |
| [operators.md](operators.md) · [zh/operators.md](zh/operators.md) | Every operator that was measured, its accuracy, and the ones that silently misbehave |
| [performance.md](performance.md) · [zh/performance.md](zh/performance.md) | End-to-end numbers, the bandwidth ceiling, and the list of optimisations that were tried and rejected |

## Reading order

**If you just want to see it work:** [examples/minimal_add](../examples/minimal_add/README.md).

**If you want to bring a model over:** read `guide.md` §2 (format), §3 (protocol), then §5
(porting). Keep `operators.md` open next to it — the three-input reduction rule and the FC
activation-code trap account for most of the lost afternoons.

**If you want to know whether it is worth it:** `performance.md` §1 and §2. The short version
is that this NPU is bandwidth-limited at 8.7 GB/s, and the reference implementation already
runs at 79 % of that floor.

**If something is broken:** `guide.md` §6 (diagnosis) and §7 (the pitfall list).

## What is deliberately not documented here

* The vendor API. Get `NeuronAdapter.h` from your own device; this project only uses a small
  subset of it and does not redistribute it.
* Model architecture details beyond what the reference implementation needed. `KAdit` and
  `KAvae` are examples, not specifications.
* Anything about the Android packaging layer. That lives in its own project (see
  [NOTICE](../NOTICE)).
