# minimal_add

The smallest graph that exercises the entire stack: `a + 0` over a `[256,512]` fp16 tensor.

Because the right operand is an exact zero constant, `a + 0` is bit-for-bit identical to
`a` in fp16. That makes the expected result unambiguous — any deviation is a defect in the
graph, the compiler, or the runtime, not a rounding question.

## Run it on the device

```sh
clang -O2 -o host_new ../../编译器/host.c -ldl
python run.py --host ./host_new
```

Expected:

```
banner: {"ok":true,"devices":["mtk-gpu","mtk-dsp","mtk-mdla"],"ready":true}
build  : {'ok': True, 'ops': 1, 'slot': 0}
compile: {'ok': True, 'rc': 0}
run    : {'ok': True, 'rc': 0}

max|out - in| = 0
OK: bit-exact.
```

## Run it without a device

```sh
python run.py --emit-only
```

Prints the graph text without touching the NPU. Useful for checking the format by hand, or
on a laptop where no adapter exists.

## What the graph says

```text
OPD 0 8 2 256 512 0 0 i          operand 0: fp16 [256,512], graph input
OPD 1 8 2 256 512 0 0 w b.bin    operand 1: fp16 [256,512], constant from b.bin
OPD 2 8 2 256 512 0 0 o          operand 2: fp16 [256,512], graph output
OPD 3 4 1 1 0 0 a 0              operand 3: int32 [1], immediate 0 (activation code)
OP 0 3 0 1 3 1 2                 ADD, three inputs (0, 1, 3), one output (2)
IO 1 0 1 2                       one input (0), one output (2)
END
```

Full format reference: [docs/guide.md](../../docs/guide.md#2-the-graph-format).
