#!/usr/bin/env python3
"""Smallest end-to-end KAtorch example: build, compile and run `a + 0`.

The point is not the arithmetic. It is to prove, in about 60 lines, that the graph
format parses, the adapter is reachable, the compiler works, and tensors survive a
round trip through the device.

Usage:

    python run.py --host /path/to/host_new

On a machine without the NPU (no host binary), pass --emit-only to just write the
graph text and the constant file, so you can inspect or hand-check them:

    python run.py --emit-only

Requires numpy.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

# 256x512 keeps the compile fast and still exercises every code path.
SHAPE = (256, 512)


def graph_text(weight_path):
    """One ADD of a graph input, a constant, and an INT32 activation code."""
    rows, cols = SHAPE
    return (
        "OPD 0 8 2 %d %d 0 0 i\n"        # fp16 input
        "OPD 1 8 2 %d %d 0 0 w %s\n"     # fp16 constant from file
        "OPD 2 8 2 %d %d 0 0 o\n"        # fp16 output
        "OPD 3 4 1 1 0 0 a 0\n"          # int32 activation code, immediate 0
        "OP 0 3 0 1 3 1 2\n"             # ADD: in[0 1 3] -> out[2]
        "IO 1 0 1 2\n"
        "END\n"
    ) % (rows, cols, rows, cols, weight_path, rows, cols)


class Host:
    """Line-oriented JSON client for host_new."""

    def __init__(self, path):
        self.p = subprocess.Popen(
            [path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        # The banner MUST be consumed first, or every later reply is off by one.
        banner = self.p.stdout.readline().strip()
        if not banner:
            raise RuntimeError("host produced no banner")
        print("banner:", banner)

    def cmd(self, payload):
        self.p.stdin.write(payload + "\n")
        self.p.stdin.flush()
        return json.loads(self.p.stdout.readline().strip())

    def close(self):
        try:
            self.cmd('{"cmd":"quit"}')
            self.p.wait(timeout=30)
        except Exception:
            self.p.kill()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=os.environ.get("KA_HOST"),
                    help="path to the compiled host_new binary")
    ap.add_argument("--emit-only", action="store_true",
                    help="write the graph and constant, do not touch the device")
    ap.add_argument("--work", default=None,
                    help="working directory (default: a temporary one)")
    args = ap.parse_args()

    work = args.work
    if not work:
        # The graph format separates fields with spaces, so the constant's path must not
        # contain any. On device, /data/local/tmp is short and clean; elsewhere, fall back
        # to a temporary directory (fine for --emit-only inspection).
        work = ("/data/local/tmp/katorch_min" if os.path.isdir("/data/local/tmp")
                else tempfile.mkdtemp(prefix="katorch_min_"))
    os.makedirs(work, exist_ok=True)
    if " " in work:
        print("warning: work dir contains a space; the 'w' field in the graph text will "
              "not parse. Use --work with a space-free path.", file=sys.stderr)

    graph_path = os.path.join(work, "add.txt")
    weight_path = os.path.join(work, "b.bin")
    in_path = os.path.join(work, "a.bin")
    out_path = os.path.join(work, "out.bin")

    # A zero constant makes the expected result exactly the input:
    # adding zero is bit-exact in fp16, so any deviation is a real defect.
    np.zeros(SHAPE, dtype=np.float16).tofile(weight_path)
    open(graph_path, "w").write(graph_text(weight_path))

    rng = np.random.default_rng(20261115)
    x = rng.standard_normal(SHAPE).astype(np.float16)
    x.tofile(in_path)

    print("work dir :", work)
    print("graph    :", graph_path)

    if args.emit_only:
        print("\n--- graph ---")
        print(open(graph_path).read())
        return 0

    if not args.host or not os.path.exists(args.host):
        print("\nNo host binary. Use --host <path>, or --emit-only to inspect the graph.",
              file=sys.stderr)
        return 2

    h = Host(args.host)
    try:
        print("build  :", h.cmd('{"cmd":"build","path":"%s"}' % graph_path))
        print("compile:", h.cmd('{"cmd":"compile"}'))
        print("run    :", h.cmd('{"cmd":"run","spec":"1 0 %s 1 %s"}'
                                % (in_path, out_path)))
    finally:
        h.close()

    out = np.fromfile(out_path, dtype=np.float16).reshape(SHAPE)
    diff = float(np.abs(out.astype(np.float32) - x.astype(np.float32)).max())
    print("\nmax|out - in| = %g" % diff)
    if diff == 0.0:
        print("OK: bit-exact.")
        return 0
    print("FAILED: expected 0, the graph or the runtime is not doing what it says.",
          file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
