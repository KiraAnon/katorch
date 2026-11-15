"""算子表 v3：直接用 host，打印完整输出。"""

# === KAtorch 路径配置（环境变量可覆盖）===
import os as _ka_os
_KA_HOME = _ka_os.environ.get("KA_HOME", "/data/data/com.termux/files/home")
_KA_WORK = _ka_os.environ.get("KA_WORK", _KA_HOME + "/npt")
_KA_MODELS = _ka_os.environ.get("KA_MODELS", _KA_HOME + "/ComfyUI/models")
# =========================================

import sys, os, subprocess
sys.path.insert(0, _KA_WORK)
import torch, numpy as np
import torch.nn as nn
import torch.nn.functional as F
from lowering import Lowering

D = os.environ.get("KAWORK", _KA_WORK)
OUT = D + "/optable3"
os.makedirs(OUT, exist_ok=True)
CACHE = OUT + "/nc"
os.makedirs(CACHE, exist_ok=True)
S, DIM = 16, 128

class One(nn.Module):
    def __init__(s, fn): super().__init__(); s.fn = fn
    def forward(s, x): return s.fn(x)
class Two(nn.Module):
    def __init__(s, fn): super().__init__(); s.fn = fn
    def forward(s, x, y): return s.fn(x, y)

def drive(cmds):
    """跑 host，返回每行响应。"""
    p = subprocess.Popen([D+"/host"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
    outs = []
    p.stdout.readline()          # greeting
    for c in cmds:
        p.stdin.write(c + "\n"); p.stdin.flush()
        l = p.stdout.readline()
        outs.append(l.strip() if l else "(died)")
        if not l: break
    p.stdin.write('{"cmd":"quit"}\n'); p.stdin.flush()
    try: p.wait(timeout=120)
    except: p.kill()
    return outs

def test(name, mod, ins_shapes, ref_fn, tol=1e-2):
    gdir = "%s/%s" % (OUT, name)
    os.makedirs(gdir, exist_ok=True)
    try:
        low = Lowering(gdir)
        args = [low.input(sh) for sh in ins_shapes]
        low.lower_module(mod, args)
        txt = low.graph_text()
        open(gdir+"/g.txt","w").write(txt)
        errs = low.validate()
        if errs: return "VALIDATE", str(errs[:1])[:60], 0
    except Exception as e:
        return "LOWER", str(e)[:60], 0

    torch.manual_seed(0)
    xs = [torch.randn(*sh) for sh in ins_shapes]
    for i, x in enumerate(xs):
        x.numpy().astype(np.float16).tofile(OUT+"/in%d.bin" % i)
    with torch.no_grad():
        try: ref = (ref_fn(*xs) if len(xs) > 1 else ref_fn(xs[0])).numpy()
        except Exception as e: return "REF", str(e)[:40], 0

    op = OUT + "/o_%s.bin" % name
    spec = "%d %s 1 %s" % (len(ins_shapes),
        " ".join("%d %s" % (i, OUT+"/in%d.bin" % i) for i in range(len(ins_shapes))), op)
    outs = drive([
        '{"cmd":"cachedir","path":"%s"}' % CACHE,
        '{"cmd":"cachekey","v":"t_%s"}' % name,
        '{"cmd":"build","path":"%s/g.txt"}' % gdir,
        '{"cmd":"compile"}',
        '{"cmd":"run","spec":"%s"}' % spec,
    ])
    if len(outs) < 5: return "HOSTDIED", str(outs[-1])[:50], len(low.ops)
    if '"ok":true' not in outs[2]: return "BUILD", outs[2][:60], len(low.ops)
    if '"ok":true' not in outs[3]: return "COMPILE", outs[3][:60], len(low.ops)
    if '"ok":true' not in outs[4]: return "RUN", outs[4][:60], len(low.ops)
    if not os.path.exists(op): return "NOFILE", "", len(low.ops)
    got = np.fromfile(op, dtype=np.float16).astype(np.float32)
    if got.size != ref.size: return "SIZE", "got=%d ref=%d" % (got.size, ref.size), len(low.ops)
    got = got.reshape(ref.shape)
    e = np.abs(got - ref).max(); sc = max(np.abs(ref).max(), 1e-6)
    return ("OK" if e/sc < tol else "BAD"), "rel=%.4f%%" % (100*e/sc), len(low.ops)

W = torch.randn(DIM, DIM) * 0.05
tests = [
    ("ADD",        Two(lambda x,y: x+y),                  [(S,DIM),(S,DIM)], lambda x,y: x+y),
    ("SUB",        Two(lambda x,y: x-y),                  [(S,DIM),(S,DIM)], lambda x,y: x-y),
    ("MUL",        Two(lambda x,y: x*y),                  [(S,DIM),(S,DIM)], lambda x,y: x*y),
    ("DIV",        Two(lambda x,y: x/(torch.abs(y)+0.5)), [(S,DIM),(S,DIM)], lambda x,y: x/(torch.abs(y)+0.5)),
    ("ADD_scalar", One(lambda x: x+1.0),                  [(S,DIM)], lambda x: x+1.0),
    ("MUL_scalar", One(lambda x: x*0.5),                  [(S,DIM)], lambda x: x*0.5),
    ("ADD_bcast",  Two(lambda x,y: x+y),                  [(S,DIM),(1,DIM)], lambda x,y: x+y),
    ("MUL_bcast",  Two(lambda x,y: x*y),                  [(S,DIM),(1,DIM)], lambda x,y: x*y),
    ("FC",         One(lambda x: F.linear(x, W)),         [(S,DIM)], lambda x: F.linear(x, W)),
    ("TANH",       One(lambda x: torch.tanh(x)),          [(S,DIM)], lambda x: torch.tanh(x)),
    ("SIGMOID",    One(lambda x: torch.sigmoid(x)),       [(S,DIM)], lambda x: torch.sigmoid(x)),
    ("EXP",        One(lambda x: torch.exp(x*0.1)),       [(S,DIM)], lambda x: torch.exp(x*0.1)),
    ("SQRT",       One(lambda x: torch.sqrt(x*x+1.0)),    [(S,DIM)], lambda x: torch.sqrt(x*x+1.0)),
    ("RSQRT",      One(lambda x: torch.rsqrt(x*x+1.0)),   [(S,DIM)], lambda x: torch.rsqrt(x*x+1.0)),
    ("ABS",        One(lambda x: torch.abs(x)),           [(S,DIM)], lambda x: torch.abs(x)),
    ("NEG",        One(lambda x: -x),                     [(S,DIM)], lambda x: -x),
    ("MAXIMUM",    Two(lambda x,y: torch.maximum(x,y)),   [(S,DIM),(S,DIM)], lambda x,y: torch.maximum(x,y)),
    ("MINIMUM",    Two(lambda x,y: torch.minimum(x,y)),   [(S,DIM),(S,DIM)], lambda x,y: torch.minimum(x,y)),
    ("MEAN_last",  One(lambda x: x.mean(-1, keepdim=True)),[(S,DIM)], lambda x: x.mean(-1, keepdim=True)),
    ("REDUCE_SUM", One(lambda x: x.sum(-1, keepdim=True)),[(S,DIM)], lambda x: x.sum(-1, keepdim=True)),
    ("TRANSPOSE",  One(lambda x: x.transpose(0,1)),       [(S,DIM)], lambda x: x.transpose(0,1)),
    ("RESHAPE",    One(lambda x: x.reshape(S*DIM)),       [(S,DIM)], lambda x: x.reshape(S*DIM)),
    ("CONCAT",     Two(lambda x,y: torch.cat([x,y], dim=-1)),[(S,DIM),(S,DIM)], lambda x,y: torch.cat([x,y], dim=-1)),
    ("SOFTMAX",    One(lambda x: torch.softmax(x, dim=-1)),[(S,DIM)], lambda x: torch.softmax(x, dim=-1)),
    ("LOG",        One(lambda x: torch.log(torch.abs(x)+1.0)), [(S,DIM)], lambda x: torch.log(torch.abs(x)+1.0)),
    ("SIN",        One(lambda x: torch.sin(x)),           [(S,DIM)], lambda x: torch.sin(x)),
    ("GELU_TANH",  One(lambda x: 0.5*x*(1+torch.tanh(0.7978845608*(x+0.044715*x*x*x)))), [(S,DIM)], lambda x: 0.5*x*(1+torch.tanh(0.7978845608*(x+0.044715*x*x*x)))),
    ("SILU",       One(lambda x: x*torch.sigmoid(x)),     [(S,DIM)], lambda x: x*torch.sigmoid(x)),
]
print("%-14s %-10s %-30s %s" % ("算子", "结果", "说明", "算子数"))
print("-" * 74)
res = []
for nm, mod, shp, rf in tests:
    r, note, nops = test(nm, mod, shp, rf)
    res.append((nm, r, note)); print("%-14s %-10s %-30s %d" % (nm, r, note, nops), flush=True)

print("\n=== 汇总 ===")
for st in ["OK","BAD","LOWER","VALIDATE","BUILD","COMPILE","RUN","REF","SIZE","HOSTDIED","NOFILE"]:
    g = [r for r in res if r[1] == st]
    if g: print("%-10s (%2d): %s" % (st, len(g), ", ".join(r[0] for r in g)))
