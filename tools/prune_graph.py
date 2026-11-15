"""剔除图里没有被任何算子引用的常量，并重新连续编号。

为什么需要「重新编号」
  host.c 的 build 逻辑是：M_addOp 按 OPD **出现顺序**分配索引，而 M_setVal 用的是
  **文件里写的 id**。两者必须一致，也就是说 operand id 必须 0..N-1 连续且按序出现。
  直接删掉中间的若干行会让 id 出现空洞，build 会以 rc=-304 (M_setVal 失败) 挂掉
  —— 这个坑实测踩过。

  所以本脚本做两件事：先剔除死常量，再把所有 id 压成连续序列，OP/IO 行同步改写。

死的常量从哪来
  FX 图里 get_attr 会立刻物化一个常量 operand，而 matmul 的 3D@2D 分支又会从
  const_data 取 numpy 重新 self.const() 一份（因为 FC 要 [out,in]，原常量是 [in,out]）。
  结果每遇到一次这种 matmul 就留下两个没人引用的常量。实测占常量的 52%。

用法：
    python prune_graph.py <图目录> [输出目录]
  不给输出目录时就地改写（会先写 .bak）。

校验：
    prune_graph.py 只做结构改写，不改数值。改完务必编译 + 跑一次对拍。
"""
import os
import shutil
import sys


def parse(path):
    opds, ops, ios, tail = [], [], [], []
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if line.startswith("OPD "):
            opds.append(line)
        elif line.startswith("OP "):
            ops.append(line)
        elif line.startswith("IO "):
            ios.append(line)
        else:
            tail.append(line)
    return opds, ops, ios, tail


def op_refs(line):
    t = line.split()
    ni = int(t[2])
    ins = [int(x) for x in t[3:3 + ni]]
    no = int(t[3 + ni])
    outs = [int(x) for x in t[4 + ni:4 + ni + no]]
    return t[1], ins, outs


def const_files(path, line):
    """OPD 行里的 w 常量文件路径"""
    t = line.split()
    nd = int(t[3])
    rest = t[6 + nd:]
    if rest and rest[0] == "w":
        return " ".join(rest[1:])
    return None


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    srcdir = sys.argv[1]
    dstdir = sys.argv[2] if len(sys.argv) > 2 else srcdir
    gpath = os.path.join(srcdir, "g.txt")
    if not os.path.exists(gpath):
        print("找不到", gpath)
        return 2

    opds, ops, ios, tail = parse(gpath)
    used = set()
    for l in ops:
        _, ins, outs = op_refs(l)
        used.update(ins)
        used.update(outs)
    for l in ios:
        t = l.split()
        ni = int(t[1])
        used.update(int(x) for x in t[2:2 + ni])
        no = int(t[2 + ni])
        used.update(int(x) for x in t[3 + ni:3 + ni + no])

    # OPD 的 id 必须唯一且按顺序；这里保留 id 与行的对应
    kept, dropped = [], []
    for l in opds:
        oid = int(l.split()[1])
        (kept if oid in used else dropped).append((oid, l))

    if not dropped:
        print("没有死常量，无需处理")
        return 0

    # 重新编号：按原出现顺序压成 0..N-1
    remap = {}
    for new_id, (old_id, _) in enumerate(kept):
        remap[old_id] = new_id

    def rewrite_opd(l):
        t = l.split(" ", 2)
        oid = int(t[1])
        return "OPD %d %s" % (remap[oid], t[2])

    def rewrite_op(l):
        code, ins, outs = op_refs(l)
        ni, no = len(ins), len(outs)
        parts = ["OP", code, str(ni)]
        parts += [str(remap[x]) for x in ins]
        parts.append(str(no))
        parts += [str(remap[x]) for x in outs]
        return " ".join(parts)

    def rewrite_io(l):
        t = l.split()
        ni = int(t[1])
        ins = [int(x) for x in t[2:2 + ni]]
        no = int(t[2 + ni])
        outs = [int(x) for x in t[3 + ni:3 + ni + no]]
        parts = ["IO", str(len(ins))] + [str(remap[x]) for x in ins]
        parts.append(str(len(outs)))
        parts += [str(remap[x]) for x in outs]
        return " ".join(parts)

    new_lines = [rewrite_opd(l) for _, l in kept]
    new_lines += [rewrite_io(l) for l in ios]
    new_lines += [rewrite_op(l) for l in ops]
    new_lines += tail

    # 只复制仍然被引用的常量文件 —— 否则目录大小不会变
    keep_files = set()
    for _, l in kept:
        f = const_files(srcdir, l)
        if f:
            keep_files.add(os.path.basename(f))

    if dstdir != srcdir:
        os.makedirs(dstdir, exist_ok=True)
        n_copied = 0
        for item in os.listdir(srcdir):
            if item == "g.txt":
                continue
            if item.endswith(".bin") and item not in keep_files:
                continue            # 死常量，不复制
            s, d = os.path.join(srcdir, item), os.path.join(dstdir, item)
            if os.path.isfile(s):
                shutil.copy2(s, d)
                n_copied += 1
        print("复制文件 %d 个（跳过死常量 %d 个）" % (n_copied, len(dropped)))
    else:
        shutil.copy2(gpath, gpath + ".bak")

    open(os.path.join(dstdir, "g.txt"), "w", encoding="utf-8").write("\n".join(new_lines) + "\n")

    # 删掉不再被引用的常量文件（只在输出目录就是源目录时才真删）
    dead = set()
    for _, l in dropped:
        f = const_files(srcdir, l)
        if f:
            dead.add(os.path.basename(f))
    if dstdir == srcdir:
        removed = 0
        for f in dead:
            p = os.path.join(srcdir, f)
            if os.path.exists(p):
                os.remove(p)
                removed += 1
        print("删除常量文件 %d 个" % removed)

    print("operand: %d -> %d（剔除 %d）" % (len(opds), len(kept), len(dropped)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
