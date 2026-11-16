"""解析 NPT_DEBUG 日志，提取「PyTorch 节点 -> Neuron 算子」的映射表。

日志格式（lowering.py 的 NPT_DEBUG 输出）：

       op 14 ins=[1] shapes=[(1, 1, 2048)]          ← 缩进 6 空格：一个 Neuron 算子
   silu                   call_function  -> id=10 (1, 1, 2048)   ← 缩进 3 空格：一个 fx 节点
    matmul 3D@2D: a=[...] b=[...]                   ← 缩进 4 空格：说明行，忽略

用法：
    python parse_npt_debug.py <log> [out.json]
"""
import json
import re
import sys
from collections import Counter, defaultdict

RE_NODE = re.compile(r"^   (\S+)\s+(\S+)\s+-> id=(\d+)\s*(.*)$")
RE_OP = re.compile(r"^\s+op (\d+) ins=\[([^\]]*)\]")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    path = sys.argv[1]
    lines = open(path, encoding="utf-8", errors="replace").read().split("\n")

    nodes = []
    cur = None
    block = 0
    for ln in lines:
        if ln.startswith("==="):
            block += 1
            cur = None
            continue
        m = RE_NODE.match(ln)
        if m:
            cur = {
                "block": block,
                "name": m.group(1),
                "type": m.group(2),
                "out_id": int(m.group(3)),
                "shape": m.group(4).strip(),
                "ops": [],
            }
            nodes.append(cur)
            continue
        m2 = RE_OP.match(ln)
        if m2 and cur is not None:
            cur["ops"].append(int(m2.group(1)))

    total_ops = sum(len(n["ops"]) for n in nodes)
    print("fx 节点 %d 个，生成 Neuron 算子 %d 个" % (len(nodes), total_ops))
    print("平均每节点 %.2f 个算子" % (total_ops / max(1, len(nodes))))

    # 按 fx 节点类型统计
    by_type = defaultdict(lambda: {"n": 0, "ops": 0, "hist": Counter()})
    for nd in nodes:
        k = nd["type"]
        by_type[k]["n"] += 1
        by_type[k]["ops"] += len(nd["ops"])
        by_type[k]["hist"][len(nd["ops"])] += 1

    print("\n=== 按 fx 节点类型 ===")
    print("%-16s %6s %8s %8s  算子数分布" % ("类型", "个数", "总算子", "平均"))
    for k, v in sorted(by_type.items(), key=lambda x: -x[1]["ops"]):
        dist = " ".join("%d×%d" % (n, c) for n, c in sorted(v["hist"].items()))
        print("%-16s %6d %8d %8.2f  %s" % (k, v["n"], v["ops"], v["ops"] / v["n"], dist[:46]))

    # 最重的映射（生成算子最多的 fx 节点）
    print("\n=== 生成算子最多的 15 个 fx 节点 ===")
    for nd in sorted(nodes, key=lambda x: -len(x["ops"]))[:15]:
        ops = " ".join(str(c) for c in nd["ops"][:14])
        print("  %-22s %-14s %2d 个: %s%s"
              % (nd["name"][:22], nd["type"], len(nd["ops"]), ops,
                 " ..." if len(nd["ops"]) > 14 else ""))

    # 零算子的节点（只产生常量或别名）
    zero = [n for n in nodes if not n["ops"]]
    print("\n=== 不产生算子的 fx 节点：%d 个 ===" % len(zero))
    zc = Counter(n["type"] for n in zero)
    for k, v in zc.most_common(8):
        print("  %-16s %d" % (k, v))

    if len(sys.argv) > 2:
        json.dump(nodes, open(sys.argv[2], "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        print("\n已写出 %s" % sys.argv[2])
    return 0


if __name__ == "__main__":
    sys.exit(main())
