import ast, os, sys

root = r"F:\KAtorch\katorch_nodes\katorch_npu"
ok = True
for f in sorted(os.listdir(root)):
    if not f.endswith(".py"):
        continue
    p = os.path.join(root, f)
    try:
        src = open(p, encoding="utf-8").read()
        ast.parse(src)
        print("  %-16s OK  (%d 行)" % (f, src.count("\n") + 1))
    except SyntaxError as e:
        ok = False
        print("  %-16s 失败: %s" % (f, e))

print()
print("全部通过" if ok else "有失败项")
sys.exit(0 if ok else 1)
