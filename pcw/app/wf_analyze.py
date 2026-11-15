import json, io, os, collections

P = r"D:\comfyui\ComfyUI-master\user\default\workflows\工作流-Anima-带画师串-8步.json"

raw = open(P, "rb").read()
print("文件大小:", len(raw), "字节")
for enc in ("utf-8-sig", "utf-8", "gbk"):
    try:
        txt = raw.decode(enc)
        print("编码:", enc)
        break
    except Exception:
        continue
else:
    print("无法解码")
    raise SystemExit

d = json.loads(txt)
print("顶层键:", list(d.keys())[:10])

nodes = d.get("nodes") or []
print("节点数:", len(nodes))

cnt = collections.Counter(n.get("type") for n in nodes)
print()
print("=== 节点类型统计 ===")
for t, c in cnt.most_common(40):
    print("  %-32s %d" % (t, c))
