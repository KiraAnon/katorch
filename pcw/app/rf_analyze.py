import json, collections, os

P = r"D:\comfyui\ComfyUI-master\user\default\workflows\RuleFlow粗生成-Anima-8步.json"
raw = open(P, "rb").read()
print("文件大小:", len(raw), "字节")

txt = None
for enc in ("utf-8-sig", "utf-8", "gbk"):
    try:
        txt = raw.decode(enc)
        print("编码:", enc)
        break
    except Exception:
        continue
if txt is None:
    raise SystemExit("无法解码")

d = json.loads(txt)
print("顶层键:", list(d.keys()))
nodes = d.get("nodes") or []
print("节点数:", len(nodes))
cnt = collections.Counter(n.get("type") for n in nodes)
print()
print("=== 节点类型 ===")
for t, c in cnt.most_common(40):
    print("  %-36s %d" % (t, c))

out = []
for n in nodes:
    out.append("=" * 70)
    out.append("id=%s  type=%s  title=%s" % (n.get("id"), n.get("type"), n.get("title") or ""))
    wv = n.get("widgets_values")
    if wv is not None:
        s = json.dumps(wv, ensure_ascii=False)
        out.append("  参数: " + (s[:900] + " …" if len(s) > 900 else s))

dump = "\n".join(out)
open(r"F:\KAtorch\pcw\app\rf_dump.txt", "w", encoding="utf-8").write(dump)
print()
print("已导出 dump: %d 字符" % len(dump))
