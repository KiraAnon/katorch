import json

P = r"D:\comfyui\ComfyUI-master\user\default\workflows\工作流-Anima-带画师串-8步.json"
d = json.loads(open(P, "rb").read().decode("utf-8-sig"))

out = []
for n in d.get("nodes", []):
    t = n.get("type")
    title = n.get("title") or ""
    wv = n.get("widgets_values")
    inputs = [(i.get("name"), i.get("type")) for i in (n.get("inputs") or [])]
    outs = [(o.get("name"), o.get("type")) for o in (n.get("outputs") or [])]
    out.append("=" * 70)
    out.append("id=%s  type=%s  title=%s" % (n.get("id"), t, title))
    if inputs:
        out.append("  输入: %s" % inputs)
    if outs:
        out.append("  输出: %s" % outs)
    if wv is not None:
        s = json.dumps(wv, ensure_ascii=False)
        if len(s) > 1500:
            s = s[:1500] + " …(截断)"
        out.append("  参数: %s" % s)

txt = "\n".join(out)
open(r"F:\KAtorch\pcw\app\wf_dump.txt", "w", encoding="utf-8").write(txt)
print("已导出 %d 字符" % len(txt))
print(txt[:6000])
