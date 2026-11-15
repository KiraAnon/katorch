import json, urllib.request, time

API = "http://127.0.0.1:8188"
try:
    st = json.load(urllib.request.urlopen(API + "/queue", timeout=15))
    run = st.get("queue_running", [])
    pend = st.get("queue_pending", [])
    print("=== 队列 ===")
    print("运行中:", len(run))
    for item in run:
        try:
            pid, prompt = item[0], item[1]
            print("  prompt_id:", pid)
            for nid, n in (prompt or {}).items():
                ct = n.get("class_type")
                inputs = n.get("inputs", {})
                brief = {k: v for k, v in inputs.items()
                         if isinstance(v, (int, float, str)) and k in
                         ("prompt", "seed", "steps", "cfg", "unet_name", "clip_name", "vae_name", "width", "height")}
                print("    [%s] %s %s" % (nid, ct, json.dumps(brief, ensure_ascii=False)[:150]))
        except Exception as e:
            print("    解析失败:", e)
    print("排队:", len(pend))
    for item in pend:
        try:
            print("  ", item[0])
        except Exception:
            pass

    h = json.load(urllib.request.urlopen(API + "/history", timeout=20))
    print()
    print("=== 已完成 ===", len(h))
    for pid, rec in list(h.items())[-3:]:
        msgs = rec.get("status", {}).get("messages", [])
        for m in msgs:
            if m and m[0] == "execution_success":
                pass
        for nid, o in rec.get("outputs", {}).items():
            for im in o.get("images", []):
                print("  ", im.get("filename"))
except Exception as e:
    print("查询失败:", e)
