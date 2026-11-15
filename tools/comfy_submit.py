"""向 ComfyUI 提交一个 API 格式工作流，轮询直到出结果。

用法：
    python comfy_submit.py <workflow_api.json> [--url http://127.0.0.1:18188] [--timeout 900]
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request


def post_json(url, obj, timeout=30):
    data = json.dumps(obj).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def get_json(url, timeout=30):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("workflow")
    ap.add_argument("--url", default="http://127.0.0.1:18188")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--poll", type=float, default=5.0)
    args = ap.parse_args()

    with open(args.workflow, "r", encoding="utf-8") as f:
        wf = json.load(f)

    print("提交 %d 个节点到 %s" % (len(wf), args.url), flush=True)
    t0 = time.time()
    try:
        res = post_json(args.url + "/prompt", {"prompt": wf, "client_id": "katorch-test"})
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        print("!! HTTP %s 提交被拒绝：" % e.code, flush=True)
        print(body[:4000], flush=True)
        return 2

    pid = res.get("prompt_id")
    print("prompt_id = %s（队列号 %s）" % (pid, res.get("number")), flush=True)

    last = None
    while time.time() - t0 < args.timeout:
        time.sleep(args.poll)
        try:
            hist = get_json(args.url + "/history/" + str(pid))
        except Exception as e:
            print("  查询失败：%s" % e, flush=True)
            continue
        if pid in hist:
            entry = hist[pid]
            status = entry.get("status", {})
            print("完成，耗时 %.1fs，状态：%s" % (time.time() - t0, status.get("status_str")), flush=True)
            for node_id, out in (entry.get("outputs") or {}).items():
                imgs = out.get("images") or []
                for im in imgs:
                    print("  产物：%s" % im.get("filename"), flush=True)
            msgs = status.get("messages") or []
            for m in msgs:
                if m and m[0] in ("execution_error", "execution_interrupted"):
                    print("  !! %s" % json.dumps(m, ensure_ascii=False)[:2000], flush=True)
            return 0 if status.get("status_str") == "success" else 1
        q = get_json(args.url + "/queue")
        running = [x for x in q.get("queue_running", []) if x and x[1] == pid]
        pending = [x for x in q.get("queue_pending", []) if x and x[1] == pid]
        state = "运行中" if running else ("排队中" if pending else "未知")
        if state != last:
            print("  %s（已等 %.0fs）" % (state, time.time() - t0), flush=True)
            last = state
    print("!! 超时 %ds" % args.timeout, flush=True)
    return 3


if __name__ == "__main__":
    sys.exit(main())
