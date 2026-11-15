import sys, os, json, time, subprocess
sys.path.insert(0, "/data/data/com.termux/files/home/npt")
D = "/data/data/com.termux/files/home/npt"
OUT = D + "/full28"
man = json.load(open(OUT + "/manifest.json"))

def host(lines, timeout=400):
    p = subprocess.Popen([D+"/host"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
    out = []
    try:
        p.stdout.readline()
        for s in lines:
            p.stdin.write(s+"\n"); p.stdin.flush(); out.append(p.stdout.readline().strip())
        p.stdin.write('{"cmd":"quit"}\n'); p.stdin.flush(); p.wait(timeout=timeout)
    except Exception:
        p.kill(); out.append("CRASH")
    return out

total = 0.0
for b in man["blocks"]:
    gp = b["graph"]; tag = os.path.basename(gp)[:-4]
    blob = OUT + "/" + tag + ".blob"
    t0 = time.time()
    r = host(['{"cmd":"build","path":"%s"}' % gp, '{"cmd":"compile"}', '{"cmd":"export","path":"%s"}' % blob])
    dt = time.time() - t0; total += dt
    ok = len(r) > 1 and '"ok":true' in r[1]
    sz = os.path.getsize(blob) // 1024 if os.path.exists(blob) else 0
    print("%-12s %4d算子  compile=%-6s  %.1fs  blob=%dKB" %
          (tag, b["ops"], "OK" if ok else "FAIL", dt, sz), flush=True)
print("总编译 %.1fs" % total)
