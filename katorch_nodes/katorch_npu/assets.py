"""资产层：扫描手机上的 NPU blob 集，建立 manifest。

一个「资产」= 一份编译好的 DiT 图（pe + b00..bNN + fl）+ 对应的校准文件 + 分辨率。
分辨率在编译时被固化进 blob（MDLA 是静态图），所以「换分辨率」= 换一个 blob 目录。

manifest 是**缓存**，不是事实来源：每次加载会比对目录 mtime 与文件清单，
发现变化就重扫。手写的修正放在 `$KA_WORK/assets/overrides.json`，优先级最高。

overrides.json 例子：
    {"blobs512m": {"width": 512, "height": 512, "calib": "/path/to/calib.json",
                   "name": "anima_v1@512"}}
"""
import json
import os
import re
import time

HOME = os.environ.get("KA_HOME", "/data/data/com.termux/files/home")
WORK = os.environ.get("KA_WORK", os.path.join(HOME, "npt"))
MODELS = os.environ.get("KA_MODELS", os.path.join(HOME, "ComfyUI", "models"))
ASSET_DIR = os.path.join(WORK, "assets")
MANIFEST = os.path.join(ASSET_DIR, "manifest.json")
OVERRIDES = os.path.join(ASSET_DIR, "overrides.json")

BLOB_ROOTS = [p for p in os.environ.get("KA_BLOB_ROOTS", "/data/local/tmp:" + WORK).split(":") if p]

PE = "pe"
FL = "fl"


def _log(msg):
    print("[katorch.assets] " + str(msg), flush=True)


def _isdir(p):
    try:
        return os.path.isdir(p)
    except OSError:
        return False


def _listdir(p):
    try:
        return sorted(os.listdir(p))
    except OSError:
        return []


def _size(p):
    try:
        return os.path.getsize(p)
    except OSError:
        return 0


def _load_json(p):
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _save_json(p, obj):
    try:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        tmp = p + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=1)
        os.replace(tmp, p)
    except Exception as e:
        _log("写 manifest 失败：%s" % e)


def _count_blocks(names):
    """连续 b00.dla, b01.dla ... 的个数。"""
    n = 0
    while ("b%02d.dla" % n) in names:
        n += 1
    return n


def _guess_resolution(dirname, tokens=None):
    """从目录名解析边长（blobs512m → 512）。

    注意：不能用 calib 里的 S 字段反推 —— calib8.json 的 S=256 指的是标定时用的
    采样 token 数，不是 latent 边长，拿它算会得到 256 而不是 512。
    """
    m = re.search(r"(\d{3,4})", dirname)
    if m:
        v = int(m.group(1))
        if v % 16 == 0:
            return v, v
    return None, None


def _calib_candidates(blob_dir):
    out = []
    for name in ("calib.json", "calib8.json"):
        p = os.path.join(blob_dir, name)
        if os.path.isfile(p):
            out.append(p)
    g = os.path.join(WORK, "gen")
    if _isdir(g):
        for name in sorted(_listdir(g)):
            if name.startswith("calib") and name.endswith(".json"):
                out.append(os.path.join(g, name))
    return out


def _pick_calib(blob_dir, n_blocks):
    """选与块数匹配的校准文件；返回 (path, data)。

    手机上有 calib4/5/6/7/8 好几份，post_scale 各不相同，用错画质直接崩。
    默认优先 calib8.json（当前管线验证过的那份），其余靠 overrides 显式指定。
    """
    cands = _calib_candidates(blob_dir)
    parsed = []
    for p in cands:
        d = _load_json(p)
        if not isinstance(d, dict):
            continue
        pb = d.get("per_block")
        if not isinstance(pb, list):
            continue
        if len(pb) != 2 * n_blocks:
            continue
        parsed.append((p, d))
    if not parsed:
        return (None, None)
    for p, d in parsed:
        if os.path.basename(p) == "calib8.json":
            return p, d
    # 否则取文件名里数字最大的（通常是最新的标定）
    def _num(p):
        m = re.search(r"(\d+)\.json$", p)
        return int(m.group(1)) if m else 0
    parsed.sort(key=lambda x: _num(x[0]))
    return parsed[-1]


def _scan_blob_dir(path):
    """检查一个目录是不是完整的 blob 集；是则返回 info。"""
    names = set(_listdir(path))
    if PE + ".dla" not in names or FL + ".dla" not in names:
        return None
    n_blocks = _count_blocks(names)
    if n_blocks < 1:
        return None
    order = [PE] + ["b%02d" % i for i in range(n_blocks)] + [FL]
    files = []
    total = 0
    for n in order:
        p = os.path.join(path, n + ".dla")
        sz = _size(p)
        if sz <= 0:
            return None
        files.append({"name": n, "path": p, "bytes": sz})
        total += sz
    # 完整性：真品每块约 277MB、pe 约 300KB、fl 约 3.4MB；明显小一圈的是中途失败或链接坏掉的残件
    block_bytes = [f["bytes"] for f in files if f["name"].startswith("b")]
    avg_block = sum(block_bytes) / max(len(block_bytes), 1)
    pe_bytes = _size(os.path.join(path, PE + ".dla"))
    fl_bytes = _size(os.path.join(path, FL + ".dla"))
    complete = (avg_block > 100 * 1024 * 1024
                and pe_bytes > 100 * 1024
                and fl_bytes > 1024 * 1024)
    calib_path, calib = _pick_calib(path, n_blocks)
    w, h = _guess_resolution(os.path.basename(path.rstrip("/")))
    tokens = (w // 16) ** 2 if w else None
    return {
        "id": os.path.basename(path.rstrip("/")),
        "blob_dir": path,
        "blocks": n_blocks,
        "order": order,
        "files": files,
        "bytes": total,
        "calib": calib_path,
        "width": w,
        "height": h,
        "tokens": tokens,
        "latent": [w // 8, h // 8] if w else None,
        "complete": complete,
        "verified": False,
        "mtime": int(os.path.getmtime(path)),
    }


def _iter_blob_dirs():
    """遍历候选目录：根目录本身 + 一层子目录。"""
    seen = set()
    for root in BLOB_ROOTS:
        if not _isdir(root):
            continue
        yield root
        for name in _listdir(root):
            p = os.path.join(root, name)
            if p in seen:
                continue
            if _isdir(p):
                seen.add(p)
                yield p


_CACHE = {"t": 0.0, "v": None}


def scan(force=False, ttl=15.0):
    """扫描资产。返回 manifest dict（同时写入缓存文件供别的工具读）。

    ComfyUI 前端每次拉 object_info 都会问一次下拉框，所以带 15 秒 TTL 缓存。
    """
    now = time.time()
    if not force and _CACHE["v"] is not None and (now - _CACHE["t"]) < ttl:
        return _CACHE["v"]
    os.makedirs(ASSET_DIR, exist_ok=True)
    overrides = _load_json(OVERRIDES) or {}

    fresh = {}
    for path in _iter_blob_dirs():
        info = _scan_blob_dir(path)
        if info is None:
            continue
        fresh[info["id"]] = info

    # 手工修正
    for k, v in overrides.items():
        if k.startswith("_"):
            continue
        if k in fresh:
            fresh[k].update(v)
        else:
            _log("overrides 里的 %s 没找到对应 blob 目录，忽略" % k)

    # 补全派生字段
    for a in fresh.values():
        w, h = a.get("width"), a.get("height")
        if w and h:
            a["latent"] = [w // 8, h // 8]
            a["tokens"] = (w // 16) * (h // 16)
        a.setdefault("model", "anima_v1")
        a.setdefault("name", "%s@%s" % (a["model"], a.get("width") or "?"))
        a.setdefault("complete", True)
        a.setdefault("verified", False)

    manifest = {
        "version": 2,
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "home": HOME,
        "work": WORK,
        "models_dir": MODELS,
        "assets": sorted(fresh.values(), key=lambda a: (-(a.get("width") or 0), a["id"])),
        "clips": _scan_dir(os.path.join(MODELS, "clip")),
        "text_encoders": _scan_dir(os.path.join(MODELS, "text_encoders")),
        "vaes": _scan_dir(os.path.join(MODELS, "vae")),
        "loras": _scan_dir(os.path.join(MODELS, "loras")),
        "diffusion": _scan_dir(os.path.join(MODELS, "diffusion_models")),
    }
    _save_json(MANIFEST, manifest)
    _CACHE["t"] = now
    _CACHE["v"] = manifest
    return manifest


def _scan_dir(path):
    out = []
    for name in _listdir(path):
        if name.endswith((".safetensors", ".sft", ".ckpt", ".pt", ".bin")):
            out.append({"name": name, "path": os.path.join(path, name), "bytes": _size(os.path.join(path, name))})
    return out


def get_manifest(force=False):
    return scan(force)


def find_asset(manifest, asset_id):
    for a in manifest.get("assets", []):
        if a["id"] == asset_id or a.get("name") == asset_id:
            return a
    return None


def pick_asset_for(manifest, model, width, height):
    """按模型名 + 分辨率选资产；分辨率不可用时给出明确错误。

    优先在「已验证」的资产里找 —— 手机上有好几份历史编译产物，其中 `blobs` 是
    实验残留（pe 是 256 token 的图、fl 是 512 token 的图，拼在一起，跑起来
    第一个块就崩），未验证的目录不能自动选用。
    """
    cands = [a for a in manifest.get("assets", []) if a.get("model") == model]
    verified = [a for a in cands if a.get("verified")]
    if verified:
        cands = verified
    for a in cands:
        if a.get("width") == width and a.get("height") == height:
            return a
    sizes = sorted(set("%sx%s" % (a.get("width"), a.get("height")) for a in cands))
    raise RuntimeError(
        "没有 %s @ %dx%d 的编译产物。NPU 是静态图，换分辨率必须重新编译。\n"
        "当前可用：%s\n编译好的 blob 目录需要放在 %s 下，并在 %s 里登记。"
        % (model, width, height, "、".join(sizes) or "（无）", " / ".join(BLOB_ROOTS), OVERRIDES)
    )


def asset_options(manifest):
    """给下拉框用的 [(值, id)]。

    值就是资产 id（如 blobs512m）—— 因为 ComfyUI 会校验下拉值必须在列表里，
    用带中文和体积的显示名会让 API 工作流和脚本非常难写。

    默认只列「已验证」的资产：手机上有好几份历史编译产物（旧的、量化的、残件），
    选错了画质直接崩，而界面上看不出区别。要显示全部就设 KA_SHOW_ALL_ASSETS=1，
    或者去 overrides.json 把想用的那份标成 "verified": true。
    """
    usable = [a for a in manifest.get("assets", [])
              if a.get("width") and a.get("complete", True)]
    verified = [a for a in usable if a.get("verified")]
    show_all = os.environ.get("KA_SHOW_ALL_ASSETS", "") not in ("", "0")
    pool = usable if show_all else (verified or usable)
    pool.sort(key=lambda a: (not a.get("verified"), a.get("width") or 0))
    return [(a["id"], a["id"]) for a in pool]
