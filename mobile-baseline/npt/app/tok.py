"""纯 Python tokenizer：Qwen2/3 系的 GPT-2 式 BPE，以及 T5 的 unigram。

不依赖 HuggingFace tokenizers 的 Rust 扩展（Termux 上 ABI 不兼容）。
"""
import json
import re
import bisect


def _bytes_to_unicode():
    bs = (list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) +
          list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


B2U = _bytes_to_unicode()
U2B = {v: k for k, v in B2U.items()}


class QwenTok(object):
    """Qwen2/Qwen3 的字节级 BPE（vocab.json + merges.txt）。"""
    # GPT-2 / Qwen 的预切分：注意空格要并进后面的词（"Ġwhite" 是一个整体）
    PAT = re.compile(
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
        r"| ?[^\r\n\W\d_]+"
        r"| ?\d{1,3}"
        r"| ?[^\s\w]+[\r\n]*"
        r"|\s*[\r\n]+"
        r"|\s+(?!\S)"
        r"|\s+", re.UNICODE)

    def __init__(self, vocab_path, merges_path):
        self.vocab = json.load(open(vocab_path, encoding="utf-8"))
        self.ranks = {}
        with open(merges_path, encoding="utf-8") as f:
            lines = f.read().split("\n")
        idx = 0
        if lines and lines[0].startswith("#"):
            idx = 1
        for i, ln in enumerate(lines[idx:]):
            ln = ln.strip()
            if not ln:
                continue
            p = ln.split(" ")
            if len(p) == 2:
                self.ranks[(p[0], p[1])] = i
        self.cache = {}

    def _bpe(self, token):
        got = self.cache.get(token)
        if got is not None:
            return got
        word = list(token)
        while len(word) > 1:
            best, bi = None, None
            for i in range(len(word) - 1):
                r = self.ranks.get((word[i], word[i + 1]))
                if r is not None and (best is None or r < best):
                    best, bi = r, i
            if bi is None:
                break
            word = word[:bi] + [word[bi] + word[bi + 1]] + word[bi + 2:]
        self.cache[token] = word
        return word

    def encode(self, text, add_bos=False, bos_id=None):
        ids = []
        for piece in self.PAT.findall(text):
            bt = "".join(B2U[b] for b in piece.encode("utf-8"))
            for w in self._bpe(bt):
                v = self.vocab.get(w)
                if v is not None:
                    ids.append(v)
        if add_bos and bos_id is not None:
            ids = [bos_id] + ids
        return ids


class T5Tok(object):
    """T5 的 sentencepiece unigram（读 tokenizer.json 的 Unigram model）。"""

    def __init__(self, tok_json):
        d = json.load(open(tok_json, encoding="utf-8"))
        m = d["model"]
        self.vocab = m["vocab"]                    # [ [piece, log_prob], ... ]
        self.unk_id = m.get("unk_id", 2)
        self.pieces = [p for p, _ in self.vocab]
        self.scores = [s for _, s in self.vocab]
        self.lookup = {p: i for i, p in enumerate(self.pieces)}
        self.max_len = max((len(p) for p in self.pieces), default=1)
        # 特殊 token：<pad>=0, </s>=1, <unk>=2
        self.pad_id = self.lookup.get("<pad>", 0)
        self.eos_id = self.lookup.get("</s>", 1)

    def _encode_piece(self, s):
        n = len(s)
        best = [0.0] + [-1e30] * n
        back = [0] * (n + 1)
        for i in range(1, n + 1):
            for j in range(max(0, i - self.max_len), i):
                sub = s[j:i]
                vi = self.lookup.get(sub)
                sc = (self.scores[vi] if vi is not None else -100.0)
                cand = best[j] + sc
                if cand > best[i]:
                    best[i] = cand
                    back[i] = j
        out = []
        i = n
        while i > 0:
            j = back[i]
            out.append(self.lookup.get(s[j:i], self.unk_id))
            i = j
        return list(reversed(out))

    def encode(self, text, add_eos=True):
        text = text.replace("\u2581", " ")
        s = "\u2581" + text.replace(" ", "\u2581")
        ids = []
        # 按 unigram 常见做法：特殊符号单独切
        buf = ""
        for ch in s:
            if ch in "<>":
                if buf:
                    ids.extend(self._encode_piece(buf)); buf = ""
                buf += ch
            else:
                buf += ch
        if buf:
            ids.extend(self._encode_piece(buf))
        if add_eos:
            ids.append(self.eos_id)
        return ids
