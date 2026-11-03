import functools
import json
from pathlib import Path

import numpy as np

import torch
import torch.nn.functional as F
from torch import nn

from .data import beat_mel, beat_to_sec, sec_to_beat
from .tokens import STYLES, TICKS

PER_BEAT = 24
N_MELS = 80
N_IN = N_MELS + 1
CFG_STARS, CFG = [7, 10], [1.0, 1.0]
AUDIO_STARS, AUDIO_CFG = [8, 11], [0.0, 1.0]
SAMPLE_STARS, TEMP, TOP_P = [6, 12], [0.8, 0.8], [0.95, 0.95]
LOUD_CORR, LOUD_WEIGHT = 0.3, 1.0
AUTO_STARS = 0.0
STEP_BF16 = True
USUAL_RATINGS = [(2, 0.70, 0.57), (3, 1.46, 1.25), (4, 2.23, 2.01), (5, 2.90, 2.85), (6, 3.35, 3.56), (7, 3.94, 4.51),
                 (8, 4.59, 5.18), (9, 5.06, 6.26), (10, 5.52, 7.26), (11, 5.60, 8.54), (12, 6.11, 9.71), (13, 7.98, 10.15),
                 (14, 9.04, 10.86), (15, 10.40, 13.72)]
TRIES, TRIES_CHUNK = 64, 16
GAP_ACT, GAP_BEATS = 0.5, 32
EVEN_STARS, EVEN_MIN = [2, 5.5, 8, 10, 12, 14.5], [0.65, 0.60, 0.58, 0.53, 0.54, 0.47]
REPEAT_STARS, REPEAT_MAX, REPEAT_BARS = [3.5, 7.5, 10.5, 14], [0.73, 0.80, 0.83, 0.90], 55
MAX_BEATS = 1536


def snap(ticks):
    t = np.asarray(ticks, np.int64)
    a3, a4 = np.round(t / 3) * 3, np.round(t / 4) * 4
    return np.where(np.abs(a3 - t) <= np.abs(a4 - t), a3, a4).astype(np.int64)


class Vocab:

    def __init__(self, patterns):
        self.patterns = [tuple(p) for p in patterns]
        self.index = {p: i for i, p in enumerate(self.patterns)}
        self.bos = len(self.patterns)
        self.size = self.bos + 1
        self._near = {}

    def id(self, p):
        i = self.index.get(p)
        if i is None:
            i = self._near.get(p)
        if i is None:
            q = np.array(p)
            cost = [len(q) + len(r) if not len(r) or not len(q) else
                    int((np.abs(q[:, None] - np.array(r)[None]).min(1) > 3).sum() + (np.abs(np.array(r)[:, None] - q[None]).min(1) > 3).sum())
                    for r in self.patterns]
            i = self._near[p] = int(np.argmin(cost))
        return i

    def ids(self, patterns):
        return np.array([self.id(p) for p in patterns], np.int64)

    def ticks(self, ids, first_beat=0):
        return np.array([(first_beat + k) * TICKS + x for k, i in enumerate(ids) for x in self.patterns[int(i)]], np.int64)


def song_frames(logmel, fps, beat_times, offset, n_beats):
    n = n_beats * PER_BEAT
    m = beat_mel(logmel, fps, beat_times, offset, n=n, per_beat=PER_BEAT)
    lm = logmel.astype(np.float32)
    full = np.maximum(np.diff(lm, axis=0, prepend=lm[:1]), 0).sum(1)
    pos = full[full > 0]
    full = full / max(float(np.median(pos)) if len(pos) else 1.0, 1e-6)
    edges = beat_to_sec(offset + (np.arange(n + 1) - 0.5) / PER_BEAT, beat_times) * fps
    f = np.clip(np.round(edges).astype(np.int64), 0, len(full))
    a, b = f[:-1], np.maximum(f[1:], f[:-1] + 1)
    ok = a < len(full)
    onset = np.zeros(n, np.float32)
    if ok.any():
        k = np.maximum(b[ok] - a[ok], 1)
        lvl = int(np.floor(np.log2(k.max())))
        tab = [np.r_[full, 0.0]]
        for _ in range(lvl):
            t = tab[-1]
            s = 1 << (len(tab) - 1)
            tab.append(np.maximum(t, np.r_[t[s:], np.zeros(s)]))
        j = np.floor(np.log2(k)).astype(np.int64)
        lo, hi = a[ok], np.minimum(b[ok], len(full)) - (1 << j)
        hi = np.maximum(hi, lo)
        stack = np.stack(tab)
        onset[ok] = np.maximum(stack[j, lo], stack[j, hi])
    return np.concatenate([m, np.log1p(onset)[:, None]], 1).astype(np.float16)


def beat_seconds(beat_times, offset, n_beats):
    return np.log(np.maximum(np.diff(beat_to_sec(offset + np.arange(n_beats + 1, dtype=np.float64), beat_times)), 1e-2)).astype(np.float32)


def rope(T, dh, start=0, device=None):
    f = 1.0 / 10000 ** (torch.arange(0, dh, 2, device=device).float() / dh)
    a = torch.arange(start, start + T, device=device).float()[:, None] * f
    return torch.cos(a), torch.sin(a)


def _rot(x, cs):
    cos, sin = cs
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2).type_as(x)


class Block(nn.Module):

    def __init__(self, d, heads, dropout, cross=False):
        super().__init__()
        self.h, self.drop = heads, dropout
        self.n1, self.qkv, self.o = nn.LayerNorm(d), nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.cross = cross
        if cross:
            self.n2, self.q2, self.kv2, self.o2 = nn.LayerNorm(d), nn.Linear(d, d), nn.Linear(d, 2 * d), nn.Linear(d, d)
        self.n3 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.do = nn.Dropout(dropout)

    def split(self, y, n):
        B, T, _ = y.shape
        return y.view(B, T, n, self.h, -1).permute(2, 0, 3, 1, 4).unbind(0)

    def merge(self, y):
        B, h, T, dh = y.shape
        return y.transpose(1, 2).reshape(B, T, h * dh)

    def forward(self, x, cs, causal=False, mask=None, mem=None, mem_cs=None, mem_mask=None, cache=None):
        p = self.drop if self.training else 0.0
        q, k, v = self.split(self.qkv(self.n1(x)), 3)
        q, k = _rot(q, cs), _rot(k, cs)
        if cache is not None and "k" in cache:
            k, v = torch.cat([cache["k"], k], 2), torch.cat([cache["v"], v], 2)
        if cache is not None:
            cache["k"], cache["v"] = k, v
        if causal and mask is not None:
            T = q.shape[2]
            mask = mask & torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
            causal = False
        x = x + self.do(self.o(self.merge(F.scaled_dot_product_attention(q, k, v, mask, p, is_causal=causal and q.shape[2] > 1))))
        if self.cross:
            q = _rot(self.split(self.q2(self.n2(x)), 1)[0], cs)
            if cache is not None and "mk" in cache:
                mk, mv = cache["mk"], cache["mv"]
            else:
                mk, mv = self.split(self.kv2(mem), 2)
                mk = _rot(mk, mem_cs)
                if cache is not None:
                    cache["mk"], cache["mv"] = mk, mv
            x = x + self.do(self.o2(self.merge(F.scaled_dot_product_attention(q, mk, mv, mem_mask, p))))
        return x + self.do(self.ff(self.n3(x)))


class RhythmModel(nn.Module):

    def __init__(self, n_vocab, d=384, enc_layers=6, dec_layers=6, heads=6, dropout=0.1, stem=96, level=False,
                 ratings=False, audio_guidance=False):
        super().__init__()
        self.config = dict(n_vocab=n_vocab, d=d, enc_layers=enc_layers, dec_layers=dec_layers, heads=heads, stem=stem,
                           level=level, ratings=ratings, audio_guidance=audio_guidance)
        self.heads = heads
        self.stem = nn.Sequential(nn.Conv1d(N_IN, stem, 5, padding=2), nn.GELU(), nn.Conv1d(stem, stem, 5, padding=2), nn.GELU())
        self.beat_in = nn.Linear(PER_BEAT * stem, d)
        self.sec = nn.Linear(1, d)
        self.stars = nn.Sequential(nn.Linear(1, d), nn.GELU(), nn.Linear(d, d))
        self.no_stars = nn.Parameter(torch.zeros(d))
        self.style = nn.Embedding(len(STYLES) + 1, d)
        if level:
            self.level = nn.Sequential(nn.Linear(1, d), nn.GELU(), nn.Linear(d, d))
            self.no_level = nn.Parameter(torch.zeros(d))
        if ratings:
            self.ratings = nn.Sequential(nn.Linear(4, d), nn.GELU(), nn.Linear(d, d))
        if audio_guidance:
            self.no_audio = nn.Parameter(torch.zeros(d))
        self.enc = nn.ModuleList(Block(d, heads, dropout) for _ in range(enc_layers))
        self.enc_norm = nn.LayerNorm(d)
        self.tok = nn.Embedding(n_vocab, d)
        self.dec = nn.ModuleList(Block(d, heads, dropout, cross=True) for _ in range(dec_layers))
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, n_vocab - 1)

    def cond(self, stars, style, level=None, ratings=None):
        s = torch.nan_to_num(stars, 0.0)[:, None] / 10
        s = torch.where(torch.isnan(stars)[:, None], self.no_stars.expand(len(stars), -1), self.stars(s))
        if hasattr(self, "level"):
            if level is None:
                level = torch.full_like(stars, float("nan"))
            v = self.level(torch.nan_to_num(level, 0.0)[:, None])
            s = s + torch.where(torch.isnan(level)[:, None], self.no_level.expand(len(level), -1), v)
        if hasattr(self, "ratings"):
            if ratings is None:
                ratings = torch.full((len(stars), 2), float("nan"), device=stars.device)
            s = s + self.ratings(torch.cat([torch.nan_to_num(ratings, 0.0) / 10, torch.isnan(ratings).float()], 1))
        return s + self.style(style)

    def encode(self, frames, sec, stars, style, pad=None, level=None, ratings=None, no_audio=None):
        B, T = sec.shape
        y = self.beat_in(self.stem(frames.transpose(1, 2)).transpose(1, 2).reshape(B, T, -1))
        if no_audio is not None:
            y = torch.where(no_audio[:, None, None], self.no_audio.to(y.dtype), y)
        x = y + self.sec(sec[..., None]) + self.cond(stars, style, level, ratings)[:, None]
        cs = rope(T, x.shape[-1] // self.heads, device=x.device)
        mask = None if pad is None else ~pad[:, None, None, :]
        for b in self.enc:
            x = b(x, cs, mask=mask)
        return self.enc_norm(x), mask

    def decode(self, mem, prev, mem_mask=None, cond=None):
        T = prev.shape[1]
        cs = rope(T, mem.shape[-1] // self.heads, device=mem.device)
        x = self.tok(prev) + mem
        for b in self.dec:
            x = b(x, cs, causal=True, mask=mem_mask, mem=mem, mem_cs=cs, mem_mask=mem_mask)
        return self.out(self.norm(x))

    def forward(self, frames, sec, stars, style, prev, pad=None, level=None, ratings=None, no_audio=None):
        mem, mask = self.encode(frames, sec, stars, style, pad, level, ratings, no_audio)
        return self.decode(mem, prev, mask)

    def step(self, mem, prev, t, caches, mem_cs):
        cs = rope(1, mem.shape[-1] // self.heads, start=t, device=mem.device)
        x = self.tok(prev)[:, None] + mem[:, t:t + 1]
        for b, c in zip(self.dec, caches):
            x = b(x, cs, mem=mem, mem_cs=mem_cs, cache=c)
        return self.out(self.norm(x[:, 0]))


def style_index(style):
    return STYLES.index(style) if style in STYLES else len(STYLES)


@functools.lru_cache(2)
def loaded(path, dev="cuda"):
    return load(path, dev)


def load(path, dev="cuda"):
    path = Path(path)
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        ck = dict(json.loads(path.with_suffix(".json").read_text()), model=load_file(path))
    else:
        ck = torch.load(path, map_location="cpu", weights_only=False)
    model = RhythmModel(**ck["config"])
    model.load_state_dict(ck["model"])
    model.vocab = Vocab(ck["patterns"])
    model.usual = ck.get("usual")
    return model.to(dev).eval()


def usual_ratings(stars):
    s, t, p = np.array(USUAL_RATINGS, np.float64).T
    return float(np.interp(stars, s, t)), float(np.interp(stars, s, p))


def usual_level(model, stars):
    s, h = np.asarray(model.usual, np.float64).T
    return float(np.interp(stars, s, h))


@torch.no_grad()
def generate(model, logmel, fps, beat_times, offset, n_beats, stars, style=None, temp=1.0, top_p=0.95, cfg=1.0,
             n=1, level=None, ratings=None, dev="cuda", gen=None, audio_cfg=None):
    rows = np.broadcast_to(np.asarray(stars, np.float64), (n,))
    levels = list(level) if isinstance(level, list) else [level] * n
    if model.config.get("level") and model.usual is not None:
        levels = [usual_level(model, x) if v is None else v for v, x in zip(levels, rows)]
    rts = list(ratings) if isinstance(ratings, list) else [ratings] * n
    if model.config.get("ratings"):
        rts = [usual_ratings(x) if r is None else r for r, x in zip(rts, rows)]
    lv = torch.tensor([np.log(v) if v is not None and v == v else float("nan") for v in levels], dtype=torch.float32,
                      device=dev)
    rt = torch.tensor([[float("nan") if x is None else float(x) for x in (r or (None, None))] for r in rts], device=dev)
    per_row = lambda x: x if np.isscalar(x) else torch.tensor(np.asarray(x, np.float64), dtype=torch.float32,
                                                              device=dev)[:, None]
    guided = bool(np.any(np.asarray(cfg) != 1.0))
    audio_cfg = star_audio_cfg(rows) if audio_cfg is None else audio_cfg
    heard = bool(model.config.get("audio_guidance")) and bool(np.any(np.asarray(audio_cfg) != 0.0))
    temp, top_p, cfg, audio_cfg = per_row(temp), per_row(top_p), per_row(cfg), per_row(audio_cfg)
    frames = torch.from_numpy(song_frames(logmel, fps, beat_times, offset, n_beats).astype(np.float32)).to(dev)
    sec = torch.from_numpy(beat_seconds(beat_times, offset, n_beats)).to(dev)
    vocab = model.vocab
    out = [[] for _ in range(n)]
    start, done = 0, 0
    while done < n_beats:
        T = min(MAX_BEATS, n_beats - start)
        f = frames[start * PER_BEAT:(start + T) * PER_BEAT][None].expand(n, -1, -1)
        st = torch.tensor(rows, dtype=torch.float32, device=dev)
        sty = torch.full((n,), style_index(style), device=dev, dtype=torch.long)
        with torch.autocast("cuda", torch.bfloat16):
            mem, _ = model.encode(f, sec[start:start + T][None].expand(n, -1), st, sty, level=lv, ratings=rt)
            if guided:
                mem0, _ = model.encode(f, sec[start:start + T][None].expand(n, -1), torch.full_like(st, float("nan")), sty,
                                       level=torch.full_like(lv, float("nan")), ratings=torch.full_like(rt, float("nan")))
                mem = torch.cat([mem, mem0])
            if heard:
                mem1, _ = model.encode(f, sec[start:start + T][None].expand(n, -1), st, sty, level=lv, ratings=rt,
                                       no_audio=torch.ones(n, dtype=torch.bool, device=dev))
                mem = torch.cat([mem, mem1])
        mem = mem.float()
        mem_cs = rope(T, mem.shape[-1] // model.heads, device=dev)
        caches = [{} for _ in model.dec]
        B = mem.shape[0]
        prev = torch.full((B,), vocab.bos, device=dev, dtype=torch.long)
        for t in range(T):
            known = start + t < done
            with torch.autocast("cuda", torch.bfloat16, enabled=STEP_BF16):
                lg = model.step(mem, prev, t, caches, mem_cs).float()
            if guided or heard:
                c, k = lg[:n], 1
                out_lg = c
                if guided:
                    out_lg = lg[n:2 * n] + cfg * (c - lg[n:2 * n])
                    k = 2
                if heard:
                    out_lg = out_lg + audio_cfg * (c - lg[k * n:(k + 1) * n])
                lg = out_lg
            if known:
                tok = torch.tensor([out[i][start + t] for i in range(n)], device=dev)
            else:
                tok = sample_top_p(lg / temp, top_p, gen)
                for i, v in enumerate(tok.tolist()):
                    out[i].append(v)
            prev = tok.repeat(B // n)
        done = start + T
        start = done - MAX_BEATS // 2 if done < n_beats else done
    return [vocab.ticks(o) for o in out]


def drop_outs(ticks, logmel, fps, beat_times, offset, active=0.95, end=None, start=None):
    from .sample import window_activity
    t = np.asarray(ticks)
    if len(t) < 10:
        return 1.0, 1e9
    lo, hi = int(t.min() // TICKS), int(t.max() // TICKS)
    if end is not None:
        hi = max(hi, int(end))
    if start is not None:
        lo = min(lo, max(int(start), 0))
    sp = [(s, s + 32) for s in range(lo, hi - 31, 32)]
    thin = 0.0
    if len(sp) >= 2:
        act, dur = window_activity(logmel, fps, beat_times, offset, sp)
        c = np.array([((t >= a * TICKS) & (t < z * TICKS)).sum() for a, z in sp]) / dur
        busy = act >= 0.95
        thin = float(np.mean(c[busy] < 0.6 * np.median(c))) if busy.any() else 0.0
    ch = [(s, s + 8) for s in range(lo, hi - 7, 8)]
    run = longest = 0
    if ch:
        act, _ = window_activity(logmel, fps, beat_times, offset, ch)
        for (a, z), x in zip(ch, act):
            empty = not ((t >= a * TICKS) & (t < z * TICKS)).any()
            run = run + 1 if empty and x >= active else 0
            longest = max(longest, run)
    return thin, 8 * longest


def evenness(ticks, beat_times, offset):
    sec = np.sort(beat_to_sec(np.unique(np.asarray(ticks)) / TICKS + offset, beat_times))
    if len(sec) < 10:
        return 1.0
    n = np.histogram(sec, np.append(np.arange(sec[0], sec[-1], 8.0), sec[-1] + 1e-9))[0]
    return float(np.median(n) / max(np.percentile(n, 90), 1e-9))


def bar_repeat(ticks):
    t, bar = np.unique(np.asarray(ticks)), 4 * TICKS
    if not len(t):
        return 0.0
    keys = [tuple(t[(t >= s) & (t < s + bar)] - s) for s in range(int(t.min()) // bar * bar, int(t.max()) + 1, bar)]
    if len(keys) < REPEAT_BARS:
        return 0.0
    from collections import Counter
    return max(Counter(k for k in keys[i:i + REPEAT_BARS] if k).most_common(1)[0][1]
               for i in range(len(keys) - REPEAT_BARS + 1)) / REPEAT_BARS


SPIKE_MAX = {"acc": 2.9, "speed": 4.5}
PEAK_MAX = {"acc": 8.0}
SPIKE_DEFAULT = 3.5

BURST_SECS = 2.0
BURST_STARS = (2, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 13)
BURST_MAX = (4.5, 6.5, 8.5, 9.6, 12.5, 14.0, 17.0, 19.5, 22.5, 27.5, 29.6)


def spike(ticks, beat_times, offset):
    sec = np.sort(beat_to_sec(np.unique(np.asarray(ticks)) / TICKS + offset, beat_times))
    if len(sec) < 10:
        return 1.0, 0.0
    n = np.histogram(sec, np.append(np.arange(sec[0], sec[-1], 8.0), sec[-1] + 1e-9))[0] / 8
    peak = max(float(np.max([((sec >= a) & (sec < a + 8)).sum() / 8 for a in sec[::4]])), float(n.max()))
    return peak / max(float(np.median(n)), 0.1), peak


def burst(ticks, beat_times, offset):
    sec = np.sort(beat_to_sec(np.unique(np.asarray(ticks)) / TICKS + offset, beat_times))
    if len(sec) < 2:
        return 0.0
    return float((np.searchsorted(sec, sec + BURST_SECS, "left") - np.arange(len(sec))).max()) / BURST_SECS


def star_cfg(stars):
    return float(np.interp(stars, CFG_STARS, CFG))


def star_audio_cfg(stars):
    return np.interp(stars, AUDIO_STARS, AUDIO_CFG)


def loud_corr(ticks, logmel, fps, beat_times, offset):
    from .sample import window_activity
    t = np.asarray(ticks)
    lo, hi = int(t.min() // TICKS), int(t.max() // TICKS)
    sp = [(s, s + 8) for s in range(lo, hi - 7, 8)]
    if len(sp) < 4:
        return 0.0
    act, _ = window_activity(logmel, fps, beat_times, offset, sp)
    c = np.array([((t >= a * TICKS) & (t < z * TICKS)).sum() for a, z in sp], float)
    rank = lambda x: np.argsort(np.argsort(x, kind="stable"), kind="stable").astype(float)
    rc, ra = rank(c), rank(act)
    return float(np.corrcoef(rc, ra)[0, 1]) if rc.std() > 0 and ra.std() > 0 else 0.0


def star_sampling(stars):
    return float(np.interp(stars, SAMPLE_STARS, TEMP)), float(np.interp(stars, SAMPLE_STARS, TOP_P))


def best_of(model, logmel, fps, beat_times, offset, n_beats, stars, tries=TRIES, level=None, cfg=None, temp=None,
            top_p=None, ratings=None, wanted=None, **kw):
    many = not np.isscalar(stars)
    sl = list(stars) if many else [stars]
    m = len(sl)
    lvl = list(level) if many and isinstance(level, list) else [level] * m
    rl = list(ratings) if many and isinstance(ratings, list) else [ratings] * m
    cf = [star_cfg(x) if cfg is None else cfg for x in sl]
    tp = [star_sampling(x) for x in sl]
    te, pp = [t if temp is None else temp for t, _ in tp], [p if top_p is None else top_p for _, p in tp]
    style = kw.get("style")
    gens = [[] for _ in sl]
    for k in range(0, tries, TRIES_CHUNK):
        c = min(TRIES_CHUNK, tries - k)
        if not many:
            gens[0] += generate(model, logmel, fps, beat_times, offset, n_beats, stars, n=c, level=level, cfg=cf[0],
                                temp=te[0], top_p=pp[0], ratings=ratings, **kw)
            continue
        rep = lambda xs: [x for x in xs for _ in range(c)]
        out = generate(model, logmel, fps, beat_times, offset, n_beats, rep(sl), n=c * m, level=rep(lvl), cfg=rep(cf),
                       temp=rep(te), top_p=rep(pp), ratings=rep(rl), **kw)
        for j in range(m):
            gens[j] += out[j * c:(j + 1) * c]
    if tries == 1:
        return [g[0] for g in gens] if many else gens[0][0]

    from .sample import song_end, song_start
    start, end = sec_to_beat(np.array([song_start(logmel, fps), song_end(logmel, fps)]), beat_times) - offset
    start, end = max(start, 0.0), min(end, n_beats)

    def score(t, want):
        thin, drop = drop_outs(t, logmel, fps, beat_times, offset, end=end, start=start)
        if len(t) < 10:
            return 4 * thin + drop / 32
        drop += max(drop_outs(t, logmel, fps, beat_times, offset, GAP_ACT, end=end, start=start)[1] - GAP_BEATS, 0)
        loud = LOUD_WEIGHT * max(loud_corr(t, logmel, fps, beat_times, offset) - LOUD_CORR, 0.0)
        if want is None:
            return 4 * thin + drop / 32 + loud
        span = np.diff(beat_to_sec(np.array([min(t.min() / TICKS, start), max(t.max() / TICKS, end)]) + offset,
                                   beat_times))[0]
        return 4 * thin + drop / 32 + abs(np.log(len(t) / max(span, 1.0) / want)) + loud
    picks = []
    for g, x, v, aim in zip(gens, sl, lvl, sl if wanted is None else wanted if many else [wanted]):
        want = v if v is not None and v == v else usual_level(model, x) if model.usual else None
        floor = float(np.interp(x, EVEN_STARS, EVEN_MIN))
        even = [t for t in g if evenness(t, beat_times, offset) >= floor] or g
        top = float(np.interp(x, REPEAT_STARS, REPEAT_MAX))
        varied = [t for t in even if bar_repeat(t) <= top] or even
        cap, peak_cap = SPIKE_MAX.get(style, SPIKE_DEFAULT), PEAK_MAX.get(style, np.inf)
        spikes = [spike(t, beat_times, offset) for t in varied]
        calm = [t for t, (r, p) in zip(varied, spikes) if r <= cap and p <= peak_cap]
        if not calm:
            calm = [min(zip(varied, spikes), key=lambda q: max(q[1][0] / cap, q[1][1] / peak_cap))[0]]
        most = float(np.interp(aim, BURST_STARS, BURST_MAX))
        rates = [burst(t, beat_times, offset) for t in calm]
        calm = [t for t, r in zip(calm, rates) if r <= most] or [calm[int(np.argmin(rates))]]
        picks.append(min(calm, key=lambda t: score(t, want)))
    return picks if many else picks[0]


def sample_top_p(logits, p, gen=None):
    probs = logits.softmax(-1)
    s, i = probs.sort(-1, descending=True)
    keep = s.cumsum(-1) - s < p
    s = s * keep
    return i.gather(-1, torch.multinomial(s / s.sum(-1, keepdim=True), 1, generator=gen))[:, 0]
