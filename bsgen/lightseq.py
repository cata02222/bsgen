import collections
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from . import showlang
from .rhythm import Block, _rot

TPB = showlang.TICKS_PER_BEAT
AUDIO_PER_BEAT = 12
MELS = 80
SONG_INPUTS = 3
N_IN = MELS + SONG_INPUTS
MAX_BEATS = 1536
N_WHEN, N_WHAT, N_MOTION = 1024, 4096, 512
WHEN, WHAT, MOTION = range(3)
SPIN_MAX, ZOOM_MAX = 2, 1
NOTE_BITS = 2


def look_key(act_row, col_row):
    return tuple((int(a), int(c) if a >= showlang.ON else 0) for a, c in zip(act_row, col_row))


def motion_key(spin, zoom, sweep_l, sweep_r):
    return (min(int(spin), SPIN_MAX), min(int(zoom), ZOOM_MAX), int(sweep_l), int(sweep_r))


def show_items(m, n_beats):
    out = []
    act, col = m["act"], m["col"]
    for b in range(n_beats):
        a = act[b * TPB:(b + 1) * TPB]
        if not len(a):
            break
        ticks = np.nonzero((a != showlang.NONE).any(1))[0]
        looks = [look_key(a[t], col[b * TPB + t]) for t in ticks]
        sw = m["sweep"][b * TPB:(b + 1) * TPB]
        last = [int(sw[sw[:, s] >= 0, s][-1]) if (sw[:, s] >= 0).any() else -1 for s in range(2)]
        mo = motion_key(m["spin"][b * TPB:(b + 1) * TPB].sum(), m["zoom"][b * TPB:(b + 1) * TPB].sum(), *last)
        out.append((tuple(int(t) for t in ticks), looks, mo))
    return out


class Vocab:

    def __init__(self, when, what, motion):
        self.when, self.what, self.motion = [tuple(p) for p in when], [tuple(map(tuple, x)) for x in what], [tuple(x) for x in motion]
        self.base = (0, len(self.when), len(self.when) + len(self.what))
        self.bos = self.base[2] + len(self.motion)
        self.size = self.bos + 1
        self.index = [{p: i for i, p in enumerate(v)} for v in (self.when, self.what, self.motion)]
        self._near = [{}, {}, {}]
        self._arr = [None, np.array([[p for pair in x for p in pair] for x in self.what]), np.array(self.motion)]

    def kind_range(self, kind):
        return self.base[kind], self.base[kind] + (len(self.when), len(self.what), len(self.motion))[kind]

    def id(self, kind, item):
        i = self.index[kind].get(item)
        if i is None:
            i = self._near[kind].get(item)
        if i is None:
            if kind == WHEN:
                q = np.array(item)
                cost = [len(q) + len(r) if not len(r) or not len(q) else
                        int((np.abs(q[:, None] - np.array(r)[None]).min(1) > 1).sum() + (np.abs(np.array(r)[:, None] - q[None]).min(1) > 1).sum())
                        + 0.01 * abs(len(q) - len(r)) for r in self.when]
                same = [k for k, r in enumerate(self.when) if len(r) == len(item)]
                i = min(same, key=lambda k: cost[k]) if same else int(np.argmin(cost))
            elif kind == WHAT:
                q = np.array([p for pair in item for p in pair])
                i = int(np.argmin((self._arr[1] != q).sum(1)))
            else:
                i = int(np.argmin((self._arr[2] != np.array(item)).sum(1)))
            self._near[kind][item] = i
        return self.base[kind] + i

    def item(self, tok):
        tok = int(tok)
        if tok < self.base[1]:
            return WHEN, self.when[tok]
        if tok < self.base[2]:
            return WHAT, self.what[tok - self.base[1]]
        return MOTION, self.motion[tok - self.base[2]]

    def to_json(self):
        return json.dumps(dict(when=self.when, what=self.what, motion=self.motion))

    @classmethod
    def from_json(cls, s):
        d = json.loads(s)
        return cls(d["when"], d["what"], d["motion"])


def encode_show(items, vocab):
    tok, kind, beat, tick = [], [], [], []
    for b, (when, looks, mo) in enumerate(items):
        w = vocab.id(WHEN, when)
        ticks = vocab.when[w]
        if len(ticks) != len(looks):
            looks = (looks + looks[-1:] * len(ticks))[:len(ticks)] if looks else []
        tok.append(w); kind.append(WHEN); beat.append(b); tick.append(0)
        for t, lk in zip(ticks, looks):
            tok.append(vocab.id(WHAT, lk)); kind.append(WHAT); beat.append(b); tick.append(t)
        tok.append(vocab.id(MOTION, mo)); kind.append(MOTION); beat.append(b); tick.append(TPB - 1)
    return (np.array(tok, np.int16), np.array(kind, np.int8), np.array(beat, np.int16), np.array(tick, np.int8))


def decode_show(tokens, vocab, n_beats, first_beat=0):
    n = n_beats * TPB
    m = dict(act=np.zeros((n, 5), np.int8), col=np.zeros((n, 5), np.int8), spin=np.zeros(n, np.uint8),
             zoom=np.zeros(n, np.uint8), sweep=np.full((n, 2), -1, np.int8))
    b, ticks, k = first_beat - 1, (), 0
    for tok in tokens:
        kind, it = vocab.item(tok)
        if kind == WHEN:
            b, ticks, k = b + 1, it, 0
        elif kind == WHAT and 0 <= b < n_beats and k < len(ticks):
            t = b * TPB + ticks[k]
            for p, (a, c) in enumerate(it):
                m["act"][t, p], m["col"][t, p] = a, c
            k += 1
        elif kind == MOTION and 0 <= b < n_beats:
            t = b * TPB + (ticks[0] if ticks else 0)
            spin, zoom, sl, sr = it
            m["spin"][t] += spin
            m["zoom"][t] += zoom
            if sl >= 0:
                m["sweep"][t, 0] = sl
            if sr >= 0:
                m["sweep"][t, 1] = sr
    return m


def show_kind(items):
    n = max(len(items), 1)
    moments = sum(len(w) for w, _, _ in items)
    acts = [a for _, looks, _ in items for lk in looks for a, _ in lk if a >= showlang.ON]
    fade = float(np.mean(np.array(acts) == showlang.FADE)) if acts else float("nan")
    return float(np.log(moments / n + 0.1)), fade


UNKNOWN, TOOL, COPY, HAND = range(4)
MADE = ("unknown", "tool", "copy", "hand")
NOT_LIKED, LIKED = 1, 2


def made_marks(items, note_ticks=None):
    nan = float("nan")
    zoom = np.array([b for b, (_, _, mo) in enumerate(items) if mo[1] > 0])
    spin = np.array([b for b, (_, _, mo) in enumerate(items) if mo[0] > 0])
    sweep = np.array([v for _, _, mo in items for v in mo[2:] if v >= 0])
    gaps = np.diff(spin)
    gaps = gaps[gaps >= 2]
    out = dict(zoom8=float((np.diff(zoom) == 8).mean()) if len(zoom) > 4 else nan,
               spin4=float((gaps == 4).mean()) if len(gaps) > 4 else nan,
               sweep_odd=float(np.isin(sweep, (1, 3, 5, 7)).mean()) if len(sweep) > 10 else nan, on_note=nan, note_lit=nan)
    if note_ticks is not None:
        nt = np.asarray(note_ticks).reshape(-1)[:len(items) * TPB] > 0
        lit = np.zeros(len(nt), bool)
        g = np.array([b * TPB + t for b, (w, _, _) in enumerate(items) for t in w], np.int64)
        lit[g[g < len(nt)]] = True
        if lit.sum() >= 20 and nt.sum() >= 20:
            out["on_note"], out["note_lit"] = float(nt[lit].mean()), float(lit[nt].mean())
    return out


def made_by(items, note_ticks=None):
    k = made_marks(items, note_ticks)
    if k["zoom8"] >= 0.7 or (k["spin4"] >= 0.7 and k["sweep_odd"] >= 0.99):
        return TOOL
    if k["on_note"] >= 0.9 and k["note_lit"] >= 0.9:
        return COPY
    return HAND


def rope_at(pos, dh):
    f = 1.0 / 10000 ** (torch.arange(0, dh, 2, device=pos.device).float() / dh)
    a = pos.float()[..., None] * f
    return torch.cos(a)[:, None], torch.sin(a)[:, None]


class LightSeq(nn.Module):

    def __init__(self, n_vocab, d=256, enc_layers=4, dec_layers=4, heads=4, dropout=0.1, stem=64, tags=False, whole=False):
        super().__init__()
        self.config = dict(n_vocab=n_vocab, d=d, enc_layers=enc_layers, dec_layers=dec_layers, heads=heads, stem=stem, tags=tags,
                           whole=whole)
        self.heads = heads
        self.stem = nn.Sequential(nn.Conv1d(N_IN, stem, 3, padding=1), nn.GELU(), nn.Conv1d(stem, stem, 3, padding=1), nn.GELU())
        self.beat_in = nn.Linear(AUDIO_PER_BEAT * stem + TPB * NOTE_BITS, d)
        self.sec = nn.Linear(1, d)
        self.kind_in = nn.Sequential(nn.Linear(2, d), nn.GELU(), nn.Linear(d, d))
        self.no_busy, self.no_fade = nn.Parameter(torch.zeros(d)), nn.Parameter(torch.zeros(d))
        self.no_audio = nn.Parameter(torch.zeros(d))
        self.phase = nn.Embedding(4, d)
        if tags:
            self.made, self.liked = nn.Embedding(len(MADE), d), nn.Embedding(3, d)
            nn.init.zeros_(self.made.weight); nn.init.zeros_(self.liked.weight)
        self.enc = nn.ModuleList(Block(d, heads, dropout) for _ in range(enc_layers))
        self.enc_norm = nn.LayerNorm(d)
        self.tok = nn.Embedding(n_vocab, d)
        self.slot = nn.Embedding(3 * TPB, d)
        self.note_at = nn.Embedding(1 << NOTE_BITS, d)
        self.dec = nn.ModuleList(Block(d, heads, dropout, cross=True) for _ in range(dec_layers))
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, n_vocab - 1)

    def encode(self, frames, notes, sec, busy, fade, pad=None, no_audio=None, made=None, liked=None):
        B, T = sec.shape
        y = self.stem(frames.transpose(1, 2)).transpose(1, 2).reshape(B, T, -1)
        if no_audio is not None:
            y = y * (~no_audio)[:, None, None].to(y.dtype)
        nb = torch.stack([(notes >> k) & 1 for k in range(NOTE_BITS)], -1).reshape(B, T, -1).to(y.dtype)
        x = self.beat_in(torch.cat([y, nb], -1)) + self.sec(sec[..., None])
        if no_audio is not None:
            x = x + torch.where(no_audio[:, None, None], self.no_audio, 0.0)
        k = self.kind_in(torch.stack([torch.nan_to_num(busy, 0.0), torch.nan_to_num(fade, 0.0)], -1))
        k = k + torch.where(torch.isnan(busy)[:, None], self.no_busy, 0.0) + torch.where(torch.isnan(fade)[:, None], self.no_fade, 0.0)
        if self.config.get("tags") and made is not None:
            k = k + self.made(made) + self.liked(liked if liked is not None else torch.zeros_like(made))
        x = x + k[:, None] + self.phase(torch.arange(T, device=x.device) % 4)[None]
        cs = rope_at(torch.arange(T, device=x.device)[None].expand(B, T), x.shape[-1] // self.heads)
        mask = None if pad is None else ~pad[:, None, None, :]
        for b in self.enc:
            x = b(x, cs, mask=mask)
        return self.enc_norm(x), mask

    def embed(self, mem, prev, kind, beat, tick, note):
        idx = beat.long().clamp(0, mem.shape[1] - 1)
        at = torch.gather(mem, 1, idx[..., None].expand(-1, -1, mem.shape[-1]))
        return self.tok(prev) + self.slot(kind.long() * TPB + tick.long()) + self.note_at(note.long()) + at

    def decode(self, mem, mem_mask, prev, kind, beat, tick, note):
        B, T = prev.shape
        dh = mem.shape[-1] // self.heads
        cs = rope_at(beat.float() + tick.float() / TPB, dh)
        mem_cs = rope_at(torch.arange(mem.shape[1], device=mem.device)[None].expand(B, -1), dh)
        x = self.embed(mem, prev, kind, beat, tick, note)
        for b in self.dec:
            x = b(x, cs, causal=True, mem=mem, mem_cs=mem_cs, mem_mask=mem_mask)
        return self.out(self.norm(x))

    def step(self, mem, mem_cs, prev, kind, beat, tick, note, caches, keep=1024):
        dh = mem.shape[-1] // self.heads
        cs = rope_at((beat.float() + tick.float() / TPB)[:, None], dh)
        x = self.embed(mem, prev[:, None], kind[:, None], beat[:, None], tick[:, None], note[:, None])
        for b, c in zip(self.dec, caches):
            x = b(x, cs, mem=mem, mem_cs=mem_cs, cache=c)
            if c["k"].shape[2] > keep:
                c["k"], c["v"] = c["k"][:, :, -keep:], c["v"][:, :, -keep:]
        return self.out(self.norm(x[:, 0]))


def load(path, dev="cuda"):
    path = Path(path)
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        ck = dict(json.loads(path.with_suffix(".json").read_text()), model=load_file(path))
    else:
        ck = torch.load(path, map_location="cpu", weights_only=False)
    model = LightSeq(**ck["config"])
    model.load_state_dict(ck["model"])
    model.vocab = Vocab.from_json(ck["vocab"])
    return model.to(dev).eval()


def song_frames(logmel, fps, beat_times, offset, n_beats):
    from .data import beat_mel
    from .lightmodel import song_inputs
    mel = beat_mel(logmel, fps, np.asarray(beat_times, float), offset, n=n_beats * AUDIO_PER_BEAT, per_beat=AUDIO_PER_BEAT)
    mel = np.asarray(mel, np.float32)
    return np.concatenate([mel, song_inputs(mel, len(mel))], 1)


def note_bits(beats, colors, n_beats):
    from .lightmodel import note_ticks
    return note_ticks(beats, colors, n_beats * TPB).reshape(n_beats, TPB)


@torch.no_grad()
def generate(model, frames, notes, sec, busy=float("nan"), fade=float("nan"), temp=1.0, top_p=0.95, seed=0, dev="cuda"):
    vocab = model.vocab
    T = len(sec)
    g = torch.Generator(device=dev).manual_seed(seed)
    f = torch.from_numpy(np.asarray(frames, np.float32))[None].to(dev)
    nt = torch.from_numpy(np.asarray(notes, np.int64))[None].to(dev)
    s = torch.from_numpy(np.asarray(sec, np.float32))[None].to(dev)
    with torch.autocast("cuda", torch.bfloat16, enabled=str(dev).startswith("cuda")):
        mem, _ = model.encode(f, nt, s, torch.tensor([busy], device=dev), torch.tensor([fade], device=dev))
    mem_cs = rope_at(torch.arange(T, device=dev)[None], mem.shape[-1] // model.heads)
    caches = [dict() for _ in model.dec]
    ranges = [vocab.kind_range(k) for k in range(3)]
    empty = vocab.index[WHAT].get(tuple((showlang.NONE, 0) for _ in range(5)))
    out, prev = [], vocab.bos

    def pick(kind, b, t):
        nonlocal prev
        note = int(notes[b, t]) if kind == WHAT else 0
        args = [torch.tensor([v], device=dev) for v in (prev, kind, b, t, note)]
        with torch.autocast("cuda", torch.bfloat16, enabled=str(dev).startswith("cuda")):
            logits = model.step(mem, mem_cs, *args, caches)[0].float()
        lo, hi = ranges[kind]
        z = torch.full_like(logits, -float("inf"))
        z[lo:hi] = logits[lo:hi] / temp
        if kind == WHAT and empty is not None:
            z[ranges[WHAT][0] + empty] = -float("inf")
        p = torch.softmax(z, -1)
        if top_p < 1:
            sp, si = torch.sort(p, descending=True)
            cut = (sp.cumsum(0) - sp) > top_p
            p[si[cut]] = 0
            p /= p.sum()
        tok = int(torch.multinomial(p, 1, generator=g))
        out.append(tok)
        prev = tok
        return tok

    for b in range(T):
        w = pick(WHEN, b, 0)
        for t in vocab.when[w - ranges[WHEN][0]]:
            pick(WHAT, b, t)
        pick(MOTION, b, TPB - 1)
    return out


BUSY_FIT = (0.586, -0.026, 0.669, 0.094)
FADE_USUAL = 0.93
FADE_HAND = 0.9
NOTE_MISS, NOTE_LIT, FLAT, IDLE, SAME_LOOK, SWAP, SWAP_MAX = 4.0, 0.9, 2.0, 2.0, 2.0, 2.0, 0.7
AGAIN, AGAIN_MAX = 3.0, 0.8


def rises(m, spb=0.5):
    act, col = m["act"], m["col"]
    up, idle = np.zeros(act.shape, bool), np.zeros(act.shape, bool)
    decay = max(int(1.0 / spb * TPB), 1)
    for p in range(act.shape[1]):
        kind, at, c0 = showlang.OFF, 0, -1
        for t in np.flatnonzero(act[:, p] != showlang.NONE):
            since = t - 1 - at
            if kind in (showlang.ON, showlang.TO):
                was = 1.0
            elif kind == showlang.FLASH:
                was = 1.0 if since < TPB // 4 else 0.75
            elif kind == showlang.FADE:
                was = max(1 - since / decay, 0.0)
            else:
                was = 0.0
            kind, at = act[t, p], t
            if kind == showlang.OFF:
                idle[t, p] = was <= 0.05
            else:
                up[t, p] = was <= 0.8 or col[t, p] != c0 or kind == showlang.FLASH
                idle[t, p] = not up[t, p] and kind != showlang.FADE
                c0 = col[t, p]
    return up, idle


def hits(m, notes, spb=0.5):
    nt = np.asarray(notes).reshape(-1)[:len(m["act"])] > 0
    k = np.flatnonzero(nt)
    if not len(k):
        return 1.0, 0.0, 0.0, 0.0, 0.0, 1.0
    a, b = k[0] // TPB * TPB, (k[-1] // TPB + 1) * TPB
    act, col, nt = m["act"][a:b], m["col"][a:b], nt[a:b]
    mom = (act != showlang.NONE).any(1)
    if not mom.any():
        return 0.0, 1.0, 0.0, 1.0, 0.0, 0.0
    up, idle = rises(dict(act=act, col=col), spb)
    lit = up.any(1) & nt
    n8 = len(nt) // (8 * TPB) * 8 * TPB
    per = nt[:n8].reshape(-1, 8 * TPB).sum(1)
    flat = float((lit[:n8].reshape(-1, 8 * TPB).sum(1)[per >= 4] < 0.5 * per[per >= 4]).mean()) if (per >= 4).any() else 0.0
    left, right = showlang.PLACES.index("left"), showlang.PLACES.index("right")
    side = [(t, bool(up[t, left])) for t in np.flatnonzero(up[:, left] ^ up[:, right])]
    turn = np.zeros(len(nt) // TPB + 1, bool)
    for (t0, s0), (t1, s1) in zip(side, side[1:]):
        if s0 != s1 and t1 - t0 <= 2 * TPB:
            turn[t0 // TPB:t1 // TPB + 1] = True
    looks = np.concatenate([act, np.where(act >= showlang.ON, col, 0)], 1)[mom]
    beat = up[:len(nt) // TPB * TPB].reshape(len(nt) // TPB, -1)
    again = np.zeros(len(beat), bool)
    for k in (1, 2, 4, 8):
        again[k:] |= (beat[k:] == beat[:-k]).all(1)
    has = beat.any(1)
    return (float(lit.sum() / nt.sum()), flat, float(idle.sum() / (act != showlang.NONE).sum()),
            float(np.unique(looks, axis=0, return_counts=True)[1].max() / len(looks)), float(turn[:len(nt) // TPB].mean()),
            float((again & has).sum() / max(has.sum(), 1)))


ROUNDS, NOT_HAND = 3, 10.0
LIT_MIN, LIT_MAX, LOOKS_MIN = 1.4, 3.9, 7.0


def dull(m, notes, sec):
    has = np.flatnonzero(np.asarray(notes).reshape(len(notes), -1).any(1))
    if not len(has):
        return False
    a, b = has[0] * TPB, (has[-1] + 1) * TPB
    act = m["act"][a:b]
    looks = {(tuple(x), tuple(c)) for x, c in zip(act.tolist(), m["col"][a:b].tolist()) if any(x)}
    if len(looks) < LOOKS_MIN * (has[-1] + 1 - has[0]) / 100:
        return True
    spb = float(np.exp(np.mean(sec[has[0]:has[-1] + 1])))
    return not LIT_MIN <= float(showlang.brightness(dict(act=act), spb).sum(1).mean()) <= LIT_MAX


END_LIT, END_AT = 0.5, 0.6


def left_on(m):
    act = m["act"]
    out = []
    for p in range(act.shape[1]):
        k = np.flatnonzero(act[:, p])
        if len(k) and act[k[-1], p] in (showlang.ON, showlang.FLASH, showlang.TO):
            out.append(p)
    return out


def fade_end(m, last):
    on = left_on(m)
    if not on:
        return m
    act, col = m["act"], m["col"]
    n = len(act) // TPB
    at = max(int(round(last + END_AT * (n - last))), int(np.flatnonzero(act.any(1))[-1]) // TPB + 1)
    t = min(at, n - 1) * TPB
    for p in on:
        k = int(np.flatnonzero(act[:, p])[-1])
        act[max(t, k), p], col[max(t, k), p] = showlang.FADE, col[k, p]
    return m


COUNT_GROUPS = 8
TRIES = 4
TRIES_HAND = 8


def busy_for(notes, bpm):
    n = np.asarray(notes) > 0
    x = (1.0, math.log(max(bpm, 1.0)), math.log(n.sum() / max(len(n), 1) + 0.05), n.any(1).mean())
    return float(np.dot(BUSY_FIT, x))


@torch.no_grad()
def generate_batch(model, frames, notes, sec, busy, fade, temp=1.0, top_p=0.95, seed=0, dev="cuda", keep=1024, graph=True,
                   made=UNKNOWN, liked=0, grouped=False):
    vocab = model.vocab
    notes = np.asarray(notes)
    R, T = notes.shape[0], len(sec)
    cuda = str(dev).startswith("cuda")
    torch.manual_seed(seed)
    ac = lambda: torch.autocast("cuda", torch.bfloat16, enabled=cuda, cache_enabled=False)
    f = torch.from_numpy(np.asarray(frames, np.float32)).to(dev)[None].repeat(R, 1, 1)
    nt = torch.from_numpy(notes.astype(np.int64)).to(dev)
    s = torch.from_numpy(np.asarray(sec, np.float32)).to(dev)[None].repeat(R, 1)
    bz = torch.tensor(np.asarray(busy, np.float32), device=dev)
    fd = torch.tensor(np.asarray(fade, np.float32), device=dev)
    with ac():
        mem, _ = model.encode(f, nt, s, bz, fd, made=torch.full((R,), made, device=dev), liked=torch.full((R,), liked, device=dev))
    H = model.heads
    dh = mem.shape[-1] // H
    mem_cs = rope_at(torch.arange(T, device=dev)[None].expand(R, T), dh)
    with ac():
        cross = []
        for blk in model.dec:
            mk, mv = blk.split(blk.kv2(mem), 2)
            cross.append((_rot(mk, mem_cs), mv))
    dt = cross[0][0].dtype
    K = [torch.zeros(R, H, keep, dh, dtype=dt, device=dev) for _ in model.dec]
    V = [torch.zeros(R, H, keep, dh, dtype=dt, device=dev) for _ in model.dec]
    filled = torch.zeros(R, keep, dtype=torch.bool, device=dev)
    n_out = model.out.out_features
    bias = torch.full((3, n_out), -float("inf"), device=dev)
    for k in range(3):
        lo, hi = vocab.kind_range(k)
        bias[k, lo:hi] = 0
    empty = vocab.index[WHAT].get(tuple((showlang.NONE, 0) for _ in range(5)))
    if empty is not None:
        bias[WHAT, vocab.base[WHAT] + empty] = -float("inf")
    grp = torch.zeros(n_out, dtype=torch.long, device=dev)
    grp[vocab.base[WHEN]:vocab.base[WHEN] + len(vocab.when)] = torch.tensor([min(len(p), COUNT_GROUPS - 1) for p in vocab.when], device=dev)
    gids = torch.arange(COUNT_GROUPS, device=dev)
    tp = torch.tensor([float(x) for x in (top_p if isinstance(top_p, (tuple, list)) else (top_p,) * 3)], device=dev)
    wl = torch.tensor([len(p) for p in vocab.when], device=dev)
    wt = torch.tensor([list(p) + [0] * (TPB - len(p)) for p in vocab.when], device=dev)
    maxlen = T * (TPB + 2)
    out = torch.zeros(R, maxlen + 1, dtype=torch.long, device=dev)
    ar = torch.arange(R, device=dev)
    yes = torch.ones(R, dtype=torch.bool, device=dev)
    st = dict(prev=torch.full((R,), vocab.bos, device=dev), kind=torch.full((R,), WHEN, device=dev),
              beat=torch.zeros(R, dtype=torch.long, device=dev), tick=torch.zeros(R, dtype=torch.long, device=dev),
              pat=torch.zeros(R, dtype=torch.long, device=dev), pidx=torch.zeros(R, dtype=torch.long, device=dev),
              n=torch.zeros(R, dtype=torch.long, device=dev), done=torch.zeros(R, dtype=torch.bool, device=dev))

    def step():
        kind, beat, tick, n, done = st["kind"], st["beat"], st["tick"], st["n"], st["done"]
        b = beat.clamp(max=T - 1)
        note = torch.where(kind == WHAT, nt[ar, b, tick], 0)
        slot = n % keep
        filled[ar, slot] = yes
        m = filled[:, None, None, :]
        with ac():
            x = model.embed(mem, st["prev"][:, None], kind[:, None], b[:, None], tick[:, None], note[:, None])
            cs = rope_at((b.float() + tick.float() / TPB)[:, None], dh)
            for li, blk in enumerate(model.dec):
                q, k, v = blk.split(blk.qkv(blk.n1(x)), 3)
                q, k = _rot(q, cs), _rot(k, cs)
                K[li][ar, :, slot] = k[:, :, 0].to(dt)
                V[li][ar, :, slot] = v[:, :, 0].to(dt)
                x = x + blk.o(blk.merge(F.scaled_dot_product_attention(q.to(dt), K[li], V[li], m)))
                q2 = _rot(blk.split(blk.q2(blk.n2(x)), 1)[0], cs)
                mk, mv = cross[li]
                x = x + blk.o2(blk.merge(F.scaled_dot_product_attention(q2.to(dt), mk, mv)))
                x = x + blk.ff(blk.n3(x))
            logits = model.out(model.norm(x[:, 0])).float()
        p = torch.softmax(logits / temp + bias[kind], -1)
        sp, si = torch.sort(p, -1, descending=True)
        if grouped:
            gs = grp[si]
            oh = gs[:, None, :] == gids[None, :, None]
            cg = torch.where(oh, sp[:, None, :], 0.0).cumsum(-1)
            own = cg.gather(1, gs[:, None, :])[:, 0]
            mass = cg[:, :, -1].gather(1, gs)
            kept = sp.masked_fill(own - sp > tp[kind][:, None] * mass, 0)
            left = torch.where(oh, kept[:, None, :], 0.0).sum(-1).gather(1, gs)
            sp = kept * mass / left.clamp(min=1e-12)
        else:
            sp = sp.masked_fill(sp.cumsum(-1) - sp > tp[kind][:, None], 0)
        c = sp.cumsum(-1)
        u = torch.rand(R, 1, device=dev) * c[:, -1:]
        tok = si.gather(1, (c < u).sum(-1, keepdim=True).clamp(max=n_out - 1))[:, 0]
        live = ~done
        out[ar, torch.where(live, n, maxlen)] = tok
        is_when, is_what, is_mot = kind == WHEN, kind == WHAT, kind == MOTION
        pat = torch.where(is_when, tok - vocab.base[WHEN], st["pat"]).clamp(0, len(vocab.when) - 1)
        pidx = torch.where(is_when, 0, torch.where(is_what, st["pidx"] + 1, st["pidx"]))
        more = pidx < wl[pat]
        nkind = torch.where(is_mot, WHEN, torch.where(more, WHAT, MOTION))
        ntick = torch.where(is_mot, 0, torch.where(more, wt[pat, pidx.clamp(max=TPB - 1)], TPB - 1))
        nbeat = beat + is_mot.long()
        st["prev"].copy_(torch.where(live, tok, st["prev"]))
        st["pat"].copy_(pat); st["pidx"].copy_(pidx)
        st["kind"].copy_(torch.where(live, nkind, kind)); st["tick"].copy_(torch.where(live, ntick, tick))
        st["beat"].copy_(torch.where(live, nbeat, beat)); st["n"].copy_(n + live.long())
        st["done"].copy_(done | (nbeat >= T))

    run = step
    if cuda and graph:
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                step()
        torch.cuda.current_stream().wait_stream(side)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            step()
        run = g.replay
    steps = 3 if cuda and graph else 0
    while steps < maxlen + 3:
        for _ in range(64):
            run()
        steps += 64
        if bool(st["done"].all()):
            break
    n = st["n"].tolist()
    o = out.cpu().numpy()
    return [o[r, :n[r]].tolist() for r in range(R)]


PIECE_EDGE = 64


def pieces(n_beats, size=MAX_BEATS, edge=PIECE_EDGE):
    if n_beats <= size:
        return [(0, 0, n_beats)]
    out, start, keep = [], 0, 0
    while True:
        end = start + size
        if end >= n_beats:
            out.append((start, keep, n_beats))
            return out
        cut = end - edge
        out.append((start, keep, cut))
        keep = cut
        start = (cut - edge) // 8 * 8


def song_rows(logmel, fps, beat_times, offset, n_beats):
    from .data import beat_to_sec
    bt = np.asarray(beat_times, float)
    frames = song_frames(logmel, fps, bt, offset, n_beats)
    sec = np.log(np.maximum(np.diff(beat_to_sec(offset + np.arange(n_beats + 1, dtype=np.float64), bt)), 1e-2)).astype(np.float32)
    return frames, sec


GUIDE_MIN, GUIDE_STARS = 0.70, 8.0


def thin(notes, offset):
    b = np.unique(np.floor(np.asarray(notes["beat"], np.float64) - offset + 1e-6).astype(int))
    return len(b) / max(int(b[-1] - b[0]) + 1, 1) < GUIDE_MIN if len(b) else False


def guide_notes(beats, first, last):
    b = np.unique(np.asarray(beats, np.float64))
    b = b[(b >= first) & (b <= last)]
    out = np.zeros(len(b), [("beat", "f8"), ("color", "i1")])
    out["beat"], out["color"] = b, np.arange(len(b)) % 2
    return out


OUTRO_MIN, OUTRO_FADE, OUTRO_PLACES = 4.0, 2.0, 2


def hold_outro(events, last, to, spb):
    if (to - last) * spb < OUTRO_MIN:
        return events
    lights = set(showlang.BASIC_PLACE)
    ev = sorted((e for e in events if not (e[0] > last + 1e-6 and e[1] in lights and e[2] == 0)), key=lambda e: e[0])
    state = {}
    for k, e in enumerate(ev):
        if e[1] in lights:
            state[e[1]] = k
    act = {p: showlang.basic_action(int(ev[k][2]))[0] for p, k in state.items()}
    held = [p for p, a in act.items() if a in (showlang.ON, showlang.FLASH, showlang.TO)]
    if not held:
        fades = [p for p, a in act.items() if a == showlang.FADE]
        if fades:
            at = max(ev[state[p]][0] for p in fades)
            for p in sorted((p for p in fades if ev[state[p]][0] >= at - 1.0), key=lambda p: (p != 0, p != 4, p))[:OUTRO_PLACES]:
                k = state[p]
                colour = showlang.basic_action(int(ev[k][2]))[1]
                ev[k] = (ev[k][0], p, showlang.basic_value(showlang.ON, colour))
                held.append(p)
    out_at = round(max(to - OUTRO_FADE / spb, last + 1.0), 4)
    for p in held:
        colour = showlang.basic_action(int(ev[state[p]][2]))[1]
        ev.append((out_at, p, showlang.basic_value(showlang.FADE, colour)))
    return sorted(ev, key=lambda e: e[0])


def lightshows(model, logmel, fps, beat_times, offset, notes_list, bpm, seed=0, dev="cuda", tries=None, temp=1.0,
               top_p=0.95, made=HAND, liked=LIKED, busy=True, fade="usual", end=None, song_end=None, grouped=True):
    ok = [i for i, nts in enumerate(notes_list) if len(nts)]
    shows = [[] for _ in notes_list]
    if not ok:
        return shows, []
    tags = bool(model.config.get("tags"))
    if fade == "usual":
        fade = FADE_HAND if tags else FADE_USUAL
    tries = tries or (TRIES_HAND if tags else TRIES)
    n_beats = max(int(np.ceil(notes_list[i]["beat"].max() - offset)) + 4 for i in ok)
    whole = bool(model.config.get("whole")) and song_end is not None
    if whole:
        end = song_end
    if end is not None:
        n_beats = max(n_beats, min(int(np.ceil(end - offset)), len(beat_times) - 2))
    frames, sec = song_rows(logmel, fps, beat_times, offset, n_beats)
    nb = [note_bits(notes_list[i]["beat"] - offset, notes_list[i]["color"], n_beats) for i in ok]
    want = [busy_for(x, bpm) for x in nb]
    kinds = []
    for k, (start, keep_from, keep_to) in enumerate(pieces(n_beats)):
        stop = min(start + MAX_BEATS, n_beats)
        rows = [x[start:stop] for x in nb for _ in range(tries)]
        best = [None] * len(ok)
        for r in range(ROUNDS if tags and made == HAND else 1):
            toks = generate_batch(model, frames[start * AUDIO_PER_BEAT:stop * AUDIO_PER_BEAT], np.stack(rows), sec[start:stop],
                                  [w if busy else float("nan") for w in want for _ in range(tries)],
                                  [float("nan") if fade is None else fade] * len(rows), temp=temp, top_p=top_p,
                                  seed=seed + k + 1000 * r, dev=dev, made=made, liked=liked, grouped=grouped)
            for j in range(len(ok)):
                for t in range(tries):
                    m = decode_show(toks[j * tries + t], model.vocab, stop - start)
                    items = show_items(m, stop - start)
                    bz, fd = show_kind(items)
                    hand = tags and made == HAND
                    fd = 0.0 if np.isnan(fd) else fd
                    d = abs(bz - want[j]) + (0.0 if fade is None else 0.5 * abs(fd - fade))
                    if hand:
                        lit, flat, idle, same, swap, again = hits(m, nb[j][start:stop], float(np.exp(np.mean(sec[start:stop]))))
                        d += NOTE_MISS * max(NOTE_LIT - lit, 0.0) + FLAT * flat + IDLE * idle + SAME_LOOK * same - (SWAP * min(swap, SWAP_MAX) + AGAIN * min(again, AGAIN_MAX)) * min(lit / NOTE_LIT, 1.0) ** 2
                    if hand and (made_by(items, nb[j][start:stop]) != HAND or dull(m, nb[j][start:stop], sec[start:stop])):
                        d += NOT_HAND
                    if whole and stop == n_beats and left_on(m):
                        d += END_LIT
                    if best[j] is None or d < best[j][0]:
                        best[j] = (d, m, (bz, fd, want[j]))
            if all(b[0] < NOT_HAND - SWAP - AGAIN for b in best):
                break
        for j, i in enumerate(ok):
            if whole and stop == n_beats:
                fade_end(best[j][1], int(np.flatnonzero(nb[j].any(1))[-1]) - start)
            lo, hi = offset + keep_from - 1e-6, offset + keep_to - 1e-6
            shows[i] += [e for e in showlang.to_basic(best[j][1], offset + start) if lo <= e[0] < hi or (k == 0 and e[0] < hi)]
            if k == 0:
                kinds.append(best[j][2])
    if song_end is not None and not whole:
        spb = float(np.exp(np.mean(sec[-8:])))
        for i in ok:
            shows[i] = hold_outro(shows[i], float(notes_list[i]["beat"].max()), song_end, spb)
    return shows, kinds


def lightshow(model, logmel, fps, beat_times, offset, notes, seed=0, dev="cuda", temp=1.0, top_p=0.95, busy=None, fade=None):
    if not len(notes):
        return []
    n_beats = int(np.ceil(notes["beat"].max() - offset)) + 4
    frames, sec = song_rows(logmel, fps, beat_times, offset, n_beats)
    nb = note_bits(notes["beat"] - offset, notes["color"], n_beats)
    out = []
    for k, (start, keep_from, keep_to) in enumerate(pieces(n_beats)):
        stop = min(start + MAX_BEATS, n_beats)
        toks = generate(model, frames[start * AUDIO_PER_BEAT:stop * AUDIO_PER_BEAT], nb[start:stop], sec[start:stop],
                        float("nan") if busy is None else busy, float("nan") if fade is None else fade,
                        temp=temp, top_p=top_p, seed=seed + k, dev=dev)
        m = decode_show(toks, model.vocab, stop - start)
        lo, hi = offset + keep_from - 1e-6, offset + keep_to - 1e-6
        out += [e for e in showlang.to_basic(m, offset + start) if lo <= e[0] < hi or (k == 0 and e[0] < hi)]
    return out
