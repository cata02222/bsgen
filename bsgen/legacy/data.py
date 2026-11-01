import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import torch

from . import tokens
from .tokens import CONTEXT, COND_FIELDS, MAX_MAPPERS, OBJ_ARCS, OBJ_BOMBS, OBJ_WALLS, TICKS, VOCAB, WINDOW

FRAMES_PER_BEAT = 12
AHEAD = 16
N_FRAMES = (CONTEXT + WINDOW + AHEAD) * FRAMES_PER_BEAT
OVERVIEW_BARS = 256
OVERVIEW_PER_BEAT = 4
MAX_TOKENS = 1536
COND_DROPOUT = dict(stars=0.15, style=0.3, mapper=0.5, njs=0.5, nps=0.5, objects=0.2, **{"pass": 0.5}, acc=0.5, tech=0.5)
ALL_DROPOUT = 0.1
WALL_LOOKBACK_DROPOUT = 0.2
MIN_ON_GRID = 0.6
CACHE_VERSION = "4"
EVENT_VERSION = "5"
LABELLED = "labelled"
AUDIO_VERSION = "1"


def beat_to_sec(beat, beat_times):
    k = np.arange(len(beat_times))
    s = np.interp(beat, k, beat_times)
    lo, hi = beat < 0, beat > k[-1]
    s[lo] = beat_times[0] + beat[lo] * (beat_times[1] - beat_times[0])
    s[hi] = beat_times[-1] + (beat[hi] - k[-1]) * (beat_times[-1] - beat_times[-2])
    return s


def sec_to_beat(sec, beat_times):
    k = np.arange(len(beat_times))
    b = np.interp(sec, beat_times, k)
    lo, hi = sec < beat_times[0], sec > beat_times[-1]
    b[lo] = (sec[lo] - beat_times[0]) / (beat_times[1] - beat_times[0])
    b[hi] = k[-1] + (sec[hi] - beat_times[-1]) / (beat_times[-1] - beat_times[-2])
    return b


def beat_mel(logmel, fps, beat_times, first_beat, n=N_FRAMES, per_beat=FRAMES_PER_BEAT):
    sec = beat_to_sec(first_beat + np.arange(n) / per_beat, beat_times)
    x = sec * fps
    i = np.floor(x).astype(int)
    a = (x - i)[:, None]
    ok = (i >= 0) & (i + 1 < len(logmel))
    out = np.full((n, logmel.shape[1]), -80.0, np.float32)
    j = i[ok]
    out[ok] = logmel[j].astype(np.float32) * (1 - a[ok]) + logmel[j + 1].astype(np.float32) * a[ok]
    return out / 40.0 + 1.0


def song_overview(logmel, fps, beat_times, offset):
    n_beats = int(np.ceil(sec_to_beat(np.array([len(logmel) / fps]), beat_times)[0]))
    bar = max(4, int(np.ceil(n_beats / OVERVIEW_BARS)))
    n_bars = int(np.ceil(n_beats / bar))
    m = beat_mel(logmel, fps, beat_times, offset, n=n_bars * bar * OVERVIEW_PER_BEAT, per_beat=OVERVIEW_PER_BEAT)
    m = m.reshape(n_bars, bar * OVERVIEW_PER_BEAT, -1)
    out = np.full((OVERVIEW_BARS, 2 * m.shape[2]), -1.0, np.float32)
    out[:n_bars] = np.concatenate([m.mean(1), m.max(1)], 1)
    return out, bar


def overview_rel(bar, first_beat):
    here = (first_beat + CONTEXT) // bar
    return np.clip(np.arange(OVERVIEW_BARS) - here, 1 - OVERVIEW_BARS, OVERVIEW_BARS - 1) + OVERVIEW_BARS - 1


def loss_mask(tk, t0):
    mask = np.zeros(len(tk), np.float32)
    boundary = t0 + CONTEXT * TICKS
    tick = None
    for j, k in enumerate(tk):
        if VOCAB.time0 <= k < VOCAB.swing0:
            tick = t0 + k - VOCAB.time0
        if k == VOCAB.EOS or (tick is not None and tick >= boundary and k >= VOCAB.time0):
            mask[j] = 1
    return mask


def mapper_key(r):
    return r["mapper"].strip().lower()


def mapper_list(rows, min_maps=20):
    count = {}
    for lb, r in {r["lb"]: r for r in rows}.items():
        count[mapper_key(r)] = count.get(mapper_key(r), 0) + 1
    return sorted((m for m, n in count.items() if n >= min_maps), key=lambda m: -count[m])[:MAX_MAPPERS]


def row_cond(r, mappers):
    objects = OBJ_WALLS * (r.get("walls", 0) > 0) + OBJ_BOMBS * (r.get("bombs", 0) > 0) + OBJ_ARCS * (r.get("arcs", 0) > 0)
    m = mapper_key(r)
    return dict(stars=r["stars"], style=r.get("style"), mapper=mappers.index(m) if m in mappers else MAX_MAPPERS,
                njs=r.get("njs") or None, nps=r.get("nps"), objects=int(objects), **{"pass": r.get("pass_rating")},
                acc=r.get("acc_rating"), tech=r.get("tech_rating"))


class Windows(torch.utils.data.Dataset):

    def __init__(self, root: Path, rows: list[dict], lookback_dropout=0.1, train=True, swings=True, mappers=None,
                 controls=True, labelled_stars=True):
        self.root, self.train = Path(root), train
        self.swings = swings
        self.controls = controls
        self.lookback_dropout = lookback_dropout
        self.mappers = mappers or []
        self.cond = [row_cond(r, self.mappers) for r in rows]
        if not labelled_stars:
            for r, c in zip(rows, self.cond):
                if r.get("source") == LABELLED:
                    c["stars"] = None
        self.hashes = sorted({r["hash"] for r in rows})
        files = "\n".join(r["file"] for r in rows)
        key = hashlib.sha1((files + CACHE_VERSION + EVENT_VERSION).encode()).hexdigest()[:16]
        cache = self.root / "cache" / f"windows_{key}.npz"
        if cache.exists():
            z = np.load(cache)
            for k in ("ev", "ev_start", "bt", "bt_start", "offset", "stars", "song_hash", "items"):
                setattr(self, k, z[k])
        else:
            self._pack_events(rows, cache)
        lbs = list(dict.fromkeys(r["lb"] for r in rows))
        pos = {lb: j for j, lb in enumerate(lbs)}
        self.ov_index = np.array([pos[r["lb"]] for r in rows], np.int32)
        first = np.unique(self.ov_index, return_index=True)[1]
        akey = hashlib.sha1((files + CACHE_VERSION + AUDIO_VERSION).encode()).hexdigest()[:16]
        self._pack_audio(self.root / "cache" / f"audio_{akey}", first)
        self._au = self._ov = None

    def _pack_events(self, rows, cache):
        hash_idx = {h: i for i, h in enumerate(self.hashes)}
        evs, bts, on_grid = [], [], []
        offset, stars, song_hash = (np.zeros(len(rows), d) for d in ("f8", "f4", "i4"))
        for i, r in enumerate(rows):
            mp = np.load(self.root / "maps" / f"{r['lb']}.npz")
            rp = np.load(self.root.parent / r["file"])
            offset[i] = float(mp["grid_offset"])
            evs.append(tokens.events(mp["notes"], rp["owner"], rp["swings"], rp["swing_beat"], offset[i],
                                     mp["bombs"], mp["walls"], mp["arcs"]))
            bts.append(mp["beat_times"])
            stars[i], song_hash[i] = r["stars"], hash_idx[r["hash"]]
            nb = mp["notes"]["beat"] - offset[i]
            on_grid.append((np.round(nb * TICKS).astype(np.int64), np.abs(nb * TICKS - np.round(nb * TICKS)) < 0.1))
        self.ev, self.ev_start = np.concatenate(evs), np.cumsum([0] + [len(e) for e in evs])
        self.bt, self.bt_start = np.concatenate(bts), np.cumsum([0] + [len(t) for t in bts])
        self.offset, self.stars, self.song_hash = offset, stars, song_hash
        items, skipped = [], 0
        for si, e in enumerate(evs):
            first = int(e["tick"].min()) // TICKS - CONTEXT
            last = int(e["tick"].max()) // TICKS
            nt, ok = on_grid[si]
            for b in range(first, last + 1, WINDOW):
                sel = (nt >= (b + CONTEXT) * TICKS) & (nt < (b + CONTEXT + 2 * WINDOW) * TICKS)
                if sel.sum() >= 8 and ok[sel].mean() < MIN_ON_GRID:
                    skipped += 1
                    continue
                items.append((si, b))
        print(f"{len(items)} windows, {skipped} skipped for notes off the beat grid")
        self.items = np.array(items, np.int32)
        cache.parent.mkdir(exist_ok=True)
        tmp = cache.with_name(cache.stem + f".{os.getpid()}.tmp.npz")
        np.savez(tmp, ev=self.ev, ev_start=self.ev_start, bt=self.bt, bt_start=self.bt_start, offset=self.offset,
                 stars=self.stars, song_hash=self.song_hash, items=self.items)
        tmp.replace(cache)

    def _pack_audio(self, base, first):
        self.au_path, self.ov_path, meta = base.with_suffix(".f16"), base.with_suffix(".ov.npy"), base.with_suffix(".npz")
        if not meta.exists():
            tmp = base.with_name(base.name + f".{os.getpid()}.tmp")
            au_start, fps = [0], []
            with open(tmp.with_suffix(".f16"), "wb") as f:
                for h in self.hashes:
                    z = np.load(self.root / "audio" / f"{h}.npz")
                    m = np.ascontiguousarray(z["logmel"], np.float16)
                    f.write(m.tobytes())
                    au_start.append(au_start[-1] + len(m))
                    fps.append(float(z["fps"]))
            shape = np.array([au_start[-1], m.shape[1]])
            au = np.memmap(tmp.with_suffix(".f16"), np.float16, "r", shape=tuple(shape))
            ov = np.lib.format.open_memmap(tmp.with_suffix(".ov.npy"), "w+", np.float32,
                                           (len(first), OVERVIEW_BARS, 2 * int(shape[1])))
            bar = np.zeros(len(first), np.int32)
            for j, si in enumerate(first):
                s = self.song_hash[si]
                bt = self.bt[self.bt_start[si]:self.bt_start[si + 1]]
                ov[j], bar[j] = song_overview(au[au_start[s]:au_start[s + 1]], fps[s], bt, float(self.offset[si]))
                if j % 1000 == 999:
                    ov.flush()
            ov.flush()
            del au, ov
            tmp.with_suffix(".f16").replace(self.au_path)
            tmp.with_suffix(".ov.npy").replace(self.ov_path)
            np.savez(tmp.with_suffix(".npz"), au_start=np.array(au_start), fps=np.array(fps), shape=shape, bar=bar)
            tmp.with_suffix(".npz").replace(meta)
        z = np.load(meta)
        self.au_start, self.fps, self.au_shape, self.bar = z["au_start"], z["fps"], tuple(z["shape"]), z["bar"]

    def __len__(self):
        return len(self.items)

    def lengths(self):
        out = np.zeros(len(self.items), np.int32)
        order = np.argsort(self.items[:, 0], kind="stable")
        bounds = np.searchsorted(self.items[order, 0], np.arange(len(self.offset) + 1))
        for si in range(len(self.offset)):
            sel = order[bounds[si]:bounds[si + 1]]
            ticks = self.ev["tick"][self.ev_start[si]:self.ev_start[si + 1]]
            t0 = self.items[sel, 1].astype(np.int64) * TICKS
            out[sel] = np.searchsorted(ticks, t0 + (CONTEXT + WINDOW) * TICKS) - np.searchsorted(ticks, t0)
        return out

    def mel(self, si):
        if self._au is None:
            self._au = np.memmap(self.au_path, np.float16, "r", shape=self.au_shape)
            self._ov = np.load(self.ov_path, mmap_mode="r")
        s = self.song_hash[si]
        return self._au[self.au_start[s]:self.au_start[s + 1]], float(self.fps[s])

    def __getitem__(self, i):
        si, b = (int(x) for x in self.items[i])
        ev = self.ev[self.ev_start[si]:self.ev_start[si + 1]]
        beat_times = self.bt[self.bt_start[si]:self.bt_start[si + 1]]
        offset = float(self.offset[si])
        if self.train:
            b += np.random.randint(0, WINDOW)
        t0 = b * TICKS
        cond = dict(self.cond[si])
        if self.train:
            hide_all = np.random.rand() < ALL_DROPOUT
            for f in COND_FIELDS:
                if (hide_all and f != "objects") or np.random.rand() < COND_DROPOUT[f]:
                    cond[f] = None
        if not self.controls:
            cond = cond["stars"]
            ev = ev[(ev["kind"] == tokens.SWING) | (ev["kind"] == tokens.NOTE)]
        if self.train and np.random.rand() < self.lookback_dropout:
            ev = ev[(ev["tick"] < t0) | (ev["tick"] >= t0 + CONTEXT * TICKS)]
        elif self.train and self.controls and np.random.rand() < WALL_LOOKBACK_DROPOUT:
            lookback = (ev["tick"] >= t0) & (ev["tick"] < t0 + CONTEXT * TICKS)
            ev = ev[~lookback | (ev["kind"] != tokens.WALL)]
        if not self.swings:
            ev = ev[ev["kind"] != tokens.SWING]
        tk = tokens.encode(ev, t0, cond)[:MAX_TOKENS]
        logmel, fps = self.mel(si)
        mel = beat_mel(logmel, fps, beat_times, b + offset)
        j = self.ov_index[si]
        ov, bar = np.array(self._ov[j]), int(self.bar[j])
        return (torch.from_numpy(mel), torch.from_numpy(ov), torch.from_numpy(overview_rel(bar, b)),
                torch.tensor(tk), torch.from_numpy(loss_mask(tk, t0)))


class LengthBatches(torch.utils.data.Sampler):

    def __init__(self, lengths, batch, pool=64, weights=None):
        self.lengths, self.batch, self.pool = np.asarray(lengths), batch, pool
        self.p = None if weights is None else np.asarray(weights, np.float64) / np.sum(weights)

    def __len__(self):
        return len(self.lengths) // self.batch

    def __iter__(self):
        n = len(self.lengths)
        idx = np.random.permutation(n) if self.p is None else np.random.choice(n, n, p=self.p)
        batches = []
        for s in range(0, len(idx), self.batch * self.pool):
            chunk = idx[s:s + self.batch * self.pool]
            chunk = chunk[np.argsort(self.lengths[chunk], kind="stable")]
            batches += [chunk[i:i + self.batch] for i in range(0, len(chunk) - self.batch + 1, self.batch)]
        for k in np.random.permutation(len(batches)):
            yield batches[k].tolist()


def collate(batch):
    audio = tuple(torch.stack([b[i] for b in batch]) for i in range(3))
    n = max(len(b[3]) for b in batch)
    tk = torch.full((len(batch), n), VOCAB.PAD)
    mask = torch.zeros(len(batch), n)
    for j, (*_, t, m) in enumerate(batch):
        tk[j, :len(t)], mask[j, :len(m)] = t, m
    return audio, tk, mask


def song_key(r):
    t = re.sub(r"[^a-z0-9]", "", r["song"].lower())
    return t if len(t) > 3 else r["hash"]


def split(index_path: Path, val_share=0.05, seed=0, per_map=3):
    rows = [json.loads(line) for line in open(index_path)]
    rows = [r for r in rows if not r["flags"]]
    audit = Path(index_path).parent / "audit.json"
    if audit.exists():
        shifted = set(json.loads(audit.read_text()).get("shifted", []))
        rows = [r for r in rows if r["lb"] not in shifted]
    count, kept = {}, []
    for r in rows:
        count[r["lb"]] = count.get(r["lb"], 0) + 1
        if count[r["lb"]] <= per_map:
            kept.append(r)
    songs = sorted({song_key(r) for r in kept if r.get("source") != LABELLED})
    rng = np.random.default_rng(seed)
    val = set(rng.choice(songs, max(1, int(len(songs) * val_share)), replace=False))
    return ([r for r in kept if song_key(r) not in val],
            [r for r in kept if song_key(r) in val and r.get("source") != LABELLED])


def star_weights(stars, power=0.5, step=1.0):
    band = np.floor(np.asarray(stars) / step).astype(int)
    _, inv, count = np.unique(band, return_inverse=True, return_counts=True)
    return count[inv].astype(np.float64) ** -power
