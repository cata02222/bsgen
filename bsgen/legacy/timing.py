from pathlib import Path

import numpy as np
import torch
from torch import nn

CHUNK = 2048
BPM_RANGE = (60.0, 400.0)
SUBDIVISIONS = tuple(sorted({k / n for n in range(2, 9) for k in range(1, n)}))


def scale(logmel):
    return logmel.astype(np.float32) / 40.0 + 1.0


class BeatNet(nn.Module):
    def __init__(self, n_mels=80, d=256, layers=4, heads=4, chunk=CHUNK, dropout=0.1):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(n_mels, d, 5, padding=2), nn.GELU(),
                                  nn.Conv1d(d, d, 4, stride=2, padding=1), nn.GELU())
        self.pos = nn.Parameter(torch.randn(chunk // 2, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, "gelu", batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, 2)

    def forward(self, mel):
        x = self.stem(mel.transpose(1, 2)).transpose(1, 2)
        x = self.enc(x + self.pos[: x.shape[1]])
        return self.out(self.norm(x)).flatten(1)


@torch.no_grad()
def activation(model, logmel, dev="cuda", batch=16):
    x = scale(logmel)
    n, hop = len(x), CHUNK // 2
    starts = list(range(0, max(n - CHUNK, 0) + 1, hop))
    if starts[-1] + CHUNK < n:
        starts.append(n - CHUNK)
    pad = max(CHUNK - n, 0)
    x = np.pad(x, ((0, pad), (0, 0)), constant_values=-1.0)
    w = np.hanning(CHUNK + 2)[1:-1].astype(np.float32)
    acc, wsum = np.zeros(len(x), np.float32), np.zeros(len(x), np.float32)
    for i in range(0, len(starts), batch):
        s = starts[i:i + batch]
        mel = torch.from_numpy(np.stack([x[a:a + CHUNK] for a in s])).to(dev)
        p = torch.sigmoid(model(mel).float()).cpu().numpy()
        for a, pa in zip(s, p):
            acc[a:a + CHUNK] += pa * w
            wsum[a:a + CHUNK] += w
    return (acc / np.maximum(wsum, 1e-6))[:n]


def _score(act, period, lo, hi, bins=None):
    bins = bins or max(8, int(round(period * 4)))
    t = np.arange(lo, hi)
    k = ((t % period) / period * bins).astype(int) % bins
    prof = np.bincount(k, act[lo:hi], bins) / np.maximum(np.bincount(k, None, bins), 1)
    r = max(1, int(round(bins / period)))
    prof = np.convolve(np.concatenate([prof[-r:], prof, prof[:r]]), np.ones(2 * r + 1) / (2 * r + 1), "valid")
    off = np.max([np.roll(prof, -int(round(f * bins))) for f in SUBDIVISIONS], 0)
    val = prof - off
    j = int(np.argmax(val))
    return float(val[j]), (j + 0.5) / bins * period


def _refine(act, period, phase, lo, hi):
    for _ in range(3):
        k = np.arange(np.ceil((lo - phase) / period), np.floor((hi - 1 - phase) / period) + 1)
        g = phase + k * period
        r = max(1, int(period / 4))
        idx, wts = [], []
        for c in g:
            a, b = max(int(c) - r, 0), min(int(c) + r + 1, len(act))
            seg = act[a:b]
            if len(seg) < 3:
                idx.append(c), wts.append(0.0)
                continue
            j = int(np.argmax(seg))
            if 0 < j < len(seg) - 1:
                d = seg[j - 1] - 2 * seg[j] + seg[j + 1]
                off = 0.5 * (seg[j - 1] - seg[j + 1]) / d if d < 0 else 0.0
            else:
                off = 0.0
            idx.append(a + j + off), wts.append(float(seg[j]))
        idx, wts = np.array(idx), np.array(wts) ** 2
        if wts.sum() <= 0 or len(k) < 4:
            break
        A = np.stack([k, np.ones_like(k)], 1) * np.sqrt(wts)[:, None]
        sol = np.linalg.lstsq(A, idx * np.sqrt(wts), rcond=None)[0]
        period, phase = float(sol[0]), float(sol[1])
    return period, phase


def fit_grid(act, fps, span=None, seg_sec=40.0, steps=401):
    lo, hi = span or (0, len(act))
    seg = int(seg_sec * fps)
    mid = (lo + hi) // 2
    slo, shi = max(lo, mid - seg // 2), min(hi, mid + seg // 2)
    min_p, max_p = 60 * fps / BPM_RANGE[1], 60 * fps / BPM_RANGE[0]
    x = act[slo:shi] - act[slo:shi].mean()
    f = np.fft.rfft(x, 2 * len(x))
    ac = np.fft.irfft(f * np.conj(f))[: int(max_p) + 2]
    lags = np.arange(max(int(min_p), 1), int(max_p) + 1)
    peaks = [int(lag) for lag in lags[1:-1] if ac[lag] >= ac[lag - 1] and ac[lag] >= ac[lag + 1]]
    peaks = sorted(peaks, key=lambda lag: -ac[lag])[:6]
    best = (-np.inf, None, None)
    for p0 in peaks:
        for rel in np.linspace(-0.04, 0.04, steps):
            s, ph = _score(act, p0 * (1 + rel), slo, shi)
            if s > best[0]:
                best = (s, p0 * (1 + rel), ph)
    s, period, phase = best
    period, phase = _refine(act, period, phase, slo, shi)
    a, b = slo, shi
    while a > lo or b < hi:
        w = b - a
        a, b = max(lo, a - w // 2), min(hi, b + w // 2)
        period, phase = _refine(act, period, phase, a, b)
    phase = phase % period
    return 60 * fps / period, phase / fps, s


OCTAVES = (0.5, 1.0, 2.0)


def _marks(act, period, phase, lo, hi):
    k = np.arange(np.ceil((lo - phase) / period), np.floor((hi - 2 - phase) / period) + 1)
    out = []
    for f in (0.0, 1 / 4, 1 / 3, 1 / 2, 2 / 3, 3 / 4):
        g = np.round(phase + (k + f) * period).astype(int)
        g = g[(g >= 1) & (g < len(act) - 1)]
        out.append(float(np.max([act[g - 1], act[g], act[g + 1]], 0).mean()) if len(g) else 0.0)
    return out


def octave_features(act, fps, bpm, first_sec, span=None):
    lo, hi = span or (0, len(act))
    period, phase = 60 * fps / bpm, first_sec * fps
    feats, cands = [], []
    for m in OCTAVES:
        p = period / m
        phases = [phase + j * period for j in range(max(1, int(round(1 / m))))]
        best = max(phases, key=lambda ph: _marks(act, p, ph, lo, hi)[0])
        mk = _marks(act, p, best, lo, hi)
        feats += [np.log2(bpm * m)] + mk + [mk[0] - max(mk[1:])]
        cands.append((bpm * m, (best % p) / fps))
    return np.array(feats, np.float32), cands


def beat_grid(bpm, first_sec, duration):
    spb = 60.0 / bpm
    beat_times = np.arange(0, duration / spb + 2) * spb
    return beat_times, (first_sec / spb) % 1.0


@torch.no_grad()
def octave_class(trees, x):
    raw = trees["baseline"].astype(np.float64).copy()
    for k, i in enumerate(trees["roots"]):
        while not trees["leaf"][i]:
            v = x[trees["feature"][i]]
            left = trees["missing_left"][i] if np.isnan(v) else v <= trees["threshold"][i]
            i = trees["left"][i] if left else trees["right"][i]
        raw[trees["cls"][k]] += trees["value"][i]
    return int(trees["classes"][np.argmax(raw)])


def _shift(act, period, phase, lo, hi):
    k = np.arange(np.ceil((lo - phase) / period), np.floor((hi - 1 - phase) / period) + 1)
    r = max(1, int(period / 4))
    offs, wts = [], []
    for c in phase + k * period:
        a, b = max(int(c) - r, 0), min(int(c) + r + 1, len(act))
        seg = act[a:b]
        if len(seg) < 3:
            continue
        j = int(np.argmax(seg))
        offs.append(a + j - c), wts.append(float(seg[j]) ** 2)
    return phase + float(np.average(offs, weights=wts)) if sum(wts) > 0 else phase


def first_beat(act, fps, bpm, seg_sec=40.0):
    lo, hi = 0, len(act)
    period = 60 * fps / bpm
    seg, mid = int(seg_sec * fps), hi // 2
    a, b = max(lo, mid - seg // 2), min(hi, mid + seg // 2)
    fit_bpm, fit_first, _ = fit_grid(act, fps)
    fp = 60 * fps / fit_bpm
    cands = [(fit_first * fps + m * fp) % period for m in range(int(np.ceil(period / fp)) + 1)]
    phase = max(cands, key=lambda ph: _marks(act, period, ph, a, b)[0])
    while True:
        phase = _shift(act, period, phase, a, b)
        if a <= lo and b >= hi:
            break
        w = b - a
        a, b = max(lo, a - w // 2), min(hi, b + w // 2)
    phase %= period
    if _marks(act, period, 0.0, lo, hi)[0] >= ZERO_KEEP * _marks(act, period, phase, lo, hi)[0]:
        return 0.0
    return phase / fps


ZERO_KEEP = 0.6


def _activation(logmel, root, dev):
    from safetensors.torch import load_file
    model = BeatNet().to(dev)
    model.load_state_dict(load_file(Path(root) / "timing.safetensors", device=str(dev)))
    model.eval()
    return activation(model, logmel, dev)


def detect(logmel, fps, root, dev="cuda", bpm=None):
    act = _activation(logmel, root, dev)
    if bpm:
        return float(bpm), first_beat(act, fps, bpm)
    bpm, first, _ = fit_grid(act, fps)
    feats, cands = octave_features(act, fps, bpm, first)
    bpm, first = cands[octave_class(dict(np.load(Path(root) / "octave.npz")), feats)]
    return float(bpm), float(first)


SECTION_SEC, SECTION_HOP = 20.0, 10.0
SWITCH = 0.1
MIN_GAIN = 0.3
RATIOS = (1, 2, 3, 4, 1 / 2, 1 / 3, 1 / 4, 2 / 3, 3 / 2, 3 / 4, 4 / 3)


def related(a, b, tol=0.01):
    return any(abs(a / b / r - 1) < tol for r in RATIOS)


def _window_scores(act, fps):
    W, H = int(SECTION_SEC * fps), int(SECTION_HOP * fps)
    starts = list(range(0, max(len(act) - W, 0) + 1, H))
    if starts[-1] + W < len(act):
        starts.append(len(act) - W)
    spans = [(lo, min(lo + W, len(act))) for lo in starts]
    fits = [fit_grid(act, fps, s, SECTION_SEC, steps=33) for s in spans]
    reps = []
    for j in np.argsort([-f[2] for f in fits]):
        if not any(related(fits[j][0], r) for r in reps):
            reps.append(fits[j][0])
    score = np.zeros((len(spans), len(reps)))
    for i, (lo, hi) in enumerate(spans):
        for k, r in enumerate(reps):
            if related(fits[i][0], r):
                score[i, k] = fits[i][2]
            else:
                p = 60 * fps / r
                score[i, k] = max(_score(act, p * (1 + x), lo, hi)[0] for x in np.linspace(-0.005, 0.005, 11))
    return spans, reps, score


def _path(score, switch=SWITCH):
    n, m = score.shape
    best, back = score[0].copy(), np.zeros((n, m), int)
    for i in range(1, n):
        move = best.max() - switch
        back[i] = np.where(best >= move, np.arange(m), int(best.argmax()))
        best = np.maximum(best, move) + score[i]
    path = [int(best.argmax())]
    for i in range(n - 1, 0, -1):
        path.append(int(back[i][path[-1]]))
    path = path[::-1]
    while True:
        runs, i = [], 0
        while i < n:
            j = i
            while j + 1 < n and path[j + 1] == path[i]:
                j += 1
            runs.append((i, j + 1, path[i]))
            i = j + 1
        weak = []
        for k, (a, b, f) in enumerate(runs):
            nb = [runs[q][2] for q in (k - 1, k + 1) if 0 <= q < len(runs)]
            if not nb:
                continue
            gain = min((score[a:b, f] - score[a:b, g]).sum() for g in nb)
            if gain < MIN_GAIN or b - a < 2:
                weak.append((gain if b - a >= 2 else -np.inf, a, b, max(nb, key=lambda g: score[a:b, g].sum())))
        if not weak:
            return path
        _, a, b, g = min(weak)
        path[a:b] = [g] * (b - a)


def _local(act, fps, bpm, first, lo, hi, step):
    period = 60 * fps / bpm
    k = np.arange(np.ceil((lo - fps - first * fps) / period), np.floor((hi + fps - first * fps) / period) + 1)
    beats = first * fps + k * period

    def peak(pos):
        g = np.clip(np.round(pos).astype(int), 1, len(act) - 2)
        return np.max([act[g - 1], act[g], act[g + 1]], 0)

    on = peak(beats)
    sub = np.stack([peak(beats + f * period) for f in SUBDIVISIONS])
    out = []
    for x in range(lo, hi, step):
        sel = np.abs(beats - x) <= fps
        out.append(on[sel].mean() - sub[:, sel].mean(1).max() if sel.any() else 0.0)
    return np.array(out)


def _fit(act, fps, span, trees):
    bpm, first, _ = fit_grid(act, fps, span)
    feats, cands = octave_features(act, fps, bpm, first, span)
    return cands[octave_class(trees, feats)]


def sections(act, fps, trees):
    spans, reps, score = _window_scores(act, fps)
    path = _path(score)
    cuts = [i for i in range(1, len(path)) if path[i] != path[i - 1]]
    if not cuts:
        return [(0.0, *_fit(act, fps, (0, len(act)), trees))]
    zones = [(spans[i][0], spans[i - 1][1]) for i in cuts]
    cores = [(a[1] if a else 0, b[0] if b else len(act)) for a, b in zip([None] + zones, zones + [None])]
    grids = [_fit(act, fps, c, trees) for c in cores]
    out = [(0.0, *grids[0])]
    step = int(0.25 * fps)
    for (lo, hi), g in zip(zones, grids[1:]):
        lo, hi = max(lo - spans[0][1] // 4, 0), min(hi + spans[0][1] // 4, len(act))
        gain = np.cumsum(_local(act, fps, *out[-1][1:], lo, hi, step) - _local(act, fps, *g, lo, hi, step))
        x = (lo + step * int(np.argmax(gain))) / fps
        if related(out[-1][1], g[0], 0.002):
            continue
        out.append((x, *g))
    return out


def tempo_map(secs):
    from .beatmap import TempoMap
    _, bpm0, beat0 = secs[0]
    offset = beat0 * bpm0 / 60 % 1
    changes = []
    for start, bpm, beat in secs[1:]:
        tm = TempoMap(bpm0, changes)
        p = 60 / bpm
        first = beat + np.ceil((start - beat) / p) * p
        n = np.floor(tm.to_beats(first) - 0.5 - offset) + offset
        gap = first - tm.to_seconds(n)
        changes += [(float(n), 60 / gap), (float(n + 1), bpm)]
    return TempoMap(bpm0, changes), offset


def detect_tempo(logmel, fps, root, dev="cuda"):
    act = _activation(logmel, root, dev)
    trees = dict(np.load(Path(root) / "octave.npz"))
    secs = sections(act, fps, trees)
    if len(secs) == 1:
        bpm, first, _ = fit_grid(act, fps)
        feats, cands = octave_features(act, fps, bpm, first)
        bpm, first = cands[octave_class(trees, feats)]
        secs = [(0.0, float(bpm), float(first))]
    tm, offset = tempo_map(secs)
    return tm, offset, secs
