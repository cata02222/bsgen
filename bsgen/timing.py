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
    shift = np.array([int(round(f * bins)) for f in SUBDIVISIONS])
    off = prof[(np.arange(len(prof))[None] + shift[:, None]) % len(prof)].max(0)
    val = prof - off
    j = int(np.argmax(val))
    return float(val[j]), (j + 0.5) / bins * period


PEAK_REACH = 0.05


def _peaks(act, beats, period, fps):
    r = max(1, int(min(period / 4, PEAK_REACH * fps)))
    pos, height = np.array(beats, np.float64), np.zeros(len(beats))
    for i, c in enumerate(beats):
        m = int(round(c))
        a, b = max(m - r, 0), min(m + r + 1, len(act))
        seg = act[a:b]
        if len(seg) < 3:
            continue
        j = int(np.argmax(seg))
        if 0 < j < len(seg) - 1:
            d = seg[j - 1] - 2 * seg[j] + seg[j + 1]
            pos[i] = a + j + (0.5 * (seg[j - 1] - seg[j + 1]) / d if d < 0 else 0.0)
            height[i] = seg[j]
    return pos, height


def _refine(act, period, phase, lo, hi, fps):
    for _ in range(3):
        k = np.arange(np.ceil((lo - phase) / period), np.floor((hi - 1 - phase) / period) + 1)
        idx, wts = _peaks(act, phase + k * period, period, fps)
        wts = wts ** 2
        if wts.sum() <= 0 or len(k) < 4:
            break
        A = np.stack([k, np.ones_like(k)], 1) * np.sqrt(wts)[:, None]
        sol = np.linalg.lstsq(A, idx * np.sqrt(wts), rcond=None)[0]
        period, phase = float(sol[0]), float(sol[1])
    return period, phase


def fit_grid(act, fps, span=None, seg_sec=40.0, steps=401):
    lo, hi = span or (0, len(act))
    fits = [_fit_from(act, fps, lo, hi, lo + (hi - lo) * f, int(seg_sec * fps), steps) for f in (0.5, 0.25, 0.75)]
    score = [_score(act, period, lo, hi)[0] for period, _, _ in fits]
    period, phase, s = fits[int(np.argmax(score))]
    return 60 * fps / period, phase / fps, s


def _fit_from(act, fps, lo, hi, centre, seg, steps):
    mid = int(centre)
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
    period, phase = _refine(act, period, phase, slo, shi, fps)
    a, b = slo, shi
    while a > lo or b < hi:
        w = b - a
        a, b = max(lo, a - w // 2), min(hi, b + w // 2)
        period, phase = _refine(act, period, phase, a, b, fps)
    return period, phase % period, s


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


def forest_raw(trees, x):
    raw = trees["baseline"].astype(np.float64).copy()
    x = np.asarray(x)
    i = np.asarray(trees["roots"], np.int64).copy()
    leaf, feat, thr = trees["leaf"], trees["feature"], trees["threshold"]
    go = ~leaf[i].astype(bool)
    while go.any():
        j = i[go]
        v = x[feat[j]]
        nan = np.isnan(v)
        left = np.where(nan, trees["missing_left"][j].astype(bool), np.where(nan, False, v <= thr[j]))
        i[go] = np.where(left, trees["left"][j], trees["right"][j])
        go = ~leaf[i].astype(bool)
    np.add.at(raw, np.asarray(trees["cls"], np.int64), trees["value"][i].astype(np.float64))
    return raw


def forests(path):
    out = {}
    with np.load(path) as z:
        for key in z.files:
            name, _, field = key.rpartition("/")
            out.setdefault(name, {})[field] = z[key]
    return out


@torch.no_grad()
def octave_class(trees, x):
    return int(trees["classes"][np.argmax(forest_raw(trees, x))])


def _shift(act, period, phase, lo, hi, fps):
    k = np.arange(np.ceil((lo - phase) / period), np.floor((hi - 1 - phase) / period) + 1)
    beats = phase + k * period
    pos, height = _peaks(act, beats, period, fps)
    wts = height ** 2
    return phase + float(np.average(pos - beats, weights=wts)) if wts.sum() > 0 else phase


ZERO_NEAR = 0.010


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
        phase = _shift(act, period, phase, a, b, fps)
        if a <= lo and b >= hi:
            break
        w = b - a
        a, b = max(lo, a - w // 2), min(hi, b + w // 2)
    phase %= period
    if min(phase, period - phase) <= ZERO_NEAR * fps:
        return 0.0
    return phase / fps


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
    i = octave_class(trees, feats)
    return (*cands[i], *cands[1 if i != 1 else 0])


FAST_BAND = (200.0, 245.0)


def choose_octave(secs, want):
    out, changed = [], []
    for start, bpm, first, *alt in secs:
        if want and len(alt) == 2 and abs(max(bpm, alt[0]) / min(bpm, alt[0]) - 2) < 0.01:
            fast, half = ((bpm, first), tuple(alt)) if bpm > alt[0] else (tuple(alt), (bpm, first))
            if FAST_BAND[0] <= fast[0] <= FAST_BAND[1]:
                pick = fast if want == "fast" else half
                if pick[0] != bpm:
                    changed.append((start, bpm, pick[0]))
                bpm, first = pick
        out.append((start, bpm, first))
    return out, changed


def sections(act, fps, trees, whole=None):
    spans, reps, score = _window_scores(act, fps)
    path = _path(score)
    cuts = [i for i in range(1, len(path)) if path[i] != path[i - 1]]
    if not cuts:
        if whole is not None:
            whole.append(True)
        return [(0.0, *_fit(act, fps, (0, len(act)), trees))]
    zones = [(spans[i][0], spans[i - 1][1]) for i in cuts]
    cores = [(a[1] if a else 0, b[0] if b else len(act)) for a, b in zip([None] + zones, zones + [None])]
    grids = [_fit(act, fps, c, trees) for c in cores]
    out = [(0.0, *grids[0])]
    step = int(0.25 * fps)
    for (lo, hi), g in zip(zones, grids[1:]):
        lo, hi = max(lo - spans[0][1] // 4, 0), min(hi + spans[0][1] // 4, len(act))
        gain = np.cumsum(_local(act, fps, *out[-1][1:3], lo, hi, step) - _local(act, fps, *g[:2], lo, hi, step))
        x = (lo + step * int(np.argmax(gain))) / fps
        if related(out[-1][1], g[0], 0.002):
            continue
        out.append((x, *g))
    return out


def tempo_map(secs, warp=None):
    from .beatmap import TempoMap
    secs = [tuple(x[:3]) for x in secs]
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
    tm = TempoMap(bpm0, changes)
    return (_warped(tm, offset, warp) if warp else tm), offset


FOLLOW_SMOOTH = 2.0
FOLLOW_MIN = 0.010
FOLLOW_GAIN = 0.08
FOLLOW_EVERY = 4
ON_GRID = 0.015


def _grid_beats(tm, offset, dur):
    return np.arange(0, int(float(tm.to_beats(dur)) - offset) + 1) + offset


def _curve(act, fps, tm, offset, passes=4, eps=0.5):
    k = _grid_beats(tm, offset, len(act) / fps)
    g = tm.to_seconds(k)
    r = np.zeros(len(g))
    cuts = [0] + [int(round(b - offset)) for b in tm.beats[2::2]] + [len(k)]
    for _ in range(passes):
        step = np.zeros(len(g))
        for a, z in zip(cuts[:-1], cuts[1:]):
            if z - a < 4:
                continue
            t = g[a:z]
            b = (t + r[a:z]) * fps
            pos, height = _peaks(act, b, float(np.median(np.diff(t))) * fps, fps)
            w = height ** 2
            if not (w > 0).any():
                continue
            near = np.exp(-0.5 * ((t[:, None] - t[None, :]) / FOLLOW_SMOOTH) ** 2)
            step[a:z] = (near @ (w * (pos - b))) / (near @ w + eps * np.median(w[w > 0]))
        r += step / fps
    return g, r


def _warped(tm, offset, warp):
    from .beatmap import TempoMap
    at, shift = (np.asarray(x, np.float64) for x in warp)
    k = _grid_beats(tm, offset, at[-1] + 1e-6)
    g = tm.to_seconds(k)
    sec = g + np.interp(g, at, shift)
    i0 = 0 if k[0] > 0.05 else 1
    if len(k) < i0 + FOLLOW_EVERY + 1:
        return tm
    sec[i0] = g[i0]
    idx = sorted(set(range(i0, len(k), FOLLOW_EVERY)) | {int(round(b - offset)) for b in tm.beats[1:]})
    idx = [i for i in idx if i0 <= i < len(k)]
    beats, secs = k[idx], sec[idx]
    keep = np.concatenate([[True], np.diff(secs) > 0.05])
    beats, secs = beats[keep], secs[keep]
    changes = [(float(b), float(60 * db / ds)) for b, db, ds in zip(beats, np.diff(beats), np.diff(secs))]
    changes.append((float(beats[-1]), float(tm.bpms[np.searchsorted(tm.beats, beats[-1], "right") - 1])))
    return TempoMap(float(tm.bpms[0]), changes)


def sounds_on(acc, strength, tm, offset):
    q = tm.to_seconds(np.arange(0, float(tm.to_beats(acc[-1])) * 4 + 8) / 4 + offset % 0.25)
    a = acc[(strength >= np.median(strength)) & (acc > q[1]) & (acc < q[-2])]
    if not len(a):
        return 0.0
    j = np.clip(np.searchsorted(q, a), 1, len(q) - 1)
    return float(np.mean(np.minimum(np.abs(q[j] - a), np.abs(a - q[j - 1])) <= ON_GRID))


def follow(act, logmel, fps, secs):
    from .rhythm import accents
    tm, offset = tempo_map(secs)
    at, shift = _curve(act, fps, tm, offset)
    if len(at) < 2 * FOLLOW_EVERY + 2 or np.abs(shift).max() < FOLLOW_MIN:
        return None
    warp = [[round(float(v), 5) for v in at], [round(float(v), 5) for v in shift]]
    acc, strength = accents(logmel, fps, strength=True)
    if len(acc) < 40 or sounds_on(acc, strength, _warped(tm, offset, warp), offset) - sounds_on(acc, strength, tm, offset) < FOLLOW_GAIN:
        return None
    return warp


FIT = 3


def detect_tempo(logmel, fps, root, dev="cuda", follow_song=False):
    import hashlib, json
    model = Path(root) / "timing.safetensors"
    key = hashlib.sha1(np.ascontiguousarray(logmel).tobytes()
                       + f"{fps}|{model.stat().st_size}|{model.stat().st_mtime_ns}|{FIT}".encode()).hexdigest()
    cache = Path(root) / "tempo_cache" / f"{key}.json"
    try:
        kept = json.loads(cache.read_text())
        secs, warp = [tuple(x) for x in kept["secs"]], kept["warp"]
    except (OSError, ValueError, KeyError, TypeError):
        act = _activation(logmel, root, dev)
        trees = dict(np.load(Path(root) / "octave.npz"))
        whole = []
        secs = sections(act, fps, trees, whole)
        if len(secs) == 1 and not whole:
            secs = [(0.0, *(float(v) for v in _fit(act, fps, (0, len(act)), trees)))]
        warp = follow(act, logmel, fps, secs)
        try:
            cache.parent.mkdir(exist_ok=True)
            cache.write_text(json.dumps({"secs": [[float(v) for v in x] for x in secs], "warp": warp}))
        except OSError:
            pass
    if follow_song:
        return (*tempo_map(secs, warp), secs, warp)
    return (*tempo_map(secs), secs)
