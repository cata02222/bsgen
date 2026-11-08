import numpy as np

from . import tokens
from .tokens import TICKS


SECTION = 8
FEATURES = ("turn", "wide", "reach", "cross", "stack", "both", "swings", "hits")
_ANGLE = np.array([180.0, 0.0, 270.0, 90.0, 225.0, 135.0, 315.0, 45.0, np.nan])


def features(ev):
    notes = ev[ev["kind"] == tokens.NOTE]
    if not len(notes):
        return 0, np.zeros((0, len(FEATURES)), np.float32)
    first = int(notes["tick"].min()) // (SECTION * TICKS)
    n = int(notes["tick"].max()) // (SECTION * TICKS) - first + 1
    span = SECTION * TICKS
    add = np.zeros((n, 6), np.float64)
    den = np.zeros((n, 3), np.float64)
    swings = ev[(ev["kind"] == tokens.SWING) & ~ev["air"]]
    for hand in (0, 1):
        s = swings["tick"][swings["hand"] == hand]
        m = notes[notes["hand"] == hand]
        if not len(s) or not len(m):
            continue
        owner = np.clip(np.searchsorted(s, m["tick"], "right") - 1, 0, len(s) - 1)
        ids, start, count = np.unique(owner, return_index=True, return_counts=True)
        tick, cell = s[ids], m["cell"][start].astype(np.int64)
        sec = tick // span - first
        ok = (sec >= 0) & (sec < n)
        x, y = cell % 4, cell // 4
        np.add.at(add[:, 3], sec[ok], ((x >= 2) if hand == 0 else (x <= 1))[ok].astype(np.float64))
        np.add.at(add[:, 4], sec[ok], count[ok].astype(np.float64))
        np.add.at(den[:, 1], sec[ok], 1.0)
        if len(ids) < 2:
            continue
        a = _ANGLE[np.clip(m["dir"][start], 0, 8)]
        off = 180.0 - np.abs((a[1:] - a[:-1] + 180.0) % 360.0 - 180.0)
        reach = np.hypot(np.diff(x.astype(np.float64)), np.diff(y.astype(np.float64)))
        s1, ok1 = sec[1:], ok[1:]
        np.add.at(add[:, 2], s1[ok1], reach[ok1])
        known = ok1 & ~np.isnan(off)
        np.add.at(add[:, 0], s1[known], off[known])
        np.add.at(add[:, 1], s1[known], (off[known] > 30.0).astype(np.float64))
        np.add.at(den[:, 0], s1[known], 1.0)
    ticks, hands = np.unique(notes["tick"], return_counts=True)
    both = np.zeros(len(ticks), bool)
    left = np.unique(notes["tick"][notes["hand"] == 0])
    right = np.unique(notes["tick"][notes["hand"] == 1])
    both[np.searchsorted(ticks, np.intersect1d(left, right))] = True
    sec = ticks // span - first
    np.add.at(add[:, 5], sec, both.astype(np.float64))
    np.add.at(den[:, 2], sec, 1.0)
    out = np.zeros((n, len(FEATURES)), np.float32)
    swing = np.maximum(den[:, 1], 1)
    out[:, 0] = add[:, 0] / np.maximum(den[:, 0], 1)
    out[:, 1] = add[:, 1] / np.maximum(den[:, 0], 1)
    out[:, 2] = add[:, 2] / swing
    out[:, 3] = add[:, 3] / swing
    out[:, 4] = add[:, 4] / swing
    out[:, 5] = add[:, 5] / np.maximum(den[:, 2], 1)
    out[:, 6], out[:, 7] = den[:, 1], den[:, 2]
    return first, out


_FIT = dict(mu=[19.0067, 0.3471, 1.6753, 0.2024, 1.087], sd=[12.6387, 0.1419, 0.2748, 0.0972, 0.0947], coef=[0.0757, 0.4817, -0.1536, 1.2379, 0.5578], mean=4.446)
LEVEL_FEATURES = (0, 1, 2, 3, 4)
LEVEL_MEAN = np.array(_FIT["mu"], np.float32)
LEVEL_STD = np.array(_FIT["sd"], np.float32)
LEVEL_COEF = np.array(_FIT["coef"], np.float32)
LEVEL_BASE = _FIT["mean"]
SMOOTH = np.array([1.0, 2.0, 1.0])
LEVEL_MAX = 12.0


def levels(ev):
    first, x = features(ev)
    if not len(x):
        return first, np.zeros((0, 3), np.float32)
    raw = LEVEL_BASE + ((x[:, LEVEL_FEATURES] - LEVEL_MEAN) / LEVEL_STD) @ LEVEL_COEF
    out = np.zeros((len(x), 3), np.float32)
    for j, (v, w) in enumerate(((raw, x[:, 6]), (x[:, 5], x[:, 7]), (x[:, 1], x[:, 6]))):
        w = w.astype(np.float64)
        num = np.convolve(v * w, SMOOTH)[1:-1]
        den = np.convolve(w, SMOOTH)[1:-1]
        fill = float((v * w).sum() / max(w.sum(), 1.0))
        out[:, j] = np.where(den > 0, num / np.maximum(den, 1e-9), fill)
    out[:, 0] = np.clip(out[:, 0], 0.0, LEVEL_MAX)
    return first, out


def plan_inputs(ev, beat_times, offset, stars, logmel, fps, song):
    tick = np.unique(ev["tick"][ev["kind"] == tokens.NOTE])
    return plan_rows(tick, beat_times, offset, stars, logmel, fps, song)


def plan_rows(tick, beat_times, offset, stars, logmel, fps, song):
    from .data import beat_to_sec

    if len(tick) < 20:
        return None, 0.0
    span = SECTION * TICKS
    first, last = int(tick[0]) // span, int(tick[-1]) // span
    n = last - first + 1
    edges = beat_to_sec((first + np.arange(n + 1)) * float(SECTION) + offset, beat_times)
    secs = np.maximum(np.diff(edges), 1e-3)
    hits = np.bincount(tick // span - first, minlength=n).astype(np.float64)
    rate = hits / secs
    sec = beat_to_sec(tick / TICKS + offset, beat_times)
    whole = len(tick) / max(sec[-1] - sec[0], 1.0)
    loud = np.asarray(logmel, np.float32).mean(1)
    pos = np.clip((edges * fps).astype(np.int64), 0, len(loud) - 1)
    near = np.array([loud[a:max(b, a + 1)].mean() for a, b in zip(pos[:-1], pos[1:])])
    before = np.concatenate([rate[:1], rate[:-1]])
    after = np.concatenate([rate[1:], rate[-1:]])
    beat_s = np.median(secs) / SECTION
    x = np.column_stack([
        np.full(n, stars), np.full(n, whole), np.full(n, whole * beat_s), rate, rate / max(whole, 1e-3), hits / SECTION,
        before, after, (edges[:-1] - edges[0]) / max(edges[-1] - edges[0], 1.0), near, near - loud.mean(),
        np.full(n, 1.0 / beat_s), np.tile(np.asarray(song, np.float64), (n, 1)),
    ])
    return x.astype(np.float32), whole


BOTH_EDGES = np.array([0.05, 0.15, 0.3, 0.5])
WIDE_EDGES = np.array([0.15, 0.25, 0.35, 0.45, 0.6])
PATTERN_DIM = int(LEVEL_MAX) + 1 + len(BOTH_EDGES) + 1 + len(WIDE_EDGES) + 1 + 1
WIDE_PER_LEVEL = 0.055


def roll(first, lv, t0, slots, slot_ticks):
    out = np.zeros((slots, PATTERN_DIM), np.float32)
    if lv is None or not len(lv):
        return out
    sec = np.clip((t0 + np.arange(slots) * slot_ticks) // (SECTION * TICKS) - first, 0, len(lv) - 1)
    k = np.arange(slots)
    out[k, np.rint(np.clip(lv[sec, 0], 0.0, LEVEL_MAX)).astype(np.int64)] = 1
    out[k, int(LEVEL_MAX) + 1 + np.searchsorted(BOTH_EDGES, lv[sec, 1])] = 1
    out[k, int(LEVEL_MAX) + 2 + len(BOTH_EDGES) + np.searchsorted(WIDE_EDGES, lv[sec, 2])] = 1
    out[:, -1] = 1
    return out


class Trees:

    def __init__(self, trees):
        self.trees = trees

    def predict(self, x):
        from .timing import forest_raw
        return np.array([forest_raw(self.trees, row)[0] for row in np.asarray(x, np.float64)])


def load_plan(path):
    from pathlib import Path
    from .timing import forests
    f = forests(path)
    names = ("level", "both", "wide")
    return dict(models={n: Trees(f[n]) for n in names}, taste={n: float(f["taste"][n]) for n in names},
                wander={n: tuple(float(v) for v in f["wander"][n]) for n in names})


def make_plan(plan, tick, beat_times, offset, stars, logmel, fps, song, lift=0.0, rng=None):
    x, _ = plan_rows(np.unique(tick), beat_times, offset, stars, logmel, fps, song)
    if x is None:
        return 0, None
    made = {}
    for name, model in plan["models"].items():
        made[name] = model.predict(x)
        if rng is not None:
            made[name] = made[name] + rng.normal(0, plan["taste"][name]) + _wander(len(x), plan["wander"][name], rng)
    out = np.column_stack([np.clip(made["level"] + lift, 0.0, LEVEL_MAX), np.clip(made["both"], 0.0, 1.0),
                           np.clip(made["wide"] + lift * WIDE_PER_LEVEL, 0.0, 1.0)]).astype(np.float32)
    return int(np.min(tick)) // (SECTION * TICKS), out


def _wander(n, sizes, rng):
    spread, corr = sizes
    out = np.zeros(n)
    now = rng.normal(0, spread)
    for i in range(n):
        out[i] = now
        now = corr * now + rng.normal(0, spread * (1 - corr * corr) ** 0.5)
    return out
