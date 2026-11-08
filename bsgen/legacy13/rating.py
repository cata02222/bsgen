import numpy as np

from . import labeler as L

DIR_ANGLE = np.array([180, 0, 270, 90, 225, 135, 315, 45, np.nan])


def _pct(x, q):
    return np.percentile(x, q) if len(x) else 0.0


def features(notes, njs=16.0, lab=None):
    lab = L.parity_rule(notes) if lab is None else lab
    t = notes["time"].astype(float)
    dur = max(t[-1] - t[0], 1.0) if len(t) else 1.0
    f = {"notes/s": len(notes) / dur, "duration": dur, "njs": njs}
    for w in (2.0, 8.0):
        k = np.searchsorted(t, t + w) - np.arange(len(t))
        f[f"peak nps {w:g}s p90"], f[f"peak nps {w:g}s max"] = _pct(k / w, 90), (k.max() / w if len(k) else 0.0)
    f["dots"] = float(np.mean(notes["cut_dir"] == 8)) if len(notes) else 0.0
    rates, gaps, dists, speeds, angles, resets = [], [], [], [], [], 0
    for h in (0, 1):
        m = notes["color"] == h
        new = m & (lab != L.JOIN)
        ts = t[new]
        rates.append(len(ts) / dur)
        if len(ts) < 3:
            continue
        g = np.diff(ts)
        x, y = notes["line"][new].astype(float), notes["layer"][new].astype(float)
        dist = np.hypot(np.diff(x), np.diff(y))
        a = DIR_ANGLE[notes["cut_dir"][new].clip(0, 8)]
        da = np.abs((np.diff(a) + 180) % 360 - 180)
        fore = lab[new] == L.NEW_FORE
        resets += int(np.sum(fore[1:] == fore[:-1]))
        gaps.append(g)
        dists.append(dist)
        speeds.append(dist / np.maximum(g, 0.05))
        angles.append(da[~np.isnan(da)])
    cat = lambda xs: np.concatenate(xs) if xs else np.zeros(0)
    gaps, dists, speeds, angles = cat(gaps), cat(dists), cat(speeds), cat(angles)
    f["swings/s"], f["swings/s hand max"] = sum(rates), max(rates)
    for q in (10, 50):
        f[f"swing gap p{q}"] = _pct(gaps, q)
    for q in (50, 90):
        f[f"jump p{q}"], f[f"move speed p{q}"], f[f"angle change p{q}"] = _pct(dists, q), _pct(speeds, q), _pct(angles, q)
    f["resets/swing"] = resets / max(len(gaps), 1)
    f["stacked"] = float(np.mean(lab == L.JOIN)) if len(lab) else 0.0
    f["crossovers/s"] = 0.0
    if len(notes):
        r, b = notes[notes["color"] == 0], notes[notes["color"] == 1]
        if len(r) and len(b):
            j = np.clip(np.searchsorted(b["time"], r["time"]), 0, len(b) - 1)
            close = np.abs(b["time"][j] - r["time"]) < 0.25
            f["crossovers/s"] = float(np.sum(close & (r["line"] > b["line"][j]))) / dur
    return f


FEATURE_NAMES = list(features(np.zeros(0, [("beat", "f8"), ("time", "f8"), ("line", "i1"), ("layer", "i1"),
                                             ("color", "i1"), ("cut_dir", "i1")]), lab=np.zeros(0, int)).keys())


def vector(notes, njs=16.0, lab=None):
    f = features(notes, njs, lab)
    return np.array([f[k] for k in FEATURE_NAMES], np.float32)


TARGETS = ("stars", "tech", "pass")
_estimator = {}


def _trees():
    if "trees" not in _estimator:
        from . import hub, timing
        path = hub.rating_model()
        _estimator["trees"] = None if path is None else timing.forests(path)
    return _estimator["trees"]


def available():
    return _trees() is not None


def estimate(notes, njs=16.0, times=None):
    trees = _trees()
    if trees is None:
        return None
    if "time" not in notes.dtype.names:
        from numpy.lib import recfunctions
        notes = recfunctions.append_fields(notes, "time", np.asarray(times, float), usemask=False)
    notes = notes[notes["color"] < 2] if "color" in notes.dtype.names else notes
    if len(notes) < 20:
        return None
    from .timing import forest_raw
    x = vector(notes[L.order(notes)], njs or 16.0).astype(np.float64)
    return tuple(float(forest_raw(trees[name], x)[0]) for name in TARGETS)
