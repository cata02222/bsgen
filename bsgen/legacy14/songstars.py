from pathlib import Path

import numpy as np

from .timing import forest_raw

PATH = Path(__file__).resolve().parents[1] / "models" / "song_stars.pkl"
NAMES = ["Easy", "Normal", "Hard", "Expert", "ExpertPlus"]
MEDIAN = {"Easy": 3.5, "Normal": 4.5, "Hard": 5.5, "Expert": 7.5, "ExpertPlus": 10.0}
GAP = 1.0
_models = {}


def features(logmel, fps):
    m = np.asarray(logmel, np.float32)
    loud = m.mean(1)
    flux = np.maximum(np.diff(m, axis=0), 0).sum(1)
    play = loud > loud.max() - 25
    f = flux[play[1:]]
    peak = (flux[1:-1] > flux[:-2]) & (flux[1:-1] >= flux[2:])
    rates = [np.sum(peak & (flux[1:-1] > t)) / (play.sum() / fps) for t in (20, 40, 80, 160)]
    return [len(m) / fps, loud.mean(), loud.std(), np.percentile(loud, 90) - np.percentile(loud, 10),
            f.mean(), np.median(f), np.percentile(f, 90), m[:, 50:].mean() - loud.mean(), *rates]


def stars(logmel, fps):
    m = load()
    if m is None:
        return dict(MEDIAN)
    x = np.array(features(logmel, fps), np.float64)
    out, low = {}, 0.0
    for name in NAMES:
        s = float(np.clip(forest_raw(m[name], x)[0], 1.0, 15.0))
        s = round(max(s, low) * 2) / 2
        out[name], low = s, s + GAP
    return out


def load():
    if "m" not in _models:
        from . import hub, timing
        path = hub.song_stars_model()
        _models["m"] = None if path is None else timing.forests(path)
    return _models["m"]


def how(logmel, fps, stars):
    m = load()
    if m is None or "how.hps" not in m:
        return None
    x = np.array(features(logmel, fps) + [stars], np.float64)
    return {"hps": float(np.exp(forest_raw(m["how.hps"], x)[0])), "tech": float(forest_raw(m["how.tech"], x)[0]),
            "pass": float(forest_raw(m["how.pass"], x)[0])}
