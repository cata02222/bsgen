import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from . import beatmap, bsor, hub, judge, labeler
from .player import HZ, UNKNOWN, Player, objects_from_map, orthonormal, play

BSOR_HEIGHT = 1.6
_labeler = {}


def load_player(path, dev="cuda"):
    path = Path(path)
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        ck = dict(json.loads(path.with_suffix(".json").read_text()), model=load_file(path))
    else:
        ck = torch.load(path, map_location="cpu")
    model = Player(**ck["config"]).to(dev)
    model.load_state_dict(ck["model"])
    return model.eval()


def swing_plan(notes, dev="cuda", bombs=None):
    if not len(notes):
        return np.zeros(0, np.int8)
    if dev not in _labeler:
        path = hub.labeler_model()
        if path.suffix == ".safetensors":
            from safetensors.torch import load_file
            weights = load_file(path)
        else:
            weights = torch.load(path, map_location="cpu")["model"]
        m = labeler.Labeler().to(dev).eval()
        m.load_state_dict(weights)
        _labeler[dev] = m
    o = labeler.order(notes)
    plan = np.full(len(notes), UNKNOWN, np.int8)
    plan[o] = labeler.label(_labeler[dev], notes[o], dev, quick_reset=labeler.QUICK_RESET_S, bombs=bombs) + 1
    return plan


def map_objects(zip_path, difficulty, characteristic="Standard", plan=False, dev="cuda"):
    with beatmap.open_map(zip_path) as z:
        info = beatmap.read_info(z)
        d = beatmap.read_difficulty(z, info, characteristic, difficulty)
    n_beats = int(max([d.notes["beat"].max() if len(d.notes) else 0] +
                      [(d.walls["beat"] + d.walls["duration"]).max() if len(d.walls) else 0])) + 4
    beat_times = d.tempo.to_seconds(np.arange(n_beats, dtype=float)) + info.song_offset
    plan = swing_plan(d.notes, dev, d.bombs) if plan else None
    return objects_from_map(d.notes, d.bombs, d.walls, beat_times, plan), d.njs or 18.0, info


def from_song_start(motion, t0):
    n0 = int(round(t0 * HZ))
    return np.concatenate([np.repeat(motion[:1], n0, 0), motion]) if n0 >= 0 else motion[-n0:]


def write_replay(path, zip_path, difficulty, motion, info, njs, objects, player="bsgen bot"):
    events = judge.note_events(judge.Notes(objects, njs), motion, BSOR_HEIGHT)
    m = motion.reshape(-1, 3, 9).astype(np.float64)
    quat = Rotation.from_matrix(judge.rotations(m[..., 3:]).reshape(-1, 3, 3)).as_quat().reshape(-1, 3, 4)
    meta = {"player": player, "hash": bsor.map_hash(zip_path, info), "song": info.song_name,
            "mapper": info.mapper, "difficulty": difficulty, "score": judge.replay_score(events),
            "environment": "DefaultEnvironment", "height": BSOR_HEIGHT}
    bsor.write_bsor(path, meta, m[..., :3], quat, events, HZ)
    return events


def summary(events):
    kinds = np.array([e[5] for e in events])
    return {"notes": len(events), "good": int(np.sum(kinds == 0)), "bad": int(np.sum(kinds == 1)),
            "missed": int(np.sum(kinds == 2)), "score": judge.replay_score(events)}


def describe(s):
    return (f"{s['good']} of {s['notes']} notes cut, {s['bad']} bad cut{'s' * (s['bad'] != 1)}, "
            f"{s['missed']} missed, score {s['score']}")


TOUCH_BEFORE, TOUCH_AFTER = 24, 18
TOUCH_SHIFTS = (-0.02, -0.04, -0.06, 0.02, -0.08, 0.04)
TOUCH_REACH = 0.12


def _outcomes(notes, motion, a, b):
    r = judge.simulate(notes, motion[None, a:b], a / HZ)
    fg, fb, never = judge.first_touches(r)
    return r.ids, np.where((fg[0] < fb[0]), 1, np.where(fb[0] < never, -1, 0))


def _bump(n, i):
    w = np.zeros(n)
    w[:i + 1] = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, i + 1))
    w[i:] = 0.5 + 0.5 * np.cos(np.linspace(0, np.pi, n - i))
    return w


def touch_up(motion, objects, njs):
    notes = judge.Notes(objects, njs)
    good, bad, _, _ = judge.replay_hits(notes, motion)
    motion = motion.copy()
    saved = 0
    for k in np.flatnonzero((good < 0) & (bad < 0)):
        f = int(round(notes.time[k] * HZ))
        a, b = f - TOUCH_BEFORE, f + TOUCH_AFTER
        if a < 9 or b + 9 > len(motion):
            continue
        h = 9 * (1 + int(notes.color[k]))
        lo, hi = a - 8, b + 8
        ids, before = _outcomes(notes, motion, lo, hi)
        want = before.copy()
        want[ids == k] = 1
        w = _bump(b - a, TOUCH_BEFORE)
        piece = motion[a:b, h:h + 9]
        tries = []
        for dt in TOUCH_SHIFTS:
            src = np.arange(a, b) - dt * HZ * w
            j = np.clip(src.astype(int), 0, len(motion) - 2)
            u = (src - j)[:, None]
            tries.append((1 - u) * motion[j, h:h + 9] + u * motion[j + 1, h:h + 9])
        hilt, blade = judge.tips(motion[a:b].astype(np.float32))
        c = int(notes.color[k])
        pts = hilt[:, c, None] + blade[:, c, None] * judge.BLADE_POINTS[:, None]
        depth = (notes.time[k] - np.arange(a, b) / HZ) * notes.njs + judge.NOTE_DEPTH + judge.CUT_BOX_SHIFT
        centre = np.concatenate([np.broadcast_to(notes.centre[k], (b - a, 2)), depth[:, None]], 1)[:, None]
        loc = judge._local(pts, centre, notes.across[k], notes.along[k])
        out = np.sign(loc) * np.maximum(np.abs(loc) - judge.CUT_BOX * 0.8, 0)
        out[..., 2] = 0
        d = np.linalg.norm(out, axis=-1)
        fi, pi = np.unravel_index(np.argmin(d), d.shape)
        if 0 < d[fi, pi] <= TOUCH_REACH:
            move = -(out[fi, pi, 0] * notes.across[k] + out[fi, pi, 1] * notes.along[k])
            t = piece.copy()
            t[:, :3] += _bump(b - a, fi)[:, None] * move
            tries.append(t)
        for t in tries:
            trial = motion[lo:hi].copy()
            trial[a - lo:b - lo, h:h + 9] = orthonormal(np.tile(t, 3))[:, h:h + 9]
            test = motion.copy()
            test[lo:hi] = trial
            ids2, after = _outcomes(notes, test, lo, hi)
            if np.array_equal(ids2, ids) and np.array_equal(after, want):
                motion[lo:hi] = trial
                saved += 1
                break
    return motion, saved


CLEAR_SHIFTS = (0.02, -0.02, 0.04, -0.04, 0.06, -0.06)
CLEAR_MOVES = (0.05, 0.1, 0.15)


def clear_crossings(motion, objects, njs):
    notes = judge.Notes(objects, njs)
    good, bad, bad_saber, _ = judge.replay_hits(notes, motion)
    motion = motion.copy()
    saved = 0
    for k in np.flatnonzero((bad >= 0) & ((good < 0) | (bad < good)) & (bad_saber >= 0) & (bad_saber != notes.color)):
        f = int(round(notes.time[k] * HZ))
        a, b = f - TOUCH_BEFORE, f + TOUCH_AFTER
        if a < 9 or b + 9 > len(motion):
            continue
        lo, hi = a - 8, b + 8
        ids, before = _outcomes(notes, motion, lo, hi)
        want = before.copy()
        want[ids == k] = 1
        w = _bump(b - a, TOUCH_BEFORE)
        own, other = 9 * (1 + int(notes.color[k])), 9 * (1 + int(bad_saber[k]))
        tries = []

        def shifted(h, dt, w=w):
            src = np.arange(a, b) - dt * HZ * w
            j = np.clip(src.astype(int), 0, len(motion) - 2)
            u = (src - j)[:, None]
            return h, (1 - u) * motion[j, h:h + 9] + u * motion[j + 1, h:h + 9]

        tries += [shifted(other, dt) for dt in CLEAR_SHIFTS]
        tries += [shifted(own, dt) for dt in TOUCH_SHIFTS if dt < 0]
        fb = int(np.clip(bad[k] - a, 0, b - a - 1))
        hilt, blade = judge.tips(motion[a + fb:a + fb + 1].astype(np.float32))
        o = int(bad_saber[k])
        pts = hilt[0, o] + blade[0, o] * judge.BLADE_POINTS[:, None]
        depth = (notes.time[k] - (a + fb) / HZ) * notes.njs + judge.NOTE_DEPTH
        centre = np.r_[notes.centre[k], depth]
        loc = judge._local(pts, centre, notes.across[k], notes.along[k])
        side = loc[np.argmin(np.linalg.norm(loc, axis=-1)), :2]
        if np.linalg.norm(side) > 1e-3:
            side = side / np.linalg.norm(side)
            way = side[0] * notes.across[k] + side[1] * notes.along[k]
            for d in CLEAR_MOVES:
                t = motion[a:b, other:other + 9].copy()
                t[:, :3] += _bump(b - a, fb)[:, None] * way * d
                tries.append((other, t))
        for h, t in tries:
            trial = motion[lo:hi].copy()
            trial[a - lo:b - lo, h:h + 9] = orthonormal(np.tile(t, 3))[:, h:h + 9]
            test = motion.copy()
            test[lo:hi] = trial
            ids2, after = _outcomes(notes, test, lo, hi)
            if np.array_equal(ids2, ids) and np.array_equal(after, want):
                motion[lo:hi] = trial
                saved += 1
                break
    return motion, saved


def play_motion(model, zip_path, difficulty, seed=0, candidates=4, steps=12, height=1.7, progress=None, dev="cuda"):
    objects, njs, _ = map_objects(zip_path, difficulty, plan=model.config.get("plan", False), dev=dev)
    motion, t0 = play(model, objects, height, njs, steps=steps, seed=seed, dev=dev, candidates=candidates,
                      progress=progress)
    motion = touch_up(from_song_start(motion, t0), objects, njs)[0]
    return clear_crossings(motion, objects, njs)[0]


def save_replay(zip_path, difficulty, motion, out):
    objects, njs, info = map_objects(zip_path, difficulty)
    return summary(write_replay(out, zip_path, difficulty, motion, info, njs, objects))


def play_map(model, zip_path, difficulty, out, seed=0, candidates=4, steps=12, height=1.7, progress=None,
             dev="cuda"):
    motion = play_motion(model, zip_path, difficulty, seed, candidates, steps, height, progress, dev)
    return save_replay(zip_path, difficulty, motion, out)
