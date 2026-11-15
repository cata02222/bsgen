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
                      [d.bombs["beat"].max() if len(d.bombs) else 0] +
                      [d.chains["tail_beat"].max() if len(d.chains) else 0] +
                      [(d.walls["beat"] + d.walls["duration"]).max() if len(d.walls) else 0])) + 4
    beat_times = d.tempo.to_seconds(np.arange(n_beats, dtype=float)) + info.song_offset
    plan = swing_plan(d.notes, dev, d.bombs) if plan else None
    seconds = lambda b: d.tempo.to_seconds(np.asarray(b, float)) + info.song_offset
    return objects_from_map(d.notes, d.bombs, d.walls, beat_times, plan, d.chains, seconds), d.njs or 18.0, info


def from_song_start(motion, t0):
    n0 = int(round(t0 * HZ))
    return np.concatenate([np.repeat(motion[:1], n0, 0), motion]) if n0 >= 0 else motion[-n0:]


def map_environment(zip_path):
    try:
        with beatmap.open_map(zip_path) as z:
            name = next(n for n in z.namelist() if n.lower().rsplit("/", 1)[-1] == "info.dat")
            d = json.loads(z.read(name).decode("utf-8-sig"))
        env = d.get("_environmentName") or d.get("environmentNames", [None])[0] or d.get("environmentName")
        return env if isinstance(env, str) and env else "DefaultEnvironment"
    except (StopIteration, ValueError, KeyError, OSError, IndexError):
        return "DefaultEnvironment"


def write_replay(path, zip_path, difficulty, motion, info, njs, objects, player="bsgen bot", characteristic="Standard"):
    objects, _, _ = map_objects(zip_path, difficulty, characteristic)
    events = judge.note_events(judge.Notes(objects, njs), motion, BSOR_HEIGHT)
    m = motion.reshape(-1, 3, 9).astype(np.float64)
    quat = Rotation.from_matrix(judge.rotations(m[..., 3:]).reshape(-1, 3, 3)).as_quat().reshape(-1, 3, 4)
    meta = {"player": player, "hash": bsor.map_hash(zip_path, info), "song": info.song_name,
            "mapper": info.mapper, "difficulty": difficulty, "score": judge.replay_score(events),
            "environment": map_environment(zip_path), "height": BSOR_HEIGHT}
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


WALL_EASE = 0.3
LOWEST_HEAD = 0.9


def _wall_push(notes, motion, margin):
    n = len(motion)
    push = np.zeros((n, 2), np.float32)
    if not len(notes.wall_time):
        return push
    t = np.arange(n) / HZ
    level = []
    for w in range(len(notes.wall_time)):
        a, b = notes.wall_time[w], notes.wall_end[w]
        f0, f1 = max(int((a - 1.0) * HZ), 0), min(int((b + 0.5) * HZ) + 1, n)
        if f0 >= f1:
            continue
        near = (a - t[f0:f1]) * notes.njs + judge.NOTE_DEPTH
        far = near + (b - a) * notes.njs + judge.NOTE_DEPTH
        z = motion[f0:f1, 2]
        on = (z > np.minimum(near, far) - 0.1) & (z < np.maximum(near, far) + 0.1)
        level += [(f0 + i, w) for i in np.flatnonzero(on)]
    by_frame = {}
    for f, w in level:
        by_frame.setdefault(f, []).append(w)
    for f, ws in by_frame.items():
        lo, hi = notes.wall_lo[ws] - margin, notes.wall_hi[ws] + margin
        h = motion[f, :2]
        if not ((h > lo) & (h < hi)).all(1).any():
            continue
        tries = [np.array([x, h[1]]) for x in np.r_[lo[:, 0], hi[:, 0]]]
        tries += [np.array([h[0], max(lo[:, 1].min(), LOWEST_HEAD)])]
        tries += [np.array([x, y]) for x in np.r_[lo[:, 0], hi[:, 0]] for y in lo[:, 1]]
        ok = [p for p in tries if not ((p > lo) & (p < hi)).all(1).any()]
        if ok:
            push[f] = min(ok, key=lambda p: np.linalg.norm(p - h)) - h
    return push


def clear_walls(motion, objects, njs, margin=judge.WALL_MARGIN, rounds=6):
    from scipy.ndimage import gaussian_filter1d, maximum_filter1d, minimum_filter1d
    notes = judge.Notes(objects, njs)
    motion = motion.copy()
    k = int(WALL_EASE * HZ)
    moved = np.zeros(len(motion), bool)
    for _ in range(rounds):
        push = _wall_push(notes, motion, margin)
        if not push.any():
            break
        hold = np.zeros_like(push)
        for j in range(2):
            hi, lo = maximum_filter1d(push[:, j], k), minimum_filter1d(push[:, j], k)
            hold[:, j] = np.where(hi >= -lo, hi, lo)
        off = gaussian_filter1d(hold, k / 4, axis=0)
        motion[:, :2] += off
        moved |= np.abs(off).sum(1) > 0.01
    return motion, int(moved.sum())


AIM_NODE, AIM_BEND, AIM_HOLD, AIM_BEND_TURN, AIM_HOLD_TURN, AIM_ROUNDS = 6, 1.0, 0.5, 0.1, 0.3, 4


def _aim_cuts_of(notes, motion, events, height):
    good = [(k, e) for k, e in enumerate(events) if e[5] == 0]
    if not good:
        return None
    k = np.array([k for k, _ in good])
    f = np.array([int(round(e[6] * HZ)) for _, e in good])
    h = np.array([int(e[7]) for _, e in good])
    hilt, blade = judge.tips(motion.astype(np.float64))
    tip = hilt + blade
    vel = np.gradient(tip, axis=0) * HZ
    rows = judge.note_heights(height)
    y = np.clip(notes.layer[k], 0, 2)
    c = np.stack([notes.centre[k, 0], rows[y] + notes.centre[k, 1] - (judge.FIRST_ROW + judge.ROW * notes.layer[k]),
                  (notes.time[k] - f / HZ) * notes.njs + 0.65], 1)
    H, B, V = hilt[f, h], blade[f, h], vel[f, h]

    def plane(b):
        n = np.cross(b, V)
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
        return n, (n * (c - H)).sum(1)
    n, s = plane(B)
    g = np.zeros_like(n)
    for i in range(3):
        w = np.zeros(3)
        w[i] = 1e-3
        g[:, i] = (plane(B + np.cross(w, B))[1] - s) / 1e-3
    return f, h, n, g, s, k


def _aim_offset(N, f, n, g, s):
    import scipy.sparse as sps
    import scipy.sparse.linalg as spl
    from scipy.ndimage import gaussian_filter1d
    G, U = AIM_NODE, 6
    M = N // G + 2
    i, u = f // G, (f % G) / G
    coef = np.concatenate([n, -g], 1)
    r = np.repeat(np.arange(len(f)), 2 * U)
    col = np.concatenate([U * i[:, None] + np.arange(U), U * (i + 1)[:, None] + np.arange(U)], 1).ravel()
    val = np.concatenate([(1 - u)[:, None] * coef, u[:, None] * coef], 1).ravel()
    A = sps.csr_matrix((val, (r, col)), shape=(len(f), U * M))
    d2 = sps.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(M - 2, M))
    bend = np.array([AIM_BEND] * 3 + [AIM_BEND_TURN] * 3) ** 2
    hold = np.array([AIM_HOLD] * 3 + [AIM_HOLD_TURN] * 3) ** 2
    Q = A.T @ A + sps.kron(d2.T @ d2, sps.diags(bend)) + sps.kron(sps.identity(M), sps.diags(hold))
    x = spl.spsolve(Q.tocsc(), A.T @ s).reshape(M, U)
    t = np.arange(N)
    w = ((t % G) / G)[:, None]
    return gaussian_filter1d((1 - w) * x[t // G] + w * x[t // G + 1], 2, axis=0)


def _aim_apply(motion, off, keep):
    out = motion.copy()
    for h in (0, 1):
        c = 9 * (1 + h)
        o = off[:, h] * keep[:, None]
        out[:, c:c + 3] += o[:, :3]
        turn = Rotation.from_rotvec(o[:, 3:])
        for a in (3, 6):
            out[:, c + a:c + a + 3] = turn.apply(motion[:, c + a:c + a + 3].astype(np.float64))
    return out


def aim_cuts(motion, objects, njs, height=BSOR_HEIGHT):
    notes = judge.Notes(objects, njs)
    hits = judge.replay_hits(notes, motion)
    cuts = _aim_cuts_of(notes, motion, judge.note_events(notes, motion, height, hits), height)
    if cuts is None:
        return motion, 0
    f, h, n, g, s, k = cuts
    off = np.zeros((len(motion), 2, 6))
    for hand in (0, 1):
        m = h == hand
        if m.any():
            off[:, hand] = _aim_offset(len(motion), f[m], n[m], g[m], s[m])

    def ok(hit):
        return (hit[0] >= 0) & ((hit[1] < 0) | (hit[0] < hit[1]))

    was = ok(hits)
    keep = np.ones(len(motion))
    for _ in range(AIM_ROUNDS):
        out = _aim_apply(motion, off, keep)
        broke = np.flatnonzero(was & ~ok(judge.replay_hits(notes, out)))
        if not len(broke):
            return out.astype(motion.dtype), int(len(f) - (keep[f] < 1).sum())
        w = int(0.5 * HZ)
        for b in broke:
            x = np.clip(np.abs(np.arange(len(motion)) - notes.time[b] * HZ) / w - 1, 0, 1)
            keep = np.minimum(keep, x * x * (3 - 2 * x))
    return motion, 0


def play_motion(model, zip_path, difficulty, seed=0, candidates=4, steps=None, height=1.7, progress=None, dev="cuda"):
    objects, njs, _ = map_objects(zip_path, difficulty, plan=model.config.get("plan", False), dev=dev)
    motion, t0 = play(model, objects, height, njs, steps=steps, seed=seed, dev=dev, candidates=candidates,
                      progress=progress)
    motion = touch_up(from_song_start(motion, t0), objects, njs)[0]
    motion = clear_crossings(motion, objects, njs)[0]
    motion = clear_walls(motion, objects, njs)[0]
    return aim_cuts(motion, objects, njs)[0]


def save_replay(zip_path, difficulty, motion, out):
    objects, njs, info = map_objects(zip_path, difficulty)
    return summary(write_replay(out, zip_path, difficulty, motion, info, njs, objects))


def play_map(model, zip_path, difficulty, out, seed=0, candidates=4, steps=None, height=1.7, progress=None,
             dev="cuda"):
    motion = play_motion(model, zip_path, difficulty, seed, candidates, steps, height, progress, dev)
    return save_replay(zip_path, difficulty, motion, out)
