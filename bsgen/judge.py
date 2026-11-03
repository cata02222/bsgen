import numpy as np
from scipy.ndimage import gaussian_filter1d

from .player import BLUE, DIM, RED, WALL

FPS = 60
NOTE_DEPTH = 0.9
FIRST_ROW = 0.7522
ROW, LANE = 0.525, 0.6
LANES = np.array([-0.9, -0.3, 0.3, 0.9])
ARROW = np.radians([0, 180, -90, 90, -45, 45, -135, 135, 0])
DOT = 8
CUT_BOX = np.array([0.4, 0.25, 0.5])
CUT_BOX_SHIFT = -0.25
SMALL_BOX = np.array([0.2, 0.2, 0.2])
SEEN = (-0.5, 4.0)
BLADE_POINTS = np.linspace(0, 1, 5, dtype=np.float32)
WALL_BOTTOM = FIRST_ROW - ROW / 2
WALL_SIDE = -1.2


def rotations(sixd):
    a, b = sixd[..., :3], sixd[..., 3:6]
    return np.stack([a, b, np.cross(a, b)], -1)


def tips(motion):
    m = motion.reshape(*motion.shape[:-1], 3, 9)
    hilt = m[..., 1:, :3]
    r = rotations(m[..., 1:, 3:])
    blade = r[..., 2]
    return hilt, blade


class Notes:

    def __init__(self, objects, njs):
        n = objects[objects["kind"] <= BLUE]
        self.time = n["time"].astype(np.float64)
        self.color = n["kind"].astype(np.int64)
        self.line, self.layer = n["line"].astype(np.int64), n["layer"].astype(np.int64)
        self.dir = np.clip(n["dir"].astype(np.int64), 0, DOT)
        w = objects[objects["kind"] == WALL]
        self.wall_time, self.wall_end = w["time"].astype(np.float64), w["end"].astype(np.float64)
        x0 = WALL_SIDE + LANE * w["line"]
        x1 = x0 + LANE * w["width"]
        y0 = WALL_BOTTOM + ROW * w["layer"]
        y1 = y0 + ROW * w["height"]
        self.wall_lo = np.stack([np.minimum(x0, x1), np.minimum(y0, y1)], 1).astype(np.float32)
        self.wall_hi = np.stack([np.maximum(x0, x1), np.maximum(y0, y1)], 1).astype(np.float32)
        self.njs = float(njs)
        a = ARROW[self.dir]
        self.along = np.stack([np.sin(a), np.cos(a), np.zeros_like(a)], 1).astype(np.float32)
        self.across = np.stack([np.cos(a), -np.sin(a), np.zeros_like(a)], 1).astype(np.float32)
        self.centre = np.stack([LANES[np.clip(self.line, 0, 3)], FIRST_ROW + ROW * self.layer], 1).astype(np.float32)

    def __len__(self):
        return len(self.time)


class Result:

    def __init__(self, ids, good, bad, angle, in_wall):
        self.ids, self.good, self.bad, self.angle, self.in_wall = ids, good, bad, angle, in_wall


def _touches(p0, p1, half):
    with np.errstate(divide="ignore", invalid="ignore"):
        d = p1 - p0
        a, b = (-half - p0) / d, (half - p0) / d
        enter = np.minimum(a, b).max(-1)
        leave = np.maximum(a, b).min(-1)
    crosses = (enter < leave) & (((enter > 0) & (enter < 1)) | ((leave > 0) & (leave < 1)))
    inside = (np.abs(p1) < half).all(-1)
    return (crosses | inside).any(-1)


def _local(points, centre, across, along):
    r = points - centre
    return np.stack([(r * across).sum(-1), (r * along).sum(-1), r[..., 2]], -1)


def velocities(hilt, blade):
    vel = gaussian_filter1d(np.gradient(hilt + blade, axis=1) * FPS, 2, axis=1, mode="nearest")
    vel[..., 2] = 0
    return vel / (np.linalg.norm(vel, axis=-1, keepdims=True) + 1e-10)


def simulate(notes, motion, t0, vel=None):
    motion = np.asarray(motion, np.float32)
    C, n = motion.shape[:2]
    hilt, blade = tips(motion)
    vel = (velocities(hilt, blade) if vel is None else vel)[:, 1:]
    times = t0 + np.arange(1, n) / FPS
    ids = np.arange(np.searchsorted(notes.time, times[0] + SEEN[0], "right") if n > 1 else 0,
                    np.searchsorted(notes.time, times[-1] + SEEN[1], "left") if n > 1 else 0)
    K, F = len(ids), n - 1
    rel = notes.time[ids][None] - times[:, None]
    seen = (rel > SEEN[0]) & (rel < SEEN[1])
    good, bad = np.zeros((C, F, 2, K), bool), np.zeros((C, F, 2, K), bool)
    angle = np.full((C, F, K), np.nan, np.float32)
    if K:
        depth = (rel * notes.njs + NOTE_DEPTH).astype(np.float32)
        z = (hilt[..., None, 2] + blade[..., None, 2] * BLADE_POINTS)
        near = np.minimum(CUT_BOX_SHIFT - CUT_BOX[2], -SMALL_BOX[2])
        far = np.maximum(CUT_BOX_SHIFT + CUT_BOX[2], SMALL_BOX[2])
        reach = ((depth + near < z.max()) & (depth + far > z.min()) & seen).any(0)
        t_ids = np.flatnonzero(reach)
        if len(t_ids):
            d = depth[:, t_ids]
            centre = np.concatenate([np.broadcast_to(notes.centre[ids[t_ids]][None], d.shape + (2,)), d[..., None]], -1)
            box = centre[None, :, None, :, None]
            across = notes.across[ids[t_ids]][None, None, None, :, None]
            along = notes.along[ids[t_ids]][None, None, None, :, None]
            prev, cur = ((hilt[:, s, :, None] + blade[:, s, :, None] * BLADE_POINTS[:, None])[:, :, :, None]
                         for s in (slice(0, F), slice(1, F + 1)))
            cut_box = box + np.array([0, 0, CUT_BOX_SHIFT], np.float32)
            hit_cut = _touches(_local(prev, cut_box, across, along), _local(cur, cut_box, across, along), CUT_BOX)
            hit_small = _touches(_local(prev, box, across, along), _local(cur, box, across, along), SMALL_BOX)
            k = ids[t_ids]
            own = np.arange(2)[:, None] == notes.color[k][None]
            right_way = ((vel[:, :, :, None] * notes.along[k]).sum(-1) > 0) | (notes.dir[k] == DOT)
            good[..., t_ids] = hit_cut & own & right_way & seen[None, :, None, t_ids]
            bad[..., t_ids] = hit_small & (~own | ~right_way) & seen[None, :, None, t_ids]
        b = blade[:, 1:][:, :, notes.color[ids]]
        p0, p1 = b[..., 2], (b * notes.along[ids]).sum(-1)
        cos = p0 / np.maximum(np.hypot(p0, p1), 1e-8)
        ang = (np.sign(p1) * np.degrees(np.arccos(np.clip(cos, -1, 1)))).astype(np.float16)
        angle = np.where(seen[None], ang, np.nan).astype(np.float32)
    in_wall = np.zeros((C, F), bool)
    rel_w0 = notes.wall_time[None] - times[:, None]
    rel_w1 = notes.wall_end[None] - times[:, None]
    wseen = (rel_w0 < SEEN[1]) & (rel_w1 > SEEN[0])
    if wseen.any():
        w = wseen.any(0)
        near = rel_w0[:, w] * notes.njs + NOTE_DEPTH
        far = near + (rel_w1[:, w] - rel_w0[:, w]) * notes.njs + NOTE_DEPTH
        z0, z1 = np.minimum(near, far), np.maximum(near, far)
        h = motion[:, 1:, None, :3]
        inside = ((h[..., :2] > notes.wall_lo[w]) & (h[..., :2] < notes.wall_hi[w])).all(-1) \
            & (h[..., 2] > z0) & (h[..., 2] < z1) & wseen[:, w]
        in_wall = inside.any(-1)
    return Result(ids, good, bad, angle, in_wall)


def first_touches(result):
    good, bad = result.good.any(2), result.bad.any(2)
    never = good.shape[1]
    fg = np.where(good.any(1), good.argmax(1), never)
    fb = np.where(bad.any(1), bad.argmax(1), never)
    return fg, fb, never


class Judge:

    CONTEXT = 8
    WALL_FRAMES = 6

    def __init__(self, objects, njs):
        self.notes = Notes(objects, njs)
        self.done = np.zeros(len(self.notes), bool)
        self.times = self.notes.time
        self.last = self.cut = None

    def score(self, cands, t, past):
        C = len(cands)
        start = int(round(t * FPS))
        ctx = min(self.CONTEXT, len(past))
        traj = np.concatenate([np.repeat(np.asarray(past)[len(past) - ctx:][None], C, 0), cands], 1)
        k = start - ctx
        if k < 0:
            traj, k = traj[:, -k:], 0
        skip = max(start - k - 1, 0)
        if traj.shape[1] <= skip + 1:
            self.last = self.cut = None
            return np.zeros(C)
        r = simulate(self.notes, traj, k / FPS)
        r.good, r.bad, r.in_wall = r.good[:, skip:], r.bad[:, skip:], r.in_wall[:, skip:]
        fg, fb, never = first_touches(r)
        open_ = ~self.done[r.ids]
        cut = (fg < fb) & open_
        wrong = (fb <= fg) & (fb < never) & open_
        self.ids, self.last = r.ids, cut | wrong
        in_wall = r.in_wall.sum(1)
        self.cut, self.clean_run = cut, ~wrong.any(1) & (in_wall == 0)
        return cut.sum(1) - wrong.sum(1) - in_wall / self.WALL_FRAMES

    def clean(self, i, lo, hi):
        if self.cut is None:
            return True
        due = ~self.done & (self.times >= lo) & (self.times < hi)
        cut = np.zeros(len(self.done), bool)
        cut[self.ids] = self.cut[i]
        return bool(self.clean_run[i] and cut[due].all())

    def keep(self, i):
        if self.last is not None:
            self.done[self.ids[self.last[i]]] = True


BAND = (-90, 60)
RATE_BEFORE, RATE_AFTER = 24, 25
MISS_AFTER = 0.2


def replay_hits(notes, motion, chunk=120):
    m = np.concatenate([motion[:1], motion])[None].astype(np.float32)
    hilt, blade = tips(m)
    vel = velocities(hilt, blade)
    N = len(notes)
    good, bad, bad_saber = np.full(N, -1), np.full(N, -1), np.full(N, -1)
    band = np.full((N, BAND[1] - BAND[0]), np.nan, np.float32)
    centre = np.round(notes.time * FPS).astype(int)
    for a in range(0, len(motion), chunk):
        r = simulate(notes, m[:, a:a + chunk + 1], (a - 1) / FPS, vel[:, a:a + chunk + 1])
        if not len(r.ids):
            continue
        g, b = r.good[0].any(1), r.bad[0].any(1)
        for hit, out in ((g, good), (b, bad)):
            new = hit.any(0) & (out[r.ids] < 0)
            out[r.ids[new]] = a + hit.argmax(0)[new]
        k = np.flatnonzero(bad[r.ids] >= a)
        bad_saber[r.ids[k]] = r.bad[0, bad[r.ids[k]] - a, :, k].argmax(-1)
        f = a + np.arange(r.angle.shape[1])
        j = f[:, None] - centre[r.ids][None] - BAND[0]
        ok = (j >= 0) & (j < band.shape[1])
        fi, ki = np.nonzero(ok)
        band[r.ids[ki], j[fi, ki]] = r.angle[0, fi, ki]
    return good, bad, bad_saber, band


def note_heights(height):
    return np.array([0.85, 1.4, 1.9]) + min(max((height - 1.8) * 0.5, -0.2), 0.6)


def _dot_angles(blade, vel, f):
    way = vel[max(f - 2, 0):f + 3, :2].mean(0)
    way = np.array([*way, 0.0]) / (np.linalg.norm(way) + 1e-9)
    b = blade[max(f - RATE_BEFORE, 0):f + RATE_AFTER]
    p0, p1 = b[:, 2], b @ way
    ang = np.sign(p1) * np.degrees(np.arccos(np.clip(p0 / np.maximum(np.hypot(p0, p1), 1e-8), -1, 1)))
    cut = min(f, RATE_BEFORE)
    return ang[:cut], ang[cut:]


def note_events(notes, motion, height):
    good, bad, bad_saber, band = replay_hits(notes, motion)
    hilt, blade = tips(motion.astype(np.float64))
    tip = hilt + blade
    vel = np.gradient(tip, axis=0) * FPS
    rows = note_heights(height)
    out = []
    for k in range(len(notes)):
        sec, x, y, c, d = notes.time[k], notes.line[k], notes.layer[k], notes.color[k], notes.dir[k]
        fg, fb = good[k], bad[k]
        if fg >= 0 and (fb < 0 or fg < fb):
            f, kind, saber = fg, 0, c
        elif fb >= 0:
            f, kind, saber = fb, 1, bad_saber[k]
        else:
            out.append((sec, x, y, c, d, 2, sec + MISS_AFTER, 0, 0.0, 0.0, 0.0, 0.0, np.zeros(3), np.zeros(3)))
            continue
        j = f - int(round(sec * FPS)) - BAND[0]
        pre, post = band[k, max(j - RATE_BEFORE, 0):max(j, 0)], band[k, max(j, 0):j + RATE_AFTER]
        if d == DOT:
            pre, post = _dot_angles(blade[:, c], vel[:, c], f)
        before = np.clip(np.nanmin(pre) / -100, 0, 1) if np.isfinite(pre).any() else 0.0
        after = np.clip(np.nanmax(post) / 60, 0, 1) if np.isfinite(post).any() else 0.0
        h = hilt[f, saber]
        v = vel[f, saber]
        normal = np.cross(tip[f, saber] - h, v)
        normal /= np.linalg.norm(normal) + 1e-9
        centre = np.array([LANES[min(max(x, 0), 3)], rows[min(max(y, 0), 2)], (sec - f / FPS) * notes.njs + 0.65])
        speed = float(np.linalg.norm(v))
        out.append((sec, x, y, c, d, kind, f / FPS, saber, float(before), float(after),
                    abs(float(normal @ (centre - h))), speed, v / (speed + 1e-9), normal))
    return out


def replay_score(events):
    total, mult, progress = 0, 1, 0
    for e in sorted(events, key=lambda e: e[6]):
        if e[5] == 0:
            cut = round(70 * e[8]) + round(30 * e[9]) + round(15 * (1 - min(e[10] / 0.3, 1)))
            total += cut * mult
            progress += 1
            if mult < 8 and progress >= mult * 2:
                mult, progress = mult * 2, 0
        else:
            mult, progress = max(mult // 2, 1), 0
    return total


assert DIM == 27 and RED == 0
