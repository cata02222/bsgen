import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

from .model import RECORDING, recording_stream

HZ = 60
P, F = 60, 120
LOOK_BACK, LOOK_AHEAD = 1.0, 1.5
MAX_OBJECTS = 128
DIM = 27

RED, BLUE, BOMB, WALL = 0, 1, 2, 3
CHAIN_HEAD, CHAIN_LINK = 6, 7
NO_DIR = 9
UNKNOWN, JOIN, NEW_FORE, NEW_BACK = 0, 1, 2, 3
PLAY_ACCURACY = 0.975
OBJECT_DTYPE = np.dtype([("time", "f8"), ("end", "f8"), ("kind", "i1"), ("line", "i1"), ("layer", "i1"),
                         ("dir", "i1"), ("width", "i1"), ("height", "i1"), ("plan", "i1"),
                         ("offset_x", "f4"), ("offset_y", "f4"), ("scoring_type", "i1")])
REPLAY_DTYPE = np.dtype([("start", "i8"), ("frames", "i4"), ("t0", "f8"), ("map", "i4"), ("height", "f4"),
                         ("njs", "f4"), ("stars", "f4"), ("val", "?")])


def objects_from_map(notes, bombs, walls, beat_times, plan=None, chains=None, seconds=None):
    out = np.zeros(len(notes) + len(bombs) + len(walls), OBJECT_DTYPE)
    n, b = len(notes), len(bombs)
    out["time"][:n], out["end"][:n] = notes["time"], notes["time"]
    if plan is not None:
        out["plan"][:n] = plan
    out["kind"][:n], out["line"][:n], out["layer"][:n] = notes["color"], notes["line"], notes["layer"]
    out["dir"][:n] = notes["cut_dir"]
    o = out[n:n + b]
    o["time"], o["end"], o["kind"], o["line"], o["layer"], o["dir"] = bombs["time"], bombs["time"], BOMB, \
        bombs["line"], bombs["layer"], NO_DIR
    w = out[n + b:]
    w["time"] = walls["time"]
    w["end"] = np.interp(walls["beat"] + walls["duration"], np.arange(len(beat_times)), beat_times)
    w["kind"], w["line"], w["layer"], w["dir"] = WALL, walls["line"], walls["layer"], NO_DIR
    w["width"], w["height"] = walls["width"], walls["height"]
    links = []
    for ch in (() if chains is None else chains):
        if seconds is None:
            raise ValueError("chains require a beat-to-song-seconds converter")
        head_time = float(seconds(np.array([ch["beat"]]))[0])
        heads = (np.abs(out["time"] - head_time) < 1e-3) & (out["kind"] == ch["color"]) \
            & (out["line"] == ch["line"]) & (out["layer"] == ch["layer"]) & (out["dir"] == ch["dir"])
        out["scoring_type"][heads] = CHAIN_HEAD
        n = int(ch["slices"]) - 1
        if n < 1:
            continue
        u = np.arange(1, n + 1) / n
        o = np.zeros(n, OBJECT_DTYPE)
        o["time"] = seconds(ch["beat"] + (ch["tail_beat"] - ch["beat"]) * u)
        o["end"] = o["time"]
        o["kind"], o["line"], o["layer"] = ch["color"], ch["line"], ch["layer"]
        o["dir"], o["plan"], o["scoring_type"] = 8, JOIN, CHAIN_LINK
        tail = np.array([float(ch["tail_line"]) - ch["line"], float(ch["tail_layer"]) - ch["layer"]])
        angle = np.radians([0, 180, -90, 90, -45, 45, -135, 135, 0][int(ch["dir"])])
        control = np.array([np.sin(angle), np.cos(angle)]) * np.linalg.norm(tail) / 2
        squish = float(ch["squish"]) if ch["squish"] >= 0.001 else 1.0
        v = (u * squish)[:, None]
        xy = 2 * (1 - v) * v * control + v * v * tail
        o["offset_x"], o["offset_y"] = xy.T
        links.append(o)
    if links:
        out = np.concatenate([out, *links])
    return out[np.argsort(out["time"], kind="stable")]


def as_objects(o):
    if o.dtype == OBJECT_DTYPE:
        return o
    out = np.zeros(len(o), OBJECT_DTYPE)
    for k in o.dtype.names:
        out[k] = o[k]
    return out


def window_objects(objects, t, n_future=F):
    a = int(np.searchsorted(objects["time"], t - LOOK_BACK))
    lo = int(np.searchsorted(objects["time"], t - LOOK_BACK - 30.0))
    hi = int(np.searchsorted(objects["time"], t + n_future / HZ + LOOK_AHEAD, "right"))
    o = objects[lo:hi]
    o = o[o["end"] >= t - LOOK_BACK]
    before = [_nearest(objects["kind"][max(a - 256, 0):a][::-1], c) for c in (RED, BLUE)]
    after = [_nearest(objects["kind"][hi:hi + 256], c) for c in (RED, BLUE)]
    before = [objects[a - 1 - i] for i in before if i is not None]
    after = [objects[hi + i] for i in after if i is not None]
    return np.concatenate([np.array(before, OBJECT_DTYPE), o, np.array(after, OBJECT_DTYPE)])[:MAX_OBJECTS]


def _nearest(kinds, c):
    i = np.flatnonzero(kinds == c)
    return int(i[0]) if len(i) else None


def object_features(o, t):
    k = len(o)
    ints = np.zeros((MAX_OBJECTS, 7), np.int64)
    times = np.zeros((MAX_OBJECTS, 2), np.float32)
    mask = np.ones(MAX_OBJECTS, bool)
    ints[:k, 0] = o["kind"]
    ints[:k, 1] = np.clip(np.rint(o["line"] + o["offset_x"]).astype(np.int64) + 4, 0, 11)
    ints[:k, 2] = np.clip(np.rint(o["layer"] + o["offset_y"]), 0, 4)
    ints[:k, 3] = np.clip(o["dir"], 0, NO_DIR)
    ints[:k, 4] = np.clip(o["width"], 0, 7)
    ints[:k, 5] = np.clip(o["height"], 0, 7)
    ints[:k, 6] = np.clip(o["plan"], 0, 3)
    times[:k, 0] = np.clip(o["time"] - t, -30.0, 30.0)
    times[:k, 1] = np.clip(o["end"] - t, -30.0, 30.0)
    mask[:k] = False
    return ints, times, mask


def hard_spans(objects):
    notes = objects[objects["kind"] <= BLUE]
    walls = objects[(objects["kind"] == WALL) & (objects["line"] < 3) & (objects["line"] + objects["width"] > 1)]
    side, dots = notes[np.isin(notes["dir"], [2, 3])], notes[notes["dir"] == 8]
    spans = [np.stack([a["time"], a["end"]], 1) for a in (side, dots, walls)]
    nt = np.unique(notes["time"])
    gap = np.flatnonzero(np.diff(nt) > 1.0)
    return spans + [np.stack([nt[gap] + 0.5, nt[gap + 1]], 1).astype(np.float32)]


def accuracy_input(acc):
    return (1.0 - acc) * 100.0


def quiet_spans(objects, min_gap=2.5):
    t = np.unique(objects["time"][objects["kind"] <= BOMB])
    g = np.flatnonzero(np.diff(t) >= min_gap)
    return np.stack([t[g], t[g + 1]], 1)


class MotionData(torch.utils.data.Dataset):

    def __init__(self, root, val=False, size=10 ** 8, hard=0.0):
        root = Path(root)
        reps = np.load(root / "replays.npy")
        self.reps = reps[reps["val"] == val]
        self.reps = self.reps[self.reps["frames"] > P + F + 1]
        self.objects = np.load(root / "objects.npy")
        self.maps = np.load(root / "maps.npy")
        self.root, self.val, self.size = root, val, size
        self.motion = None
        w = (self.reps["frames"] - P - F).astype(np.float64)
        self.cum = np.cumsum(w / w.sum())
        self.hard = hard
        if hard > 0:
            rows = [[], [], [], []]
            for i, r in enumerate(self.reps):
                mp = self.maps[r["map"]]
                for k, sp in enumerate(hard_spans(self.objects[mp["start"]: mp["start"] + mp["count"]])):
                    f = ((sp - r["t0"]) * HZ).astype(np.int64)
                    f = f[(f[:, 0] >= P) & (f[:, 1] < r["frames"] - F)]
                    rows[k].append(np.column_stack([np.full(len(f), i), f]))
            self.spans = [np.concatenate(x) for x in rows]

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        if self.motion is None:
            self.motion = np.memmap(self.root / "motion.f16", np.float16, "r").reshape(-1, DIM)
        rng = np.random.default_rng(i if self.val else None)
        if self.hard > 0 and rng.random() < self.hard:
            sp = self.spans[int(rng.integers(len(self.spans)))]
            i, a, b = sp[int(rng.integers(len(sp)))]
            r = self.reps[i]
            s = int(np.clip(rng.integers(a, b + 1) - rng.integers(0, F), P, r["frames"] - F - 1))
        else:
            r = self.reps[min(int(np.searchsorted(self.cum, rng.random())), len(self.reps) - 1)]
            s = int(rng.integers(P, r["frames"] - F))
        m = np.asarray(self.motion[r["start"] + s - P: r["start"] + s + F], np.float32)
        t = float(r["t0"] + s / HZ)
        mp = self.maps[r["map"]]
        ints, times, mask = object_features(window_objects(self.objects[mp["start"]: mp["start"] + mp["count"]], t), t)
        cond = np.array([r["height"], r["njs"]], np.float32)
        return dict(past=m[:P], future=m[P:], ints=ints, times=times, mask=mask, cond=cond)


def fourier(x, n=16, lo=0.5, hi=200.0):
    f = torch.exp(torch.linspace(math.log(lo), math.log(hi), n, device=x.device))
    a = x[..., None] * f
    return torch.cat([a.sin(), a.cos()], -1)


class Player(nn.Module):
    def __init__(self, d=384, layers=8, heads=6, stats=None, plan=False, conds=2):
        super().__init__()
        self.config = dict(d=d, layers=layers, heads=heads, plan=plan, conds=conds)
        self.past_in, self.fut_in = nn.Linear(DIM, d), nn.Linear(DIM, d)
        self.frame_time = nn.Linear(32, d)
        self.emb = nn.ModuleList([nn.Embedding(n, d) for n in (4, 12, 5, NO_DIR + 1, 8, 8, 4)[:7 if plan else 6]])
        self.obj_time = nn.Linear(64, d)
        self.cond = nn.Sequential(nn.Linear(conds + 32, d), nn.GELU(), nn.Linear(d, d))
        self.flow_time = nn.Sequential(nn.Linear(32, d), nn.GELU(), nn.Linear(d, d))
        self.segment = nn.Embedding(4, d)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, activation="gelu", batch_first=True,
                                           norm_first=True)
        self.body = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.body_dtype = torch.float32
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, DIM)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        for k in ("mean", "std", "dstd"):
            self.register_buffer(k, torch.as_tensor(stats[k], dtype=torch.float32) if stats else torch.ones(DIM))

    def forward(self, past, xt, t, ints, times, mask, cond):
        B, dev = past.shape[0], past.device
        ft = torch.arange(-P, F, device=dev, dtype=torch.float32) / HZ
        ft = self.frame_time(fourier(ft))
        tt = self.flow_time(fourier(t * 10.0, lo=0.1, hi=20.0))[:, None]
        seg = self.segment.weight
        c = self.cond(torch.cat([cond, fourier(t * 10.0, lo=0.1, hi=20.0)], -1))[:, None] + seg[0]
        h_past = self.past_in((past - self.mean) / self.std) + ft[:P] + seg[1]
        h_obj = sum(e(ints[..., i]) for i, e in enumerate(self.emb)) \
            + self.obj_time(torch.cat([fourier(times[..., 0]), fourier(times[..., 1])], -1)) + seg[2]
        h_fut = self.fut_in(xt) + ft[P:] + tt + seg[3]
        h = torch.cat([c, h_past, h_obj, h_fut], 1)
        pad = torch.cat([torch.zeros(B, 1 + P, dtype=torch.bool, device=dev), mask,
                         torch.zeros(B, F, dtype=torch.bool, device=dev)], 1)
        h = self.body(h.to(self.body_dtype), src_key_padding_mask=pad)
        return self.out(self.norm(h[:, -F:].float()))

    def loss(self, batch, generator=None):
        past, fut = batch["past"], batch["future"]
        x1 = (fut - past[:, -1:]) / self.dstd
        x0 = torch.randn(x1.shape, device=x1.device, generator=generator)
        t = torch.sigmoid(torch.randn(len(x1), device=x1.device, generator=generator))
        xt = (1 - t[:, None, None]) * x0 + t[:, None, None] * x1
        v = self(past, xt, t, batch["ints"], batch["times"], batch["mask"], batch["cond"])
        return Fn.mse_loss(v.float(), (x1 - x0).float())

    @torch.no_grad()
    def sample(self, past, ints, times, mask, cond, steps=24, generator=None, noise=None):
        x = torch.randn(past.shape[0], F, DIM, device=past.device, generator=generator) if noise is None else noise
        for i in range(steps):
            t = torch.full((len(x),), i / steps, device=x.device)
            x = x + self(past, x, t, ints, times, mask, cond).float() / steps
        return x * self.dstd + past[:, -1:]


class Sampler:

    def __init__(self, model, steps=12):
        self.model, self.steps, self.graphs = model, steps, {}
        self.cuda = model.out.weight.is_cuda
        if self.cuda:
            with RECORDING:
                model.body.to(torch.bfloat16)
                model.body_dtype = torch.bfloat16

    @torch.no_grad()
    def __call__(self, past, ints, times, mask, cond, noise):
        args = [a.expand(len(noise), *a.shape[1:]) for a in (past, ints, times, mask, cond)]
        if not self.cuda:
            return self.model.sample(*args, steps=self.steps, noise=noise)
        if len(noise) not in self.graphs:
            self.graphs[len(noise)] = self._capture(*args, noise)
        graph, ins, out = self.graphs[len(noise)]
        for buf, a in zip(ins, (*args, noise)):
            buf.copy_(a)
        graph.replay()
        return out.clone()

    def _capture(self, *args):
        with RECORDING:
            ins = [a.clone() for a in args]
            out = torch.empty_like(ins[-1])
            run = lambda: out.copy_(self.model.sample(*ins[:-1], steps=self.steps, noise=ins[-1]))
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(2):
                    run()
            torch.cuda.current_stream().wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=recording_stream(), capture_error_mode="thread_local"):
                run()
            torch.cuda.current_stream().synchronize()
            return graph, ins, out


def motion_stats(data, n=4096):
    idx = np.random.default_rng(0).integers(0, len(data), n)
    s = [data[int(i)] for i in idx]
    allm = np.concatenate([np.concatenate([x["past"], x["future"]]) for x in s])
    d = np.concatenate([x["future"] - x["past"][-1:] for x in s])
    return dict(mean=allm.mean(0), std=allm.std(0) + 1e-3, dstd=d.std(0) + 1e-3)


def orthonormal(m):
    m = m.reshape(*m.shape[:-1], 3, 9).copy()
    a, b = m[..., 3:6], m[..., 6:9]
    a /= np.linalg.norm(a, axis=-1, keepdims=True)
    b -= (a * b).sum(-1, keepdims=True) * a
    b /= np.linalg.norm(b, axis=-1, keepdims=True)
    return m.reshape(*m.shape[:-2], DIM)


TURN_RATE, TURN_CAP = [0.25, 1.0, 2.0, 3.0, 4.25, 6.0], [1800, 3900, 3900, 5000, 5100, 8300]


def _rot(six):
    a, b = six[..., :3], six[..., 3:6]
    return np.stack([a, b, np.cross(a, b)], -1)


def limit_turns(fut, prev, cap):
    from scipy.spatial.transform import Rotation
    out = fut.copy()
    step = np.radians(cap) / HZ
    for h in (1, 2):
        cols = slice(9 * h + 3, 9 * h + 9)
        R = _rot(out[..., cols].astype(np.float64))
        last = np.repeat(_rot(prev[cols].astype(np.float64))[None], len(R), 0)
        for f in range(R.shape[1]):
            rel = np.einsum("cji,cjk->cik", last, R[:, f])
            ang = np.arccos(np.clip((np.trace(rel, axis1=1, axis2=2) - 1) / 2, -1, 1))
            over = ang > step[f, h - 1]
            if over.any():
                rv = Rotation.from_matrix(rel[over]).as_rotvec() * (step[f, h - 1] / ang[over])[:, None]
                R[over, f] = last[over] @ Rotation.from_rotvec(rv).as_matrix()
            last = R[:, f]
        out[..., cols] = np.concatenate([R[..., 0], R[..., 1]], -1)
    return out


OFF_CUT = 70.0
_ARROW = np.array([(0, 1), (0, -1), (-1, 0), (1, 0), (-1, 1), (1, 1), (-1, -1), (1, -1)], np.float32)
_ARROW /= np.linalg.norm(_ARROW, axis=1, keepdims=True)


MORE_ROUNDS = 4


def off_cuts(cands, past, objects, t):
    n = objects[(objects["kind"] <= BLUE) & (objects["time"] >= t) & (objects["time"] < t + F / HZ)]
    if not len(n):
        return np.zeros(len(cands), int)
    full = np.concatenate([np.repeat(np.asarray(past)[-3:][None], len(cands), 0), cands], 1)
    out = np.zeros(len(cands), int)
    for r in n:
        f = 3 + int(round((float(r["time"]) - t) * HZ))
        if f + 2 >= full.shape[1]:
            continue
        c = 9 * (1 + int(r["kind"]))
        seg = full[:, [f - 2, f + 2], c:c + 9]
        a, b = seg[..., 3:6], seg[..., 6:9]
        tip = seg[..., :3] + np.cross(a, b)
        v = (tip[:, 1, :2] - tip[:, 0, :2])
        sp = np.linalg.norm(v, axis=1) + 1e-6
        d = int(r["dir"])
        if d < 8:
            off = np.degrees(np.arccos(np.clip(v @ _ARROW[d] / sp, -1, 1))) > OFF_CUT
        elif r["plan"] in (NEW_FORE, NEW_BACK):
            off = (v[:, 1] > 0) == (r["plan"] == NEW_FORE)
        else:
            continue
        out += off & (sp * HZ / 4 > 3.0)
    return out


def turn_caps(objects, t):
    ft = t + np.arange(F) / HZ
    caps = []
    for c in (RED, BLUE):
        nt = objects["time"][objects["kind"] == c]
        rate = (np.searchsorted(nt, ft + 1.0) - np.searchsorted(nt, ft - 1.0)) / 2.0
        caps.append(np.interp(rate, TURN_RATE, TURN_CAP))
    return np.stack(caps, 1)


def play(model, objects, height=1.7, njs=18.0, start_pose=None, steps=12, seed=0, dev="cuda", t_end=None,
         warmup=2.0, recenter=0.3, candidates=1, judge=None, progress=None, accuracy=PLAY_ACCURACY, turns=True):
    g = torch.Generator(device=dev).manual_seed(seed)
    sampler = Sampler(model, steps)
    if candidates > 1 and judge is None:
        from .judge import Judge
        judge = Judge(objects, njs)
    t0 = float(objects["time"].min()) - warmup
    t_end = t_end or float(objects["end"].max()) + 1.0
    pose = model.mean.cpu().numpy() if start_pose is None else np.asarray(start_pose, np.float32)
    home = pose[:3].copy()
    walls = objects[objects["kind"] == WALL]
    walls = walls[(walls["line"] < 3) & (walls["line"] + walls["width"] > 1)]
    ease = np.linspace(0, 1, F, dtype=np.float32)[:, None]
    motion = [np.repeat(pose[None], P, 0)]
    t = t0 + P / HZ
    while t < t_end:
        past = np.concatenate(motion)[-P:]
        ints, times, mask = object_features(window_objects(objects, t), t)
        cond = [height, njs, accuracy_input(accuracy)][:model.config.get("conds", 2)]
        inputs = [torch.as_tensor(a)[None].to(dev) for a in (past, ints, times, mask, np.array(cond, np.float32))]
        noise = torch.randn(candidates, F, DIM, device=dev, generator=g)
        near_wall = np.any((walls["time"] < t + F / HZ + 1.0) & (walls["end"] > t - 1.0))

        def draw(noise):
            fut = orthonormal(sampler(*inputs, noise).float().cpu().numpy())
            if recenter and not near_wall:
                shift = recenter * (fut[:, :, :3].mean(1) - home)
                for j in range(3):
                    fut[:, :, 9 * j:9 * j + 3] -= ease * shift[:, None]
            return limit_turns(fut, orthonormal(past[-1]), turn_caps(objects, t)) if turns else fut

        if candidates > 1:
            fut = draw(noise[:1])
            judge.score(fut, t, past)
            if not judge.clean(0, t - 0.3, t + F / HZ - 0.2) or off_cuts(fut, past, objects, t)[0]:
                fut = np.concatenate([fut, draw(noise[1:])])
                while True:
                    sc = judge.score(fut, t, past)
                    top = np.flatnonzero(sc >= sc.max() - 1e-6)
                    best = int(top[np.argmin(off_cuts(fut[top], past, objects, t))])
                    if len(fut) >= candidates * MORE_ROUNDS or judge.clean(best, t - 0.3, t + F / HZ - 0.2):
                        break
                    more = torch.randn(candidates, F, DIM, device=dev, generator=g)
                    fut = np.concatenate([fut, draw(more)])
            else:
                best = 0
            judge.keep(best)
        else:
            fut, best = draw(noise), 0
        motion.append(fut[best])
        t += F / HZ
        if progress:
            progress(min(t, t_end), t_end)
    motion = np.concatenate(motion).astype(np.float32)
    if t0 < 0:
        motion, t0 = motion[int(np.ceil(-t0 * HZ)):], t0 + np.ceil(-t0 * HZ) / HZ
    return motion, t0
