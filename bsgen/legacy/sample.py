import contextlib
import json
import threading
from pathlib import Path

import numpy as np
import torch

from . import tokens
from .data import beat_mel, beat_to_sec, overview_rel, song_overview
from .model import Generator, GraphStep
from .tokens import ARC, BOMB, CONTEXT, NOTE, OBJ_ARCS, OBJ_BOMBS, OBJ_WALLS, SWING, TICKS, VOCAB, WALL, WINDOW

FOREHAND_DIRS = {1, 6, 7, 2, 3, 8}
BACKHAND_DIRS = {0, 4, 5, 2, 3, 8}
OPENING_DIRS = {1, 6, 7, 8}
DIR_ANGLE = {0: 180, 1: 0, 2: 270, 3: 90, 4: 225, 5: 135, 6: 315, 7: 45}
MAX_STACK = 4
MAX_GAP = 96
MAX_INTRO = 48
BEHIND_WALL_M = 5.5


def behind_wall_ticks(njs, beat_times):
    spb = float(np.median(np.diff(beat_times))) if len(beat_times) > 1 else 0.5
    return max(int(round(BEHIND_WALL_M / max(njs, 1.0) / spb * TICKS)), 1)


def _wall_cells(shape):
    x, y, w, h = tokens.shape_of(shape)
    lanes = range(max(x, 0), min(x + w, 4))
    v2 = range(3) if y == 0 and h >= 3 else range(2, 3)
    layers = set(range(max(y, 0), min(y + h, 3))) | set(v2)
    return frozenset(ly * 4 + lx for lx in lanes for ly in layers), frozenset(ly * 4 + lx for lx in lanes for ly in range(3))


WALL_CELLS = [_wall_cells(sh) for sh in range(tokens.N_SHAPES)]
WALL_CAP, FEW_NOTES = 105, 4
BOMB_CAP_STARS, BOMB_CAP = [2, 6, 10, 14], [64, 178, 300, 474]
BOMB_RATIO = 1.0
MAX_LOOP = 16
LOOP_PERIODS = 32
LOOP_MATCH = 0.85
LOOP_PENALTY = 4.0
VARIETY_BEATS, VARIETY_NOTES, VARIETY_FLOOR, VARIETY_PENALTY = 16, 32, 12, 0.75
DRY_BASE, DRY_ALLOWED, DRY_WINDOW, DRY_GROW = 1.75, 6, 512, 8
XTC_THRESHOLD = 0.1
NO_NOTE = 108
ARC_PUSH = 3.0


class LoopGuard:

    def __init__(self, max_loop=MAX_LOOP, dry=0.0):
        self.limit = max_loop * TICKS
        self.dry = dry
        self.sym, self.ticks, self.codes = [], [], []
        self.expect = []
        self.overused = []
        self.repeats, self.stale = [], False

    def add(self, tick, hand, cell, d, check=True):
        gap = tick - self.ticks[-1] if self.ticks else 0
        self.sym.append((hand, cell, d, gap))
        self.ticks.append(tick)
        self.codes.append(((hand * 12 + cell) * 9 + d) * 10 ** 6 + gap)
        if len(self.sym) > max(4 * LOOP_PERIODS, DRY_WINDOW) + 64:
            del self.sym[:64], self.ticks[:64], self.codes[:64]
        self.stale = self.dry > 0
        self.expect, self.overused = [], []
        if check:
            self.check()

    def check(self):
        self.expect, self.overused = [], []
        if self.limit <= 0 or not self.ticks:
            return
        tick = self.ticks[-1]
        ticks = np.asarray(self.ticks)
        codes = np.asarray(self.codes)
        n = len(ticks)
        since = int(np.searchsorted(ticks, tick - VARIETY_BEATS * TICKS, "right"))
        recent = {s[:3] for s in self.sym[since:]}
        if n - since >= VARIETY_NOTES and len(recent) < VARIETY_FLOOR:
            amount = VARIETY_PENALTY * (VARIETY_FLOOR - len(recent))
            self.overused = [(*s, amount) for s in recent]
        first = int(np.searchsorted(ticks, tick - self.limit))
        if n - first < 4:
            return
        for p in range(1, LOOP_PERIODS + 1):
            if n - first < 2 * p or first < p:
                continue
            same = np.mean(codes[first:n] == codes[first - p:n - p])
            if same < LOOP_MATCH:
                continue
            start = first
            while start - p > 0 and self.sym[start - 1] == self.sym[start - 1 - p]:
                start -= 1
            over = (tick - ticks[start - p] - self.limit) / TICKS
            h, c, dd, g = self.sym[n - p]
            self.expect.append((tick + g, h, c, dd, LOOP_PENALTY + max(over, 0.0)))

    def _repeats(self):
        s = np.asarray(self.codes[-DRY_WINDOW:])
        n = len(s)
        if n <= DRY_ALLOWED:
            return []
        alive, length = np.arange(n - 1), np.zeros(n - 1, int)
        for k in range(DRY_ALLOWED + DRY_GROW):
            alive = alive[(alive >= k) & (s[alive - k] == s[n - 1 - k])] if len(alive) else alive
            if not len(alive):
                break
            length[alive] += 1
        best = {}
        last = self.ticks[-1]
        for j in np.nonzero(length >= DRY_ALLOWED)[0]:
            h, c, d, g = self.sym[len(self.sym) - n + j + 1]
            key = (last + g, h, c, d)
            best[key] = max(best.get(key, 0.0), self.dry * DRY_BASE ** (length[j] - DRY_ALLOWED))
        return [(*key, amount) for key, amount in best.items()]

    def penalty(self, rules):
        out = [(VOCAB.note(h, c, d), amount) for h, c, d, amount in self.overused]
        if self.stale:
            self.repeats, self.stale = self._repeats(), False
        out += [(VOCAB.note(h, c, d), amount) for t, h, c, d, amount in self.repeats if rules.tick == t]
        for t, h, c, d, amount in self.expect:
            if rules.tick == t:
                out.append((VOCAB.note(h, c, d), amount))
            elif rules.tick is not None and rules.tick < t < rules.end and not rules.after_time:
                out.append((VOCAB.time0 + t - rules.t0, amount))
        return out


class Rules:

    def __init__(self, prefix_events, end_tick, swings=True, objects=OBJ_WALLS | OBJ_BOMBS | OBJ_ARCS,
                 max_loop=MAX_LOOP, behind=TICKS // 2, dry=0.0):
        self.end, self.swings, self.objects = end_tick, swings, objects
        self.behind = behind
        self.loops = LoopGuard(max_loop, dry)
        self.tick = None
        self.face = {0: None, 1: None}
        self.need_note = None
        self.need_dur = False
        self.swing_tick = {0: None, 1: None}
        self.air = {0: True, 1: True}
        self.cells = set()
        self.after_time = False
        self.opened = {0: False, 1: False}
        self.first_dir = {0: None, 1: None}
        self.stack = {0: 0, 1: 0}
        self.prev_moment = {0: None, 1: None}
        self.notes_now = {0: [], 1: []}
        self.prev_note = {0: NO_NOTE, 1: NO_NOTE}
        self.arc_done = {0: False, 1: False}
        self.objects_now = False
        self.walls = []
        self.pending_wall = None
        self.need_time = False
        for e in prefix_events:
            if self.tick != int(e["tick"]):
                self._new_moment(int(e["tick"]))
            h, k = int(e["hand"]), int(e["kind"])
            if k == NOTE:
                self.loops.add(self.tick, h, int(e["cell"]), int(e["dir"]), check=False)
                self.notes_now[h].append(int(e["cell"]) * 9 + int(e["dir"]))
                self.opened[h] = True
                self.stack[h] += 1
                if self.first_dir[h] is None:
                    self.first_dir[h] = int(e["dir"])
                self.cells.add((h, int(e["cell"])))
            elif k == SWING:
                self.face[h], self.air[h], self.swing_tick[h] = bool(e["forehand"]), bool(e["air"]), self.tick
                self.first_dir[h] = None
            elif k == WALL:
                self._add_wall(int(e["shape"]), int(e["dur"]))
        self.loops.check()
        self.after_time = False

    def begin(self, t0, tick):
        self.t0 = t0
        self._new_moment(tick)
        self.need_time = True

    def _new_moment(self, tick):
        if self.tick is not None:
            for h in (0, 1):
                if self.stack[h]:
                    self.prev_moment[h] = self.tick
                if self.notes_now[h]:
                    self.prev_note[h] = max(self.notes_now[h])
        self.notes_now = {0: [], 1: []}
        self.tick = tick
        self.cells, self.objects_now = set(), False
        self.stack, self.first_dir, self.arc_done = {0: 0, 1: 0}, {0: None, 1: None}, {0: False, 1: False}
        self.walls = [w for w in self.walls if w[1] + self.behind > tick]

    def _add_wall(self, shape, dur):
        self.walls.append((self.tick, self.tick + dur, *WALL_CELLS[shape]))

    def _covered(self, behind=0, notes=False):
        out = set()
        for start, end, cells, lanes in self.walls:
            if start <= self.tick < end:
                out |= lanes if notes else cells
            elif end <= self.tick < end + behind:
                out |= lanes
        return out

    def _free(self, behind=0, notes=False):
        return set(range(12)) - {c for _, c in self.cells} - self._covered(behind, notes)

    def fit(self, h, dirs):
        if self.stack[h] >= MAX_STACK:
            return set()
        d0 = self.first_dir[h]
        if d0 is None or d0 == 8:
            return set(dirs)
        return {d for d in dirs if d == 8 or abs((DIR_ANGLE[d] - DIR_ANGLE[d0] + 180) % 360 - 180) <= 45}

    def mask(self):
        ok = np.zeros(VOCAB.size, bool)
        if self.need_dur:
            ok[VOCAB.dur0:VOCAB.events_end] = True
            return ok
        if self.need_note is not None:
            h = self.need_note
            dirs = FOREHAND_DIRS if self.face[h] else BACKHAND_DIRS
            if not self.opened[h]:
                dirs = dirs & OPENING_DIRS
            dirs = self.fit(h, dirs)
            for cell in self._free(self.behind, notes=True):
                for d in dirs:
                    ok[VOCAB.note(h, cell, d)] = True
            return ok
        lo = 0 if self.tick is None else self.tick + 1 - self.t0
        hi = self.end - self.t0
        if not self.after_time:
            ok[VOCAB.time0 + max(lo, 0): VOCAB.time0 + hi] = True
            ok[VOCAB.EOS] = True
        if self.tick is None or self.need_time:
            return ok
        free, note_free = self._free(), self._free(self.behind, notes=True)
        if self.objects & OBJ_BOMBS:
            for cell in free:
                ok[VOCAB.bomb(cell)] = True
        if self.objects & OBJ_WALLS:
            bombs = {c for h, c in self.cells if h == 2}
            notes = {c for h, c in self.cells if h != 2}
            for sh in range(tokens.N_SHAPES):
                cells, lanes = WALL_CELLS[sh]
                ok[VOCAB.wall(sh)] = not (cells & bombs or lanes & notes)
        if self.objects_now:
            return ok
        if self.objects & OBJ_ARCS:
            for h in (0, 1):
                ok[VOCAB.arc(h)] = self.stack[h] > 0 and self.prev_moment[h] is not None and not self.arc_done[h]
        if not self.swings:
            for h in (0, 1):
                if not self.arc_done[h]:
                    for cell in note_free:
                        for d in self.fit(h, range(9) if self.opened[h] else OPENING_DIRS):
                            ok[VOCAB.note(h, cell, d)] = True
            return ok
        for h in (0, 1):
            if self.arc_done[h]:
                continue
            if self.swing_tick[h] != self.tick and note_free:
                if self.face[h] is None:
                    ok[VOCAB.swing(h, 1, 0)] = True
                else:
                    f = int(not self.face[h])
                    ok[VOCAB.swing(h, f, 0)] = ok[VOCAB.swing(h, f, 1)] = True
            elif self.swing_tick[h] != self.tick and self.face[h] is not None:
                ok[VOCAB.swing(h, int(not self.face[h]), 1)] = True
            if not self.air[h] and self.face[h] is not None:
                dirs = self.fit(h, FOREHAND_DIRS if self.face[h] else BACKHAND_DIRS)
                for cell in note_free:
                    for d in dirs:
                        ok[VOCAB.note(h, cell, d)] = True
        return ok

    def push(self, k):
        if VOCAB.time0 <= k < VOCAB.swing0:
            self._new_moment(self.t0 + k - VOCAB.time0)
            self.after_time, self.need_time = True, False
            return
        self.after_time = False
        if VOCAB.swing0 <= k < VOCAB.note0:
            j = k - VOCAB.swing0
            h = j // 4
            self.face[h], self.air[h], self.swing_tick[h] = bool(j // 2 % 2), bool(j % 2), self.tick
            self.need_note = None if self.air[h] else h
            self.first_dir[h] = None
        elif VOCAB.note0 <= k < VOCAB.note_end:
            j = k - VOCAB.note0
            h = j // 108
            self.loops.add(self.tick, h, j // 9 % 12, j % 9)
            self.notes_now[h].append(j % 108)
            self.cells.add((h, j // 9 % 12))
            self.opened[h] = True
            self.stack[h] += 1
            if self.first_dir[h] is None:
                self.first_dir[h] = j % 9
            self.need_note = None
        elif VOCAB.arc0 <= k < VOCAB.bomb0:
            self.arc_done[k - VOCAB.arc0] = True
        elif VOCAB.bomb0 <= k < VOCAB.wall0:
            self.cells.add((2, k - VOCAB.bomb0))
            self.objects_now = True
        elif VOCAB.wall0 <= k < VOCAB.dur0:
            self.pending_wall, self.need_dur, self.objects_now = k - VOCAB.wall0, True, True
        elif VOCAB.dur0 <= k < VOCAB.events_end:
            self._add_wall(self.pending_wall, int(tokens.DURATIONS[k - VOCAB.dur0]))
            self.need_dur = False


GRAPHS = True


class Gate:

    def __init__(self):
        self.cond = threading.Condition()
        self.readers = self.waiting = 0
        self.recording = False

    @contextlib.contextmanager
    def reading(self):
        with self.cond:
            self.cond.wait_for(lambda: not self.recording and not self.waiting)
            self.readers += 1
        try:
            yield
        finally:
            with self.cond:
                self.readers -= 1
                self.cond.notify_all()

    @contextlib.contextmanager
    def writing(self):
        with self.cond:
            self.waiting += 1
            self.cond.wait_for(lambda: not self.recording and not self.readers)
            self.waiting -= 1
            self.recording = True
        try:
            yield
        finally:
            with self.cond:
                self.recording = False
                self.cond.notify_all()


GATE = Gate()
_pool_lock = threading.Lock()


def take_graph(model, batch, mem_len):
    with _pool_lock:
        pool = model.__dict__.setdefault("_graphs", {})
        for key in [k for k in pool if k[1] != mem_len]:
            del pool[key]
        free = pool.setdefault((batch, mem_len), [])
        if free:
            return free.pop()
    graph = GraphStep(model, batch, mem_len)
    with GATE.writing():
        graph.capture()
    return graph


def give_graph(model, graph):
    with _pool_lock:
        model.__dict__.setdefault("_graphs", {}).setdefault((graph.tok.shape[0], graph.mk[0].shape[2]), []).append(graph)


@torch.no_grad()
def sample_window(model, audio, prefixes, weights, rules, cfg, temp, top_p, dev, xtc=0.0, min_p=0.0, arc_boost=0.0,
                  hooks=(), gen=None, rng=None):
    n, k = len(rules), len(prefixes) - 1
    guided = cfg != 1.0 or k > 1
    use_graph = str(dev).startswith("cuda") and GRAPHS
    with GATE.reading():
        memory = model.encode(*(torch.as_tensor(a)[None].to(dev) for a in audio))
        mem_len = memory.shape[1]
    rows = [list(p) for p in prefixes[1:] for _ in range(n)] + ([list(prefixes[0]) for _ in range(n)] if guided else [])
    start = len(rows[0])
    vocab, max_len = model.tok.weight.shape[0], model.dec_pos.shape[0]
    done = np.zeros(n, bool)
    blocked = [[] for _ in range(n)]
    eos = np.zeros(vocab, bool)
    eos[VOCAB.EOS] = True
    rng = rng or np.random
    graph = take_graph(model, len(rows), mem_len) if use_graph else None
    try:
        with GATE.reading():
            w = torch.tensor(weights, device=dev, dtype=torch.float32)[:, None, None]
            cache, step = {}, torch.tensor(rows, device=dev)
            first = True
            while len(rows[0]) < max_len and not done.all():
                if graph is None:
                    lg = model.decode_step(memory, step, cache).float()
                else:
                    lg = (graph.prefill(memory, step) if first else graph.step(step)).float().clone()
                first = False
                if guided:
                    u = lg[k * n:]
                    lg = u + cfg * (w * (lg[:k * n].view(k, n, -1) - u)).sum(0)
                allowed = torch.from_numpy(np.stack([eos if done[i] else r.mask()[:vocab] for i, r in enumerate(rules)])).to(dev)
                bp = (torch.softmax(lg / temp, -1) * ~allowed).sum(1).tolist()
                if arc_boost:
                    lg[:, VOCAB.arc0:VOCAB.arc0 + 2] += arc_boost
                for i, r in enumerate(rules):
                    if not done[i]:
                        for tok, amount in r.loops.penalty(r):
                            lg[i, tok] -= amount
                for hook in hooks:
                    hook(lg, rules)
                lg[~allowed] = -float("inf")
                pr = torch.softmax(lg / temp, -1)
                if xtc > 0:
                    for i, r in enumerate(rules):
                        if not done[i] and r.need_note is not None and rng.random() < xtc:
                            top = torch.nonzero(pr[i] >= XTC_THRESHOLD)[:, 0]
                            if len(top) >= 2:
                                pr[i, top[pr[i, top] > pr[i, top].min()]] = 0
                sp, si = torch.sort(pr, -1, descending=True)
                if min_p > 0:
                    sp[sp < min_p * sp[:, :1]] = 0
                else:
                    sp[torch.cumsum(sp, -1) - sp >= top_p] = 0
                ks = si.gather(1, torch.multinomial(sp, 1, generator=gen))[:, 0].tolist()
                for i, t in enumerate(ks):
                    if not done[i]:
                        blocked[i].append(bp[i])
                        if t == VOCAB.EOS:
                            done[i] = True
                        else:
                            rules[i].push(t)
                    for j in range(i, len(rows), n):
                        rows[j].append(t)
                step = torch.tensor([r[-1:] for r in rows], device=dev)
            new = []
            for i in range(n):
                t = rows[i][start:]
                new.append(t[:t.index(VOCAB.EOS)] if VOCAB.EOS in t else t)
            return new, blocked
    finally:
        if graph is not None:
            give_graph(model, graph)


def window_activity(logmel, fps, beat_times, offset, spans):
    flux = np.maximum(np.diff(logmel.astype(np.float32), axis=0), 0).sum(1)
    sec = beat_to_sec(np.asarray(spans, np.float64).ravel() + offset, beat_times).reshape(-1, 2)
    f = np.clip((sec * fps).astype(int), 0, len(flux))
    act = np.array([flux[a:b].mean() if b > a else 0.0 for a, b in f])
    return act / max(np.median(act), 1e-6), sec[:, 1] - sec[:, 0]


def song_mel(y, sr):
    import librosa
    y = librosa.resample(y, orig_sr=sr, target_sr=22050)
    mel = librosa.feature.melspectrogram(y=y, sr=22050, n_fft=1024, hop_length=256, n_mels=80)
    return librosa.power_to_db(mel, ref=np.max).T.astype(np.float16), 22050 / 256


def read_checkpoint(path):
    path = Path(path)
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        return dict(json.loads(path.with_suffix(".json").read_text()), model=load_file(path))
    return torch.load(path, map_location="cpu")


def load_model(path, dev="cuda"):
    torch.backends.cuda.matmul.allow_tf32 = True
    ck = read_checkpoint(path)
    a = ck["args"]
    vocab, max_tokens = ck["model"]["tok.weight"].shape[0], ck["model"]["dec_pos"].shape[0]
    model = Generator(vocab, d=a["d"], enc_layers=a["layers"], dec_layers=a["layers"], heads=a["d"] // 64,
                      conv_blocks=a.get("conv_blocks", 0), global_cond=a.get("global_cond", False),
                      max_tokens=max_tokens)
    model.load_state_dict(ck["model"])
    model.controls = vocab >= VOCAB.size
    model.mappers = ck.get("mappers", [])
    model.swings = not a.get("no_swings", False)
    return model.to(dev).eval()


def generate(model, logmel, fps, beat_times, offset, n_beats, stars, cfg=1.5, temp=0.9, top_p=0.95,
             dev="cuda", log=None, candidates=1, controls=None, styles=None, max_loop=MAX_LOOP, dry=0.0, xtc=0.0,
             min_p=0.0, arc_boost=0.0, arc_rate=0.0, hooks=(), gen=None, rng=None, progress=None):
    if getattr(model, "controls", False):
        c = dict(controls or {})
        if c.get("mapper") is not None:
            name = c["mapper"].strip().lower()
            if name not in model.mappers:
                raise ValueError(f"unknown mapper {c['mapper']!r}; the model knows: {', '.join(model.mappers)}")
            c["mapper"] = model.mappers.index(name)
        c.setdefault("objects", OBJ_WALLS | OBJ_BOMBS | OBJ_ARCS)
        cond = {f: c.get(f) for f in tokens.COND_FIELDS}
        cond["stars"] = stars
        blend = {s: w / sum(styles.values()) for s, w in styles.items() if w > 0} if styles else {None: 1.0}
        conds = [dict(cond, style=s) for s in blend]
        weights, objects = list(blend.values()), cond["objects"]
        uncond = VOCAB.none_like(cond)
    else:
        conds, weights, objects, uncond = [stars], [1.0], 0, None
    ev, blocked = np.zeros(0, tokens.EVENT_DTYPE), []
    target = c.get("nps") if getattr(model, "controls", False) else None
    first_sec = None
    bomb_cap = int(np.interp(stars if stars is not None else BOMB_CAP_STARS[-1], BOMB_CAP_STARS, BOMB_CAP))
    ov, bar = song_overview(logmel, fps, beat_times, offset)
    behind = behind_wall_ticks((c.get("njs") if getattr(model, "controls", False) else None) or 18.0, beat_times)
    swings = getattr(model, "swings", True)
    last_note = 0
    for b in range(1 - CONTEXT, n_beats - 2, WINDOW):
        t0, new0 = b * TICKS, (b + CONTEXT) * TICKS
        end = min(t0 + (CONTEXT + WINDOW) * TICKS, (n_beats - 2) * TICKS)
        audio = (beat_mel(logmel, fps, beat_times, b + offset), ov, overview_rel(bar, b))

        def draw(hide_walls, allow=objects):
            prev = ev[(ev["tick"] >= t0) & (ev["tick"] < new0)]
            if hide_walls:
                prev = prev[prev["kind"] != WALL]
            prefixes = [tokens.encode(prev, t0, c, beats=CONTEXT)[:-1] for c in [uncond] + conds]
            rules = []
            for _ in range(candidates):
                r = Rules(ev, end, swings, allow, max_loop, behind, dry)
                r.begin(t0, new0 - 1)
                rules.append(r)
            boost = arc_boost
            if arc_rate > 0:
                under = np.sum(ev["kind"] == ARC) < arc_rate * max(np.sum(ev["kind"] == NOTE), 1)
                boost = ARC_PUSH if under else -ARC_PUSH
            new, bl = sample_window(model, audio, prefixes, weights, rules, cfg, temp, top_p, dev, xtc, min_p, boost,
                                    hooks, gen, rng)
            got = [tokens.decode(prefixes[1] + t, t0) for t in new]
            return [g[g["tick"] >= new0] for g in got], bl

        def gap(g):
            ticks = g["tick"][g["kind"] == NOTE]
            return (ticks.min() if len(ticks) else end) - last_note

        limit = (MAX_GAP if first_sec is not None else MAX_INTRO) * TICKS

        got, bl = draw(False)
        note = ""
        if objects & OBJ_WALLS and all(gap(g) > limit for g in got):
            got, bl = draw(True)
            note = " [gap: redrawn without the lookback walls]"
        counts = np.array([np.sum(g["kind"] == NOTE) for g in got])
        walls = np.array([np.sum(g["kind"] == WALL) for g in got])
        bombs = np.array([np.sum(g["kind"] == BOMB) for g in got])
        have_b, have_n = np.sum(ev["kind"] == BOMB), np.sum(ev["kind"] == NOTE)
        bomb_ok = (bombs <= bomb_cap) & (have_b + bombs <= BOMB_RATIO * (have_n + counts) + bomb_cap)
        if not bomb_ok.any():
            got, bl = draw(False, objects & ~OBJ_BOMBS)
            note += " [bombs: redrawn without]"
            counts = np.array([np.sum(g["kind"] == NOTE) for g in got])
            walls = np.array([np.sum(g["kind"] == WALL) for g in got])
            bombs = np.zeros(len(got), int); bomb_ok = np.ones(len(got), bool)
        ok = [i for i, g in enumerate(got) if gap(g) <= limit] or list(range(len(got)))
        gap_k = ok[pick_candidate(counts[ok])]
        if len(ok) < len(got) and gap_k != pick_candidate(counts):
            note += " [gap: passed over an empty candidate]"
        few = (counts < FEW_NOTES) & (np.sum(counts >= FEW_NOTES) * 2 > len(got))
        bad = few.astype(int) + (walls > WALL_CAP) + ~bomb_ok
        ok = [i for i in ok if bad[i] == bad[ok].min()]
        k = ok[pick_candidate(counts[ok])]
        if target and first_sec is not None:
            med = np.median(counts[ok])
            near = [i for i in ok if med / 1.25 <= counts[i] <= med * 1.25 or abs(counts[i] - med) <= 2] or ok
            span = beat_to_sec(np.array([end / TICKS + offset]), beat_times)[0] - first_sec
            have = np.sum(ev["kind"] == NOTE)
            k = near[int(np.argmin([abs(have + counts[i] - target * span) for i in near]))]
        if k != gap_k and "[gap" not in note:
            note += " [guard]"
        if walls[k] > WALL_CAP:
            g = got[k]
            got[k] = g[(g["kind"] != WALL) | (np.cumsum(g["kind"] == WALL) <= WALL_CAP)]
            note += f" [walls cut to {WALL_CAP}]"
        if bombs[k] > bomb_cap:
            g = got[k]
            got[k] = g[(g["kind"] != BOMB) | (np.cumsum(g["kind"] == BOMB) <= bomb_cap)]
            note += f" [bombs cut to {bomb_cap}]"
        ev = np.concatenate([ev, got[k]])
        blocked += bl[k]
        if counts[k]:
            last_note = int(got[k]["tick"][got[k]["kind"] == NOTE].max())
            if first_sec is None:
                first_sec = beat_to_sec(np.array([got[k]["tick"][got[k]["kind"] == NOTE].min() / TICKS + offset]), beat_times)[0]
        if log:
            log(f"beat {b + CONTEXT:4d}-{b + CONTEXT + WINDOW:4d}: {counts[k]} notes"
                + (f" (candidates {' '.join(map(str, counts))})" if candidates > 1 else "")
                + f", walls {walls[k]} ({' '.join(map(str, walls))})" + note)
        if progress:
            progress(min(b + CONTEXT + WINDOW, n_beats))
    return ev, np.array(blocked)


def pick_candidate(counts, spread=1.25):
    med = np.median(counts)
    for i, c in enumerate(counts):
        if med / spread <= c <= med * spread or abs(c - med) <= 2:
            return i
    return int(np.argmin(np.abs(counts - med)))
