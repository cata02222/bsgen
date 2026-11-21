import numpy as np

TICKS = 48
WINDOW = 32
CONTEXT = 16
STAR_STEP, N_STARS = 0.5, 33

EVENT_DTYPE = np.dtype([
    ("tick", "i8"), ("kind", "i1"), ("hand", "i1"), ("forehand", "?"), ("air", "?"),
    ("cell", "i1"), ("dir", "i1"), ("note", "i4"),
    ("shape", "i2"), ("dur", "i4"),
])
SWING, NOTE, ARC, BOMB, WALL, CHAIN = 0, 1, 2, 3, 4, 5
NO_HAND = 2

WALL_X, WALL_Y, WALL_W, WALL_H = range(-2, 6), range(3), range(1, 5), range(1, 6)
N_SHAPES = len(WALL_X) * len(WALL_Y) * len(WALL_W) * len(WALL_H)
DURATIONS = np.array([1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 72, 96, 120, 144, 192, 240, 288, 384,
                      480, 576, 768, 960, 1152, 1536, 1920, 2304, 3072, 3840, 4608, 6144])

STYLES = ("acc", "tech", "midspeed", "speed")
MAX_MAPPERS = 128
NJS_RANGE = (6, 30)
RATING_STEP, N_RATINGS = 0.5, 33
COND_FIELDS = ("stars", "style", "mapper", "njs", "nps", "objects", "pass", "acc", "tech")
WNPS_STEP, N_WNPS = 0.5, 49
OBJ_WALLS, OBJ_BOMBS, OBJ_ARCS = 1, 2, 4
OBJ_CHAINS = 8
CHAIN_BEATS, CHAIN_SLICES, CHAIN_SQUISH = 0.0625, 5, 1.0
DIR_STEP = {0: (0, 1), 1: (0, -1), 2: (-1, 0), 3: (1, 0), 4: (-1, 1), 5: (1, 1), 6: (-1, -1), 7: (1, -1)}
AMOUNT_FIELDS = ("walls", "bombs", "arcs", "chains")
AMOUNTS = ("none", "rare", "normal", "lots")


def wall_shape(x, y, w, h):
    if x not in WALL_X or y not in WALL_Y or w not in WALL_W or h not in WALL_H:
        return -1
    return ((x - WALL_X[0]) * len(WALL_Y) + y) * len(WALL_W) * len(WALL_H) + (w - 1) * len(WALL_H) + h - 1


def shape_of(i):
    i, h = divmod(int(i), len(WALL_H))
    i, w = divmod(i, len(WALL_W))
    x, y = divmod(i, len(WALL_Y))
    return x + WALL_X[0], y, w + 1, h + 1


def dur_bucket(ticks):
    return int(np.argmin(np.abs(np.log(DURATIONS) - np.log(max(ticks, 1)))))


class Vocab:
    PAD, BOS, EOS, NOSTAR = 0, 1, 2, 3

    def __init__(self):
        self.star0 = 4
        self.time0 = self.star0 + N_STARS
        self.swing0 = self.time0 + (CONTEXT + WINDOW) * TICKS
        self.note0 = self.swing0 + 8
        self.note_end = self.note0 + 2 * 12 * 9
        self.arc0 = self.note_end
        self.bomb0 = self.arc0 + 2
        self.wall0 = self.bomb0 + 12
        self.dur0 = self.wall0 + N_SHAPES
        self.events_end = self.dur0 + len(DURATIONS)
        sizes = dict(style=len(STYLES), mapper=MAX_MAPPERS + 1, njs=NJS_RANGE[1] - NJS_RANGE[0] + 1,
                     nps=N_RATINGS, objects=8, **{"pass": N_RATINGS}, acc=N_RATINGS, tech=N_RATINGS)
        self.cond0, k = {}, self.events_end
        for f in COND_FIELDS[1:]:
            self.cond0[f] = k
            k += sizes[f] + 1
        self.cond_size = sizes
        self.controls_end = k
        self.wnps0 = k
        self.plan_end = k + N_WNPS + 1
        self.chain0 = self.plan_end
        self.amount0, k = {}, self.chain0 + 2
        for f in AMOUNT_FIELDS:
            self.amount0[f] = k
            k += len(AMOUNTS) + 1
        self.size = k

    def star(self, stars):
        if stars is None:
            return self.NOSTAR
        return self.star0 + int(np.clip(round(stars / STAR_STEP), 0, N_STARS - 1))

    def cond(self, field, value):
        if field == "stars":
            return self.star(value)
        n = self.cond_size[field]
        if value is None:
            return self.cond0[field] + n
        if field == "style":
            j = STYLES.index(value)
        elif field == "njs":
            j = int(np.clip(round(value), *NJS_RANGE)) - NJS_RANGE[0]
        elif field in ("nps", "pass", "acc", "tech"):
            j = int(np.clip(round(value / RATING_STEP), 0, N_RATINGS - 1))
        else:
            j = int(value)
        return self.cond0[field] + j

    def wnps(self, value):
        if value is None:
            return self.wnps0 + N_WNPS
        return self.wnps0 + int(np.clip(round(value / WNPS_STEP), 0, N_WNPS - 1))

    def amount(self, field, value):
        return self.amount0[field] + (len(AMOUNTS) if value is None else int(value))

    def prefix(self, cond):
        if not isinstance(cond, dict):
            return [self.BOS, self.star(cond)]
        out = [self.BOS] + [self.cond(f, cond.get(f)) for f in COND_FIELDS]
        if "wnps" in cond:
            out.append(self.wnps(cond["wnps"]))
        if "amounts" in cond:
            out += [self.amount(f, (cond["amounts"] or {}).get(f)) for f in AMOUNT_FIELDS]
        return out

    def none_like(self, cond, keep=("objects",)):
        if not isinstance(cond, dict):
            return None
        out = {f: cond.get(f) if f in keep else None for f in COND_FIELDS}
        if "wnps" in cond:
            out["wnps"] = None
        if "amounts" in cond:
            out["amounts"] = cond["amounts"]
        return out

    def swing(self, hand, forehand, air):
        return self.swing0 + hand * 4 + forehand * 2 + air

    def note(self, hand, cell, d):
        return self.note0 + (hand * 12 + cell) * 9 + d

    def arc(self, hand):
        return self.arc0 + hand

    def chain(self, hand):
        return self.chain0 + hand

    def bomb(self, cell):
        return self.bomb0 + cell

    def wall(self, shape):
        return self.wall0 + shape

    def dur(self, ticks):
        return self.dur0 + dur_bucket(ticks)


VOCAB = Vocab()


def air_ticks(hands, first, raw):
    out = np.full(len(hands), -1, np.int64)
    order = np.asarray(raw, np.int64).copy()
    for s, t in first.items():
        out[s] = order[s] = t
    for h in np.unique(hands):
        idx = np.flatnonzero(hands == h)
        idx = idx[np.argsort(order[idx], kind="stable")]
        run, prev = [], None
        for s in idx:
            if s not in first:
                run.append(s)
                continue
            nxt = int(order[s])
            if prev is not None and len(run) % 2:
                out[run[0]] = reset_tick(prev, nxt)
            run, prev = [], nxt
    return out


def reset_tick(prev, nxt):
    mid = (prev + nxt) / 2
    for step in (TICKS // 4, TICKS // 8, TICKS // 12, TICKS // 16, 1):
        lo, hi = prev // step + 1, -(-nxt // step) - 1
        if lo <= hi:
            return int(np.clip(round(mid / step), lo, hi) * step)
    return prev + 1 if nxt - prev > 1 else nxt


def events(notes, owner, swings, swing_beat, grid_offset, bombs=None, walls=None, arcs=None):
    tick = lambda b: np.round((np.asarray(b) - grid_offset) * TICKS).astype(np.int64)
    nt = tick(notes["beat"])
    first = {}
    for i in np.argsort(nt, kind="stable"):
        if owner[i] >= 0:
            first.setdefault(int(owner[i]), int(nt[i]))
    ticks = air_ticks(swings["hand"], first, tick(swing_beat))
    ev = []
    for s in np.flatnonzero(ticks >= 0):
        ev.append((ticks[s], SWING, swings["hand"][s], swings["forehand"][s], s not in first, -1, -1, -1, -1, 0))
    for i in range(len(notes)):
        ev.append((nt[i], NOTE, notes["color"][i], False, False,
                   notes["layer"][i] * 4 + notes["line"][i], notes["cut_dir"][i], i, -1, 0))
    ev += _objects(nt, notes["color"], tick, bombs, walls, arcs)
    ev = np.array(ev, EVENT_DTYPE)
    ev = ev[np.lexsort((ev["shape"], ev["cell"], ev["kind"], ev["hand"], ev["tick"]))]
    if (owner >= 0).all():
        return ev

    out, face, opened = [], {0: True, 1: True}, {}
    for e in ev:
        h = int(e["hand"])
        if e["kind"] == SWING:
            face[h] = bool(e["forehand"])
        elif e["kind"] == NOTE and owner[e["note"]] < 0 and opened.get(h) != e["tick"]:
            face[h] = not face[h]
            out.append((e["tick"], SWING, h, face[h], False, -1, -1, -1, -1, 0))
            opened[h] = e["tick"]
        out.append(tuple(e))
    return np.array(out, EVENT_DTYPE)


def _objects(nt, color, tick, bombs, walls, arcs):
    ev = []
    if bombs is not None and len(bombs):
        for t, x, y in zip(tick(bombs["beat"]), bombs["line"], bombs["layer"]):
            if 0 <= x < 4 and 0 <= y < 3:
                ev.append((t, BOMB, NO_HAND, False, False, y * 4 + x, -1, -1, -1, 0))
    if walls is not None and len(walls):
        for t, d, x, y, w, h in zip(tick(walls["beat"]), walls["duration"], walls["line"], walls["layer"],
                                    walls["width"], walls["height"]):
            sh = wall_shape(int(x), int(y), int(w), int(h))
            if sh >= 0 and d > 0:
                ev.append((t, WALL, NO_HAND, False, False, -1, -1, -1, sh, max(1, round(d * TICKS))))
    if arcs is not None and len(arcs):
        moments = {h: np.unique(nt[color == h]) for h in (0, 1)}
        for a, ht, tt in zip(arcs, tick(arcs["beat"]), tick(arcs["tail_beat"])):
            m = moments.get(int(a["color"]))
            if m is None or tt not in m or ht not in m:
                continue
            j = np.searchsorted(m, tt)
            if j > 0 and m[j - 1] == ht:
                ev.append((tt, ARC, int(a["color"]), False, False, -1, -1, -1, -1, 0))
    return ev


def encode(ev, t0, cond, beats=CONTEXT + WINDOW):
    w = ev[(ev["tick"] >= t0) & (ev["tick"] < t0 + beats * TICKS)]
    out, last = VOCAB.prefix(cond), None
    for e in w:
        if e["tick"] != last:
            out.append(VOCAB.time0 + int(e["tick"] - t0))
            last = e["tick"]
        k = e["kind"]
        if k == SWING:
            out.append(VOCAB.swing(int(e["hand"]), int(e["forehand"]), int(e["air"])))
        elif k == NOTE:
            out.append(VOCAB.note(int(e["hand"]), int(e["cell"]), int(e["dir"])))
        elif k == ARC:
            out.append(VOCAB.arc(int(e["hand"])))
        elif k == BOMB:
            out.append(VOCAB.bomb(int(e["cell"])))
        elif k == CHAIN:
            out.append(VOCAB.chain(int(e["hand"])))
        else:
            out += [VOCAB.wall(int(e["shape"])), VOCAB.dur(int(e["dur"]))]
    out.append(VOCAB.EOS)
    return out


def decode(tokens, t0):
    ev, tick, wall = [], None, None
    for k in tokens:
        if VOCAB.dur0 <= k < VOCAB.events_end:
            if wall is not None:
                ev.append((tick, WALL, NO_HAND, False, False, -1, -1, -1, wall, int(DURATIONS[k - VOCAB.dur0])))
            wall = None
            continue
        wall = None
        if VOCAB.time0 <= k < VOCAB.swing0:
            tick = t0 + k - VOCAB.time0
        elif tick is None:
            continue
        elif VOCAB.swing0 <= k < VOCAB.note0:
            j = k - VOCAB.swing0
            ev.append((tick, SWING, j // 4, bool(j // 2 % 2), bool(j % 2), -1, -1, -1, -1, 0))
        elif VOCAB.note0 <= k < VOCAB.note_end:
            j = k - VOCAB.note0
            ev.append((tick, NOTE, j // 108, False, False, j // 9 % 12, j % 9, -1, -1, 0))
        elif VOCAB.arc0 <= k < VOCAB.bomb0:
            ev.append((tick, ARC, k - VOCAB.arc0, False, False, -1, -1, -1, -1, 0))
        elif VOCAB.bomb0 <= k < VOCAB.wall0:
            ev.append((tick, BOMB, NO_HAND, False, False, k - VOCAB.bomb0, -1, -1, -1, 0))
        elif VOCAB.wall0 <= k < VOCAB.dur0:
            wall = k - VOCAB.wall0
        elif VOCAB.chain0 <= k < VOCAB.chain0 + 2:
            ev.append((tick, CHAIN, k - VOCAB.chain0, False, False, -1, -1, -1, -1, 0))
    return np.array(ev, EVENT_DTYPE)


def add_chains(ev, chains, grid_offset):
    if chains is None or not len(chains):
        return ev, 0
    notes = ev[ev["kind"] == NOTE]
    have = {(int(t), int(h), int(c)) for t, h, c in zip(notes["tick"], notes["hand"], notes["cell"])}
    ticks = np.round((np.asarray(chains["beat"]) - grid_offset) * TICKS).astype(np.int64)
    new = set()
    for t, h, x, y in zip(ticks, chains["color"], chains["line"], chains["layer"]):
        if (int(t), int(h), int(y) * 4 + int(x)) in have:
            new.add((int(t), int(h)))
    if not new:
        return ev, 0
    at = [int(np.flatnonzero((ev["tick"] == t) & (ev["hand"] == h))[-1]) + 1 for t, h in sorted(new)]
    add = np.array([(t, CHAIN, h, False, False, -1, -1, -1, -1, 0) for t, h in sorted(new)], EVENT_DTYPE)
    return np.insert(ev, at, add), len(new)


def to_notes(ev, grid_offset):
    n = ev[ev["kind"] == NOTE]
    out = np.zeros(len(n), [("beat", "f8"), ("line", "i1"), ("layer", "i1"), ("color", "i1"), ("cut_dir", "i1")])
    out["beat"] = n["tick"] / TICKS + grid_offset
    out["line"], out["layer"] = n["cell"] % 4, n["cell"] // 4
    out["color"], out["cut_dir"] = n["hand"], n["dir"]
    return out


CHAIN_FRACTIONS = np.array([0, 1 / 32, 1 / 24, 1 / 16, 1 / 12, 1 / 8, 1 / 6, 1 / 4])
CHAIN_BPM_BANDS = (150, 200, 260)
CHAIN_LENGTH_WEIGHTS = np.array([
    [0.157, 0.181, 0.169, 0.317, 0.111, 0.057, 0.002, 0.007],
    [0.177, 0.117, 0.181, 0.230, 0.042, 0.224, 0.010, 0.021],
    [0.014, 0.098, 0.097, 0.178, 0.339, 0.162, 0.025, 0.087],
    [0.000, 0.031, 0.000, 0.077, 0.038, 0.843, 0.011, 0.000]])
CHAIN_LINKS = {"none": {4: 0.07, 5: 0.55, 7: 0.07, 8: 0.07},
               "one": {2: 0.03, 3: 0.29, 4: 0.34, 5: 0.29, 6: 0.03},
               "far": {3: 0.14, 4: 0.22, 5: 0.40, 6: 0.12, 7: 0.07, 8: 0.015}}
CHAIN_SQUISHES = {1.0: 0.80, 0.8: 0.07, 0.9: 0.02, 0.7: 0.02, 0.6: 0.025, 0.5: 0.022, 1.1: 0.011, 1.2: 0.016}
CHAIN_TAILS = {False: {(1, 0): 0.686, (2, 0): 0.214, (2, 1): 0.035, (2, -1): 0.035},
               True: {(2, 0): 0.403, (3, -1): 0.278, (3, 1): 0.267, (4, 0): 0.014}}
CHAIN_KEEP = 0.64
CHAIN_CLEAR_S = 0.25


def _draw(rng, weights):
    keys = list(weights)
    p = np.array([weights[k] for k in keys], np.float64)
    return keys[rng.choice(len(keys), p=p / p.sum())]


def _chain_cells(x, y, tx, ty, links):
    out = set()
    for u in np.linspace(0, 1, max(links, 2))[1:]:
        px, py = x + (tx - x) * u, y + (ty - y) * u
        out |= {(int(np.floor(a + 0.5)), int(np.floor(b + 0.5))) for a in (px - 0.3, px + 0.3) for b in (py - 0.3, py + 0.3)}
    return out - {(x, y)}


def to_chains(ev, grid_offset, seconds=None, bombs=None, walls=None, seed=0):
    from .beatmap import CHAIN_DTYPE
    notes = ev[ev["kind"] == NOTE]
    rng = np.random.default_rng(seed)
    out, last = [], None
    nb = notes["tick"] / TICKS + grid_offset
    for c in np.sort(ev[ev["kind"] == CHAIN], order=["tick", "hand"]):
        head = notes[(notes["tick"] == c["tick"]) & (notes["hand"] == c["hand"])]
        if len(head) != 1 or int(head[0]["dir"]) not in DIR_STEP:
            continue
        x, y, d = int(head[0]["cell"]) % 4, int(head[0]["cell"]) // 4, int(head[0]["dir"])
        dx, dy = DIR_STEP[d]
        tx, ty = x + dx, y + dy
        if not (0 <= tx <= 3 and 0 <= ty <= 2):
            tx, ty = x, y
        beat = c["tick"] / TICKS + grid_offset
        if seconds is None:
            out.append((beat, int(c["hand"]), x, y, d, beat + CHAIN_BEATS, tx, ty, CHAIN_SLICES, CHAIN_SQUISH))
            continue
        spb = float(seconds(beat + 0.5) - seconds(beat - 0.5))
        clear = CHAIN_CLEAR_S / max(spb, 1e-3)

        def busy(cx, cy):
            near = lambda b, a=-1e-6: (b >= beat + a) & (b <= beat + clear)
            cell = cy * 4 + cx
            n = near(nb) & (notes["cell"] == cell) & ~((notes["tick"] == c["tick"]) & (notes["hand"] == c["hand"]))
            if n.any():
                return True
            if bombs is not None and len(bombs) and (near(bombs["beat"], -0.1 / max(spb, 1e-3)) & (bombs["line"] == cx) & (bombs["layer"] == cy)).any():
                return True
            if walls is not None and len(walls):
                w = walls[(walls["beat"] <= beat + clear) & (walls["beat"] + walls["duration"] >= beat)
                          & (walls["line"] <= cx) & (walls["line"] + walls["width"] > cx)
                          & (walls["layer"] <= cy) & (walls["layer"] + walls["height"] > cy)]
                if len(w):
                    return True
            return False

        if last is not None and (last[0] == c["tick"] or rng.random() < CHAIN_KEEP):
            length, links, squish = last[1:]
        else:
            band = int(np.searchsorted(CHAIN_BPM_BANDS, 60.0 / max(spb, 1e-3), side="right"))
            w = CHAIN_LENGTH_WEIGHTS[band]
            length = float(CHAIN_FRACTIONS[rng.choice(len(w), p=w / w.sum())])
            links, squish = None, _draw(rng, CHAIN_SQUISHES)
        diag = dx != 0 and dy != 0
        tails = dict(CHAIN_TAILS[diag])
        n2 = dx * dx + dy * dy
        while tails:
            a, cr = _draw(rng, tails)
            del tails[(a, cr)]
            ox, oy = (a * dx + cr * dy) // n2, (a * dy - cr * dx) // n2
            one = (a, cr) == ((2, 0) if diag else (1, 0))
            k = links or _draw(rng, CHAIN_LINKS["none" if length == 0 else "one" if one else "far"])
            ux, uy = x + ox, y + oy
            if 0 <= ux <= 3 and 0 <= uy <= 2 and not any(busy(*q) for q in _chain_cells(x, y, ux, uy, k)):
                tx, ty, links = ux, uy, k
                break
        if links is None:
            links = _draw(rng, CHAIN_LINKS["none" if length == 0 else "one"])
        out.append((beat, int(c["hand"]), x, y, d, beat + length, tx, ty, links, squish))
        last = (int(c["tick"]), length, links, squish)
    return np.array(out, CHAIN_DTYPE) if out else np.zeros(0, CHAIN_DTYPE)


def arc_end(stack, other, last):
    if len(stack) < 2:
        return stack[0]
    x, y = stack["cell"] % 4, stack["cell"] // 4
    arrows = [int(d) for d in stack["dir"] if int(d) in DIR_STEP]
    if arrows:
        dx, dy = DIR_STEP[arrows[0]]
        along = x * dx + y * dy
        return stack[int(np.argmax(along) if last else np.argmin(along))]
    ox, oy = (other["cell"] % 4).mean(), (other["cell"] // 4).mean()
    return stack[int(np.argmin((x - ox) ** 2 + (y - oy) ** 2))]


def to_objects(ev, grid_offset):
    from .beatmap import ARC_DTYPE, BOMB_DTYPE, WALL_DTYPE
    beat = lambda t: np.asarray(t) / TICKS + grid_offset
    b = ev[ev["kind"] == BOMB]
    bombs = np.zeros(len(b), BOMB_DTYPE)
    bombs["beat"], bombs["line"], bombs["layer"] = beat(b["tick"]), b["cell"] % 4, b["cell"] // 4
    w = ev[ev["kind"] == WALL]
    walls = np.zeros(len(w), WALL_DTYPE)
    walls["beat"], walls["duration"] = beat(w["tick"]), w["dur"] / TICKS
    for i, sh in enumerate(w["shape"]):
        walls["line"][i], walls["layer"][i], walls["width"][i], walls["height"][i] = shape_of(sh)
    notes = ev[ev["kind"] == NOTE]
    arcs = []
    for a in ev[ev["kind"] == ARC]:
        h = notes[notes["hand"] == a["hand"]]
        tail, before = h[h["tick"] == a["tick"]], h[h["tick"] < a["tick"]]
        if not len(tail) or not len(before):
            continue
        head = before[before["tick"] == before["tick"].max()]
        hd, td = arc_end(head, tail, last=True), arc_end(tail, head, last=False)
        arcs.append((beat(hd["tick"]), a["hand"], hd["cell"] % 4, hd["cell"] // 4, hd["dir"], 1.0,
                     beat(td["tick"]), td["cell"] % 4, td["cell"] // 4, td["dir"], 1.0, 0))
    return bombs, walls, np.array(arcs, ARC_DTYPE) if arcs else np.zeros(0, ARC_DTYPE)
