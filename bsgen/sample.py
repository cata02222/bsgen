import bisect
import contextlib
import copy
import json
import threading
from pathlib import Path

import numpy as np
import torch

from . import tokens
from .data import (AHEAD, MEM_BEATS, REPEAT_SIM, beat_mel, beat_prints, beat_to_sec, best_match, hit_roll, memory_input, memory_match,
                   overview_rel, song_overview)
from .model import Generator, GraphStep, SharedStep
from .stats import off_flow
from .tokens import ARC, BOMB, CONTEXT, NOTE, OBJ_ARCS, OBJ_BOMBS, OBJ_WALLS, SWING, TICKS, VOCAB, WALL, WINDOW

FOREHAND_DIRS = {1, 6, 7, 2, 3, 8}
BACKHAND_DIRS = {0, 4, 5, 2, 3, 8}
OPENING_DIRS = {1, 6, 7, 8}
DIR_ANGLE = {0: 180, 1: 0, 2: 270, 3: 90, 4: 225, 5: 135, 6: 315, 7: 45}
SLIDE_TURN = 45
SWING_GAP = TICKS // 4
SWING_GAP_STARS, SWING_GAP_MS = [8, 12], [85, 60]


def swing_gap_ticks(beat_times, stars):
    spb = float(np.median(np.diff(beat_times))) if len(beat_times) > 1 else 0.5
    ms = float(np.interp(stars if stars is not None else SWING_GAP_STARS[-1], SWING_GAP_STARS, SWING_GAP_MS))
    return max(SWING_GAP, int(np.ceil(ms / 1000 / spb * TICKS)))
DOWN_DIRS, UP_DIRS = {1, 6, 7}, {0, 4, 5}
UNCROSS_STARS = 5.0
HAND_COLS = {0: (0, 1), 1: (2, 3)}


def _move_angle(a, b):
    return float(np.degrees(np.arctan2(b % 4 - a % 4, a // 4 - b // 4))) % 360


def _turn(a, b):
    return abs((a - b + 180) % 360 - 180)


def _comes_through(c, o, d):
    return (d != 8 and max(abs(c % 4 - o % 4), abs(c // 4 - o // 4)) == 1
            and _turn(_move_angle(c, o), DIR_ANGLE[d] + 180) < 1)


CLASH = [[[_comes_through(c, o, d) for d in range(9)] for o in range(12)] for c in range(12)]
MAX_STACK = 4
MAX_GAP = 96
BOMB_GAP, BOMB_GAP_RATE = 48, 0.5
MAX_INTRO = 48
BEHIND_WALL_M = 11.0


def behind_wall_ticks(njs, beat_times):
    spb = float(np.median(np.diff(beat_times))) if len(beat_times) > 1 else 0.5
    return max(int(round(BEHIND_WALL_M / max(njs, 1.0) / spb * TICKS)), 1)


DOUBLE_GAP_S = 0.08


def double_gap_ticks(beat_times):
    spb = float(np.median(np.diff(beat_times))) if len(beat_times) > 1 else 0.5
    return int(np.ceil(DOUBLE_GAP_S / spb * TICKS))


CELL_GAP_BEATS, CELL_GAP_S = 0.25, 0.06


def cell_gap_ticks(beat_times):
    spb = float(np.median(np.diff(beat_times))) if len(beat_times) > 1 else 0.5
    return max(int(round(CELL_GAP_BEATS * TICKS)), int(np.ceil(CELL_GAP_S / spb * TICKS)))


NEAR_GAP_S = 0.04


def near_gap_ticks(beat_times):
    spb = float(np.median(np.diff(beat_times))) if len(beat_times) > 1 else 0.5
    return int(np.ceil(NEAR_GAP_S / spb * TICKS))


WALL_SWITCH_S = 0.60


def wall_switch_ticks(beat_times):
    spb = float(np.median(np.diff(beat_times))) if len(beat_times) > 1 else 0.5
    return int(np.ceil(WALL_SWITCH_S / spb * TICKS))


def off_flow_rate(ev):
    e = ev[ev["kind"] == NOTE]
    n = np.zeros(len(e), [("beat", "f8"), ("color", "i1"), ("cut_dir", "i1")])
    n["beat"], n["color"], n["cut_dir"] = e["tick"] / TICKS, e["hand"], e["dir"]
    return off_flow(n)


def hits(ev):
    return len(np.unique(ev["tick"][ev["kind"] == NOTE]))


def cell_bombs(ev):
    return np.bincount(ev["cell"][ev["kind"] == BOMB].astype(int), minlength=12)


def _wall_cells(shape):
    x, y, w, h = tokens.shape_of(shape)
    lanes = range(max(x, 0), min(x + w, 4))
    v2 = range(3) if y == 0 and h >= 3 else range(2, 3)
    layers = set(range(max(y, 0), min(y + h, 3))) | set(v2)
    return frozenset(ly * 4 + lx for lx in lanes for ly in layers), frozenset(ly * 4 + lx for lx in lanes for ly in range(3))


WALL_CELLS = [_wall_cells(sh) for sh in range(tokens.N_SHAPES)]


def _wall_head(shape):
    x, y, w, h = tokens.shape_of(shape)
    if not (y <= 1 and y + h >= 2):
        return 0
    return sum(1 << (lane - 1) for lane in (1, 2) if x <= lane < x + w)


WALL_HEAD = [_wall_head(sh) for sh in range(tokens.N_SHAPES)]
WALL_CAP, FEW_NOTES = 105, 4
DIP, BUSY, STEADY_AFTER, DIP_TRIES = 0.6, 0.95, 3, 3
PLAN_CFG = 0.0
BOMB_CAP_STARS, BOMB_CAP = [2, 6, 10, 14], [64, 178, 300, 474]
BOMB_RATIO = 1.0
BOMB_CELL_CAP = [37, 64, 95, 115]
MAX_LOOP = 16
LOOP_PERIODS = 32
LOOP_MATCH = 0.85
LOOP_PENALTY = 4.0
VARIETY_BEATS, VARIETY_NOTES, VARIETY_FLOOR, VARIETY_PENALTY = 16, 32, 12, 0.75
DRY_BASE, DRY_ALLOWED, DRY_WINDOW, DRY_GROW = 1.75, 6, 512, 8
XTC_THRESHOLD = 0.1
NO_NOTE = 108
TURN_PENALTY, TURN_RATE = 2.5, 0.05
HOLD, HOLD_FROM, HOLD_GROW, HOLD_MAX = 0.0, 16, 16, 3
DOUBLE_SHARE, DOUBLE_FROM, DOUBLE_PUSH, DOUBLE_MAX = 0.35, 64, 2.0, 4.0
RATING_STARS = [1, 3, 5, 7, 9, 11, 13, 15]
TYPICAL_RATINGS = {"pass": [0.31, 1.36, 2.76, 4.37, 6.30, 8.41, 10.15, 10.55],
                   "acc": [3.28, 5.24, 6.82, 8.41, 9.94, 11.17, 12.04, 12.97],
                   "tech": [0.39, 1.59, 2.85, 3.96, 5.04, 5.70, 7.17, 9.79]}
STYLE_RATINGS = {
    "acc": ([1, 3, 5, 7], [0.31, 1.35, 2.32, 2.48], [3.28, 5.23, 6.32, 5.83], [0.39, 1.59, 2.46, 2.33]),
    "tech": ([5, 7, 9, 11, 13, 15], [3.49, 4.32, 5.20, 7.05, 9.29, 10.01], [7.09, 8.84, 10.10, 11.27, 12.12, 13.08],
             [6.33, 7.00, 7.33, 8.19, 9.12, 10.40]),
    "midspeed": ([5, 7, 9, 11, 13], [3.40, 4.51, 6.60, 8.01, 9.14], [7.43, 8.43, 9.89, 11.27, 12.60],
                 [3.32, 3.90, 4.61, 5.19, 6.58]),
    "speed": ([9, 11, 13], [9.74, 10.58, 11.55], [9.61, 10.99, 11.89], [2.79, 3.69, 4.77]),
}
TYPICAL_NPS = [1.18, 2.55, 3.91, 5.24, 6.80, 8.44, 9.64, 10.26]
STYLE_NPS = {"acc": [1.18, 2.55, 3.56, 3.45], "tech": [3.73, 4.96, 5.87, 7.21, 8.83, 9.74],
             "midspeed": [4.54, 5.39, 7.16, 8.75, 11.44], "speed": [7.97, 9.74, 11.12]}


def typical_ratings_for(stars, style=None, every=False):
    if style in STYLE_RATINGS and STYLE_RATINGS[style][0][0] - 1 <= stars <= STYLE_RATINGS[style][0][-1] + 1:
        xs, *vals = STYLE_RATINGS[style]
        out = {k: float(np.interp(stars, xs, v)) for k, v in zip(("pass", "acc", "tech"), vals)}
        return out | {"nps": float(np.interp(stars, xs, STYLE_NPS[style]))}
    out = {k: float(np.interp(stars, RATING_STARS, v)) for k, v in TYPICAL_RATINGS.items()
           if every or style is not None or k == "pass"}
    return out | {"nps": float(np.interp(stars, RATING_STARS, TYPICAL_NPS))}
ARC_PUSH = 3.0
ARC_CAP, ARC_SPAN = 0.2, 16


class LoopGuard:

    def __init__(self, max_loop=MAX_LOOP, dry=0.0, copy=None):
        self.limit = max_loop * TICKS
        self.dry, self.copy = dry, copy
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
            if self.copy and self.copy[0] <= self.ticks[len(self.ticks) - n + j] < self.copy[1]:
                continue
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
                 max_loop=MAX_LOOP, behind=TICKS // 2, dry=0.0, turn=0.0, copy=None, hold=0.0, times=None,
                 double_gap=0, cell_gap=0, swing_gap=SWING_GAP, uncross=False, near_gap=0, wall_gap=0):
        self.end, self.swings, self.objects, self.turn, self.hold = end_tick, swings, objects, turn, hold
        self.times = None if times is None else np.asarray(times, np.int64)
        self.time_set = set() if times is None else set(self.times.tolist())
        self.behind = behind
        self.wall_gap = wall_gap
        self.double_gap, self.double = double_gap, None
        self.swing_gap = swing_gap
        self.uncross = uncross
        self.cell_gap, self.cell_tick = cell_gap, {}
        self.bomb_tick = {}
        self.near_gap, self.cols = near_gap, {0: None, 1: None}
        self.moments_n = self.doubles_n = 0
        self.head = {0: None, 1: None}
        self.rest = {0: None, 1: None}
        self.owe = {0: False, 1: False}
        self.loops = LoopGuard(max_loop, dry, copy)
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
        self.note_ticks, self.arc_ticks = [], []
        self.objects_now = False
        self.walls = []
        self.pending_wall = None
        self.need_time = False
        self.swing_dir = {0: None, 1: None}
        self.last_swing = {0: None, 1: None}
        self.hit = self.run_gap = self.run_from = None
        for e in prefix_events:
            if self.tick != int(e["tick"]):
                self._new_moment(int(e["tick"]))
            h, k = int(e["hand"]), int(e["kind"])
            if k == NOTE:
                self.loops.add(self.tick, h, int(e["cell"]), int(e["dir"]), check=False)
                self.notes_now[h].append(int(e["cell"]) * 9 + int(e["dir"]))
                self.opened[h] = True
                self._ahead(h, int(e["cell"]))
                if self.first_dir[h] is None:
                    self.first_dir[h] = int(e["dir"])
                if e["dir"] != 8:
                    self.swing_dir[h] = (int(e["dir"]), self.tick)
                self.cells.add((h, int(e["cell"])))
                self.cell_tick[int(e["cell"])] = self.tick
                self._col(h, int(e["cell"]))
                self._rest(h, int(e["dir"]))
                self.stack[h] += 1
                self.note_ticks.append(self.tick)
                self._hit()
            elif k == ARC:
                self.arc_ticks.append(self.tick)
            elif k == SWING:
                same = self.face[h] == bool(e["forehand"]) and not e["air"]
                self.face[h], self.air[h], self.swing_tick[h] = bool(e["forehand"]), bool(e["air"]), self.tick
                if self.swing_tick[1 - h] == self.tick:
                    self.double = self.tick
                self.first_dir[h] = None
                self._next_swing(h, bool(e["air"]) or same)
                self._swung(h, bool(e["air"]))
            elif k == WALL:
                self._add_wall(int(e["shape"]), int(e["dur"]))
            elif k == BOMB:
                self.bomb_tick[int(e["cell"])] = self.tick
                self._bomb(int(e["cell"]))
        self.loops.check()
        self.after_time = False

    def _next_swing(self, h, air):
        self.last_swing[h] = None if air else self.swing_dir[h]
        self.swing_dir[h] = None

    def turn_penalty(self):
        h = self.need_note
        if not self.turn or h is None or self.last_swing[h] is None:
            return []
        d0, t = self.last_swing[h]
        amount = self.turn / max(1.0, (self.tick - t) / TICKS)
        back = (DIR_ANGLE[d0] + 180) % 360
        dirs = [d for d in range(8) if abs((DIR_ANGLE[d] - back + 180) % 360 - 180) >= 90]
        return [(VOCAB.note(h, cell, d), amount) for cell in range(12) for d in dirs]

    def _rest(self, h, d):
        if d in DOWN_DIRS or d == 8 and self.face[h]:
            self.rest[h] = 0
        elif d in UP_DIRS or d == 8 and self.face[h] is False:
            self.rest[h] = 2
        else:
            self.rest[h] = None

    def _swung(self, h, air):
        self.rest[h] = (None if self.rest[h] is None else 2 - self.rest[h]) if air else None
        self.owe[h] = False

    def _bomb(self, cell):
        for h in (0, 1):
            if (self.rest[h] is not None and cell // 4 == self.rest[h] and cell % 4 in HAND_COLS[h]
                    and not (self.swing_tick[h] == self.tick and not self.air[h])):
                self.owe[h] = True

    def late(self, h):
        return self.owe[h] and self.prev_moment[h] is not None and self.tick - self.prev_moment[h] > TICKS

    def reset_merge(self):
        if self.tick is None or self.need_note is not None or self.need_time or self.objects_now:
            return []
        return [(VOCAB.swing(h, int(self.face[h]), 0), VOCAB.swing(h, int(not self.face[h]), 0)) for h in (0, 1)
                if self.face[h] is not None and self.swing_tick[h] != self.tick and self.late(h)]

    def _ahead(self, h, cell):
        if not self.stack[h] or self.head[h] is None or self.swing_dir[h] is None:
            self.head[h] = cell
            return
        a = np.radians(DIR_ANGLE[self.swing_dir[h][0]])
        along = lambda c: c % 4 * np.sin(a) - c // 4 * np.cos(a)
        if along(cell) > along(self.head[h]):
            self.head[h] = cell

    def slide(self, h, cells, dirs):
        if self.stack[h] or self.swing_dir[h] is None or self.head[h] is None:
            return cells, dirs
        a = DIR_ANGLE[self.swing_dir[h][0]]
        return ({c for c in cells if c != self.head[h] and _turn(_move_angle(self.head[h], c), a) <= SLIDE_TURN},
                {d for d in dirs if d == 8 or _turn(DIR_ANGLE[d], a) <= SLIDE_TURN})

    def double_penalty(self):
        if self.moments_n < DOUBLE_FROM or self.need_note is not None or self.tick is None:
            return []
        over = self.doubles_n / self.moments_n - DOUBLE_SHARE
        if over <= 0:
            return []
        amount = min(DOUBLE_PUSH * over / 0.1, DOUBLE_MAX)
        return [(VOCAB.swing(h, f, 0), amount) for h in (0, 1) if self.stack[1 - h] and not self.stack[h]
                for f in (0, 1)]

    def _hit(self):
        if self.hit == self.tick:
            return
        if self.hit is not None and self.tick - self.hit != self.run_gap:
            self.run_gap, self.run_from = self.tick - self.hit, self.hit
        self.hit = self.tick

    def hold_penalty(self):
        if (not self.hold or self.need_note is not None or self.need_dur or self.after_time
                or self.run_gap is None or self.run_gap > TICKS):
            return []
        over = (self.hit - self.run_from) / TICKS - HOLD_FROM
        k = self.hit + self.run_gap - self.t0
        if over < 0 or not self.tick + 1 - self.t0 <= k < self.end - self.t0:
            return []
        return [(VOCAB.time0 + k, self.hold * min(1 + over / HOLD_GROW, HOLD_MAX))]

    def begin(self, t0, tick):
        self.t0 = t0
        self._new_moment(tick)
        self.need_time = True

    def _new_moment(self, tick):
        if self.tick is not None:
            if self.stack[0] or self.stack[1]:
                self.moments_n += 1
                self.doubles_n += bool(self.stack[0] and self.stack[1])
            for h in (0, 1):
                if self.stack[h]:
                    self.prev_moment[h] = self.tick
                if self.notes_now[h]:
                    self.prev_note[h] = max(self.notes_now[h])
        self.notes_now = {0: [], 1: []}
        self.tick = tick
        self.cells, self.objects_now = set(), False
        self.stack, self.first_dir, self.arc_done = {0: 0, 1: 0}, {0: None, 1: None}, {0: False, 1: False}
        self.walls = [w for w in self.walls if w[1] + max(self.behind, self.wall_gap) > tick]

    def _add_wall(self, shape, dur):
        self.walls.append((self.tick, self.tick + dur, *WALL_CELLS[shape], WALL_HEAD[shape]))

    def _covered(self, behind=0, notes=False):
        out = set()
        for start, end, cells, lanes, _ in self.walls:
            if start <= self.tick < end:
                out |= lanes if notes else cells
            elif end <= self.tick < end + behind:
                out |= lanes
        return out

    def _free(self, behind=0, notes=False):
        out = set(range(12)) - {c for _, c in self.cells} - self._covered(behind, notes)
        if self.cell_gap:
            out -= {c for c, t in self.cell_tick.items() if 0 < self.tick - t < self.cell_gap}
            if notes:
                out -= {c for c, t in self.bomb_tick.items() if 0 < self.tick - t < self.cell_gap}
        return out

    def _col(self, h, cell):
        if self.cols[h] is not None and self.cols[h][0] == self.tick:
            self.cols[h][1].add(cell % 4)
        else:
            self.cols[h] = (self.tick, {cell % 4})

    def _sides(self, h, cells, first=True):
        other = [c % 4 for g, c in self.cells if g == 1 - h]
        near = self.cols[1 - h]
        near = near[1] if near and self.near_gap and 0 < self.tick - near[0] < self.near_gap else None
        if not other and not near:
            return cells
        own = [c % 4 for g, c in self.cells if g == h]

        def fine(c):
            if near and (c % 4 >= min(near) if h == 0 else c % 4 <= max(near)):
                return False
            if not other:
                return True
            new, lo, hi = own + [c % 4], min(other), max(other)
            if min(new) < lo < max(new) or lo < min(new) < hi:
                return False
            return not self.uncross or (c % 4 <= lo if h == 0 else c % 4 >= hi)
        kept = {c for c in cells if fine(c)}
        return kept or (cells if first else kept)

    def _line(self, h, cells):
        own = [n // 9 for n in self.notes_now[h]]
        if len(own) < 2:
            return cells
        (x0, y0), (x1, y1) = (own[0] % 4, own[0] // 4), (own[1] % 4, own[1] // 4)
        return {c for c in cells if (c % 4 - x0) * (y1 - y0) == (c // 4 - y0) * (x1 - x0)}

    def _put(self, ok, h, cells, dirs, first=False):
        other = [(o // 9, o % 9) for o in self.notes_now[1 - h]]
        pairs = [(c, d) for c in cells for d in dirs
                 if not any(CLASH[c][o][d] or CLASH[o][c][od] for o, od in other)]
        if first and not pairs:
            pairs = [(c, d) for c in cells for d in dirs]
        for c, d in pairs:
            ok[VOCAB.note(h, c, d)] = True

    def fit(self, h, dirs):
        if self.stack[h] >= MAX_STACK:
            return set()
        d0 = self.first_dir[h]
        if d0 is None or d0 == 8:
            return set(dirs)
        return {d for d in dirs if d == 8 or abs((DIR_ANGLE[d] - DIR_ANGLE[d0] + 180) % 360 - 180) <= 45}

    def between(self):
        return (self.times is not None and self.tick is not None and not self.need_time
                and self.tick not in self.time_set)

    def way_on(self):
        lo = 0 if self.tick is None else self.tick + 1 - self.t0
        j = np.searchsorted(self.times, self.t0 + max(lo, 0))
        if j < len(self.times) and self.times[j] < self.end:
            return VOCAB.time0 + int(self.times[j]) - self.t0
        return VOCAB.EOS

    def off_times(self, k):
        return (self.times is not None and VOCAB.time0 <= k < VOCAB.swing0
                and self.t0 + k - VOCAB.time0 not in self.time_set)

    def snapshot(self):
        r = copy.copy(self)
        r.prev_moment, r.prev_note = dict(self.prev_moment), dict(self.prev_note)
        return r

    def mask(self):
        ok = self._mask(waive=False)
        if self.times is not None and not ok.any():
            ok = self._mask(waive=True)
        return ok

    def _mask(self, waive):
        ok = np.zeros(VOCAB.size, bool)
        if self.need_dur:
            ok[VOCAB.dur0:VOCAB.events_end] = True
            return ok
        if self.need_note is not None:
            h = self.need_note
            dirs = FOREHAND_DIRS if self.face[h] else BACKHAND_DIRS
            if not self.opened[h]:
                dirs = dirs & OPENING_DIRS
            self._put(ok, h, self._sides(h, self._free(self.behind, notes=True)), self.fit(h, dirs), first=True)
            return ok
        lo = 0 if self.tick is None else self.tick + 1 - self.t0
        hi = self.end - self.t0
        owed = (self.times is not None and not waive and not self.need_time and self.tick in self.time_set
                and not (self.stack[0] or self.stack[1]))
        between = self.between()
        filled = self.objects_now or self.tick in (self.swing_tick[0], self.swing_tick[1])
        if not self.after_time and not owed and not (between and not filled and not waive):
            if self.times is None:
                ok[VOCAB.time0 + max(lo, 0): VOCAB.time0 + hi] = True
                ok[VOCAB.EOS] = True
            else:
                way = self.way_on()
                if self.objects & (OBJ_BOMBS | OBJ_WALLS) and not waive:
                    stop = way - VOCAB.time0 if way != VOCAB.EOS else hi
                    ok[VOCAB.time0 + max(lo, 0): VOCAB.time0 + stop] = True
                ok[way] = True
        if self.tick is None or self.need_time:
            return ok
        free, note_free = self._free(), self._free(self.behind, notes=True)
        if owed:
            pass
        elif self.objects & OBJ_BOMBS and (self.times is None or between):
            for cell in free:
                ok[VOCAB.bomb(cell)] = True
        if self.objects & OBJ_WALLS and not owed:
            bombs = {c for h, c in self.cells if h == 2}
            notes = {c for h, c in self.cells if h != 2}
            closed = 0
            for start, end, _, _, head in self.walls if self.tick is not None else ():
                if start <= self.tick < end + self.wall_gap:
                    closed |= head
            for sh in range(tokens.N_SHAPES):
                cells, lanes = WALL_CELLS[sh]
                head = WALL_HEAD[sh]
                ok[VOCAB.wall(sh)] = not (cells & bombs or lanes & notes or head == 3 or head & ~closed & 3 and closed)
        if between and not self.objects_now:
            for h in (0, 1):
                if self.swing_tick[h] != self.tick and self.face[h] is not None:
                    ok[VOCAB.swing(h, int(not self.face[h]), 1)] = True
        if self.objects_now or between:
            return ok
        if self.objects & OBJ_ARCS:
            since = self.tick - ARC_SPAN * TICKS
            room = (len(self.arc_ticks) - bisect.bisect_left(self.arc_ticks, since) + 1
                    <= ARC_CAP * (len(self.note_ticks) - bisect.bisect_left(self.note_ticks, since)))
            for h in (0, 1):
                ok[VOCAB.arc(h)] = room and self.stack[h] > 0 and self.prev_moment[h] is not None and not self.arc_done[h]
        if not self.swings:
            for h in (0, 1):
                if not self.arc_done[h]:
                    for cell in note_free:
                        for d in self.fit(h, range(9) if self.opened[h] else OPENING_DIRS):
                            ok[VOCAB.note(h, cell, d)] = True
            return ok
        no_double = (self.double is not None and self.double != self.tick
                     and self.tick - self.double < self.double_gap)
        soon = {h: not self.stack[h] and self.prev_moment[h] is not None
                and self.tick - self.prev_moment[h] < self.swing_gap for h in (0, 1)}
        for h in (0, 1):
            if self.arc_done[h]:
                continue
            if no_double and self.swing_tick[1 - h] == self.tick or soon[h]:
                pass
            elif self.swing_tick[h] != self.tick and note_free:
                if self.face[h] is None:
                    ok[VOCAB.swing(h, 1, 0)] = True
                else:
                    f = int(not self.face[h])
                    ok[VOCAB.swing(h, f, 1)] = True
                    ok[VOCAB.swing(h, 1 - f if self.late(h) else f, 0)] = True
            elif self.swing_tick[h] != self.tick and self.face[h] is not None:
                ok[VOCAB.swing(h, int(not self.face[h]), 1)] = True
            if not self.air[h] and self.face[h] is not None:
                self._put(ok, h, *self.slide(h, self._line(h, self._sides(h, note_free, first=False)),
                                             self.fit(h, FOREHAND_DIRS if self.face[h] else BACKHAND_DIRS)))
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
            same = self.face[h] == bool(j // 2 % 2) and not j % 2
            self.face[h], self.air[h], self.swing_tick[h] = bool(j // 2 % 2), bool(j % 2), self.tick
            if self.swing_tick[1 - h] == self.tick:
                self.double = self.tick
            self.need_note = None if self.air[h] else h
            self.first_dir[h] = None
            self._next_swing(h, self.air[h] or same)
            self._swung(h, self.air[h])
        elif VOCAB.note0 <= k < VOCAB.note_end:
            j = k - VOCAB.note0
            h = j // 108
            self.loops.add(self.tick, h, j // 9 % 12, j % 9)
            self.notes_now[h].append(j % 108)
            self.cells.add((h, j // 9 % 12))
            self.cell_tick[j // 9 % 12] = self.tick
            self._col(h, j // 9 % 12)
            self._ahead(h, j // 9 % 12)
            self._rest(h, j % 9)
            self.opened[h] = True
            self.stack[h] += 1
            if self.first_dir[h] is None:
                self.first_dir[h] = j % 9
            if j % 9 != 8:
                self.swing_dir[h] = (j % 9, self.tick)
            self.need_note = None
            self.note_ticks.append(self.tick)
            self._hit()
        elif VOCAB.arc0 <= k < VOCAB.bomb0:
            self.arc_done[k - VOCAB.arc0] = True
            self.arc_ticks.append(self.tick)
        elif VOCAB.bomb0 <= k < VOCAB.wall0:
            self.cells.add((2, k - VOCAB.bomb0))
            self.bomb_tick[k - VOCAB.bomb0] = self.tick
            self._bomb(k - VOCAB.bomb0)
            self.objects_now = True
        elif VOCAB.wall0 <= k < VOCAB.dur0:
            self.pending_wall, self.need_dur, self.objects_now = k - VOCAB.wall0, True, True
        elif VOCAB.dur0 <= k < VOCAB.events_end:
            self._add_wall(self.pending_wall, int(tokens.DURATIONS[k - VOCAB.dur0]))
            self.need_dur = False


try:
    from . import _rules as NATIVE
except ImportError:
    NATIVE = None
NATIVE_RULES = True
_NATIVE_TABLES, _NATIVE_CONSTS = {}, {}


def native_consts():
    if not _NATIVE_TABLES:
        _NATIVE_TABLES.update(
            wall_cells=[(sorted(c), sorted(lanes)) for c, lanes in WALL_CELLS], wall_head=list(WALL_HEAD),
            durations=[int(d) for d in tokens.DURATIONS],
            dir_angle=[DIR_ANGLE[d] for d in range(8)],
            clash=[CLASH[c][o][d] for c in range(12) for o in range(12) for d in range(9)],
            dir_sin=[float(np.sin(np.radians(DIR_ANGLE[d]))) for d in range(8)],
            dir_cos=[float(np.cos(np.radians(DIR_ANGLE[d]))) for d in range(8)])
    scalars = dict(
        time0=VOCAB.time0, swing0=VOCAB.swing0, note0=VOCAB.note0, arc0=VOCAB.arc0, bomb0=VOCAB.bomb0,
        wall0=VOCAB.wall0, dur0=VOCAB.dur0, events_end=VOCAB.events_end, size=VOCAB.size, eos=VOCAB.EOS,
        ticks=TICKS, kinds=(SWING, NOTE, ARC, BOMB, WALL), slide_turn=SLIDE_TURN,
        forehand=tuple(sorted(FOREHAND_DIRS)), backhand=tuple(sorted(BACKHAND_DIRS)),
        opening=tuple(sorted(OPENING_DIRS)), down=tuple(sorted(DOWN_DIRS)), up=tuple(sorted(UP_DIRS)),
        hand_cols=(HAND_COLS[0], HAND_COLS[1]), max_stack=MAX_STACK, arc_cap=float(ARC_CAP), arc_span=ARC_SPAN,
        loop_periods=LOOP_PERIODS, loop_match=float(LOOP_MATCH), loop_penalty=float(LOOP_PENALTY),
        variety_beats=VARIETY_BEATS, variety_notes=VARIETY_NOTES, variety_floor=VARIETY_FLOOR,
        variety_penalty=float(VARIETY_PENALTY), dry_base=float(DRY_BASE), dry_allowed=DRY_ALLOWED,
        dry_window=DRY_WINDOW, dry_grow=DRY_GROW, double_share=float(DOUBLE_SHARE), double_from=DOUBLE_FROM,
        double_push=float(DOUBLE_PUSH), double_max=float(DOUBLE_MAX), hold_from=float(HOLD_FROM),
        hold_grow=float(HOLD_GROW), hold_max=float(HOLD_MAX), obj_walls=OBJ_WALLS, obj_bombs=OBJ_BOMBS,
        obj_arcs=OBJ_ARCS, no_note=NO_NOTE)
    key = tuple(scalars.items())
    if key not in _NATIVE_CONSTS:
        slide = [_turn(_move_angle(a, c), DIR_ANGLE[d]) <= SLIDE_TURN for a in range(12) for c in range(12) for d in range(8)]
        _NATIVE_CONSTS[key] = NATIVE.NativeConsts(dict(scalars, **_NATIVE_TABLES, slide_ok=slide))
    return _NATIVE_CONSTS[key]


def native_rules(prefix_events, end_tick, swings, objects, max_loop, behind, dry, turn, copy, hold, times,
                 double_gap, cell_gap, swing_gap, uncross, near_gap, wall_gap):
    e = prefix_events
    return NATIVE.Rules(native_consts(), e["tick"], e["kind"], e["hand"], e["forehand"], e["air"], e["cell"],
                        e["dir"], e["shape"], e["dur"], int(end_tick), bool(swings), int(objects), int(max_loop),
                        int(behind), float(dry), float(turn), None if copy is None else (int(copy[0]), int(copy[1])),
                        float(hold), None if times is None else [int(t) for t in times], int(double_gap),
                        int(cell_gap), int(swing_gap), bool(uncross), int(near_gap), int(wall_gap))


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


TOGETHER = 0
_shared_lock = threading.Lock()


@contextlib.contextmanager
def together(n):
    global TOGETHER
    before, TOGETHER = TOGETHER, n
    try:
        yield
    finally:
        TOGETHER = before


def shared_step(model, per, mem_len):
    key = (TOGETHER, per, mem_len)
    with _shared_lock:
        steps = model.__dict__.setdefault("_shared", {})
        have = steps.get(key)
        if have is None:
            for k in [k for k, v in steps.items() if len(v.free) == v.groups]:
                del steps[k]
            have = SharedStep(model, *key)
            with GATE.writing():
                have.capture()
            steps[key] = have
        return have


def give_graph(model, graph):
    with _pool_lock:
        model.__dict__.setdefault("_graphs", {}).setdefault((graph.tok.shape[0], graph.mk[0].shape[2]), []).append(graph)


@torch.no_grad()
def sample_window(model, audio, prefixes, weights, rules, cfg, temp, top_p, dev, xtc=0.0, min_p=0.0, arc_boost=0.0,
                  hooks=(), gen=None, rng=None, dot_bias=0.0):
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
    shared = shared_step(model, len(rows), mem_len) if use_graph and TOGETHER > 1 else None
    mine = shared.take(len(rows)) if shared is not None else None
    graph = take_graph(model, len(rows), mem_len) if use_graph and mine is None else None
    try:
        with GATE.reading():
            if mine is not None:
                shared.join(mine)
            w = torch.tensor(weights, dtype=torch.float32)[:, None, None]
            w_dev = w.to(dev)
            cache, step = {}, torch.tensor(rows, device=dev)
            host = torch.zeros(len(rows), 1, dtype=torch.long)
            host = host.pin_memory() if use_graph else host

            def last():
                host.numpy()[:, 0] = [r[-1] for r in rows]
                return host.to(dev, non_blocking=True)
            first = True
            tentative = {}

            def guide(lg):
                if guided:
                    u = lg[k * n:]
                    lg = u + cfg * ((w if lg.device.type == "cpu" else w_dev) * (lg[:k * n].view(k, n, -1) - u)).sum(0)
                return lg

            def launch():
                if mine is not None:
                    if first:
                        lg = guide(shared.prefill(mine, memory, step).float().cpu())
                        return lambda: lg
                    turn = shared.ask(mine, [r[-1] for r in rows], len(rows[0]) - 1)
                    return lambda: guide(shared.result(mine, turn))
                if graph is None:
                    lg = model.decode_step(memory, step, cache).float()
                else:
                    lg = (graph.prefill(memory, step) if first else graph.step(step)).float().clone()
                lg = guide(lg)
                return lambda: lg.cpu()

            def logits():
                return launch()()

            native = NATIVE is not None and isinstance(rules[0], NATIVE.Rules)

            def prep():
                if native:
                    return NATIVE.prep(rules, done.tolist(), vocab)
                allowed = torch.from_numpy(np.stack([eos if done[i] else r.mask()[:vocab] for i, r in enumerate(rules)]))
                pen = [(i, tok, amount) for i, r in enumerate(rules) if not done[i]
                       for tok, amount in r.loops.penalty(r) + r.turn_penalty() + r.hold_penalty() + r.double_penalty()]
                return allowed, pen, [r.reset_merge() for r in rules]

            def draw_python(lg, prepped):
                allowed, pen, merges = prepped
                bp = (torch.softmax(lg / temp, -1) * ~allowed).sum(1).tolist()
                if arc_boost:
                    lg[:, VOCAB.arc0:VOCAB.arc0 + 2] += arc_boost
                if dot_bias:
                    lg[:, VOCAB.note0 + 8:VOCAB.note0 + 24 * 9:9] -= dot_bias
                if pen:
                    ri, ti, amt = zip(*pen)
                    lg.index_put_((torch.tensor(ri), torch.tensor(ti)), -torch.tensor(amt, dtype=lg.dtype), accumulate=True)
                for hook in hooks:
                    hook(lg, rules)
                for i, r in enumerate(rules):
                    for to, fr in merges[i]:
                        lg[i, to] = torch.logsumexp(lg[i, [to, fr]], 0)
                    if not done[i] and r.times is not None and r.tick is not None:
                        way = r.way_on()
                        if allowed[i, way]:
                            on = torch.cat([lg[i, VOCAB.time0:VOCAB.swing0], lg[i, VOCAB.EOS:VOCAB.EOS + 1]])
                            shut = ~torch.cat([allowed[i, VOCAB.time0:VOCAB.swing0], allowed[i, VOCAB.EOS:VOCAB.EOS + 1]])
                            lg[i, way] = torch.logsumexp(torch.cat([lg[i, way:way + 1], on[shut]]), 0)
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
                cum = torch.cumsum(sp, -1)
                pick = torch.searchsorted(cum, torch.rand(len(cum), 1, generator=gen) * cum[:, -1:]).clamp_(max=cum.shape[1] - 1)
                ks = si.gather(1, pick)[:, 0].tolist()
                return ks, bp

            while len(rows[0]) < max_len and not done.all():
                pending = launch()
                first = False
                early = None if tentative else prep()
                lg = pending()
                if tentative:
                    p = torch.softmax(lg / temp, -1)
                    air = p[:, VOCAB.swing0 + 1:VOCAB.note0:2].sum(1)
                    obj = ((p[:, VOCAB.bomb0:VOCAB.dur0].sum(1) + air) / p[:, VOCAB.swing0:VOCAB.events_end].sum(1)).tolist()
                    back = [i for i in tentative if rng.random() >= obj[i]]
                    for i in back:
                        rules[i] = tentative[i]
                        t = rules[i].way_on()
                        if t == VOCAB.EOS:
                            done[i] = True
                        else:
                            rules[i].push(t)
                        for j in range(i, len(rows), n):
                            rows[j][-1] = t
                    tentative = {}
                    if back:
                        step = last() if mine is None else None
                        if mine is not None:
                            pass
                        elif graph is None:
                            for c in (v for key, v in cache.items() if isinstance(key, int)):
                                c["k"], c["v"] = c["k"][:, :, :-1], c["v"][:, :, :-1]
                            cache["len"] -= 1
                        else:
                            graph.len -= 1
                        lg = logits()
                if native:
                    st, a = early or prep(), lg.numpy()
                    bp = NATIVE.adjust(st, a, temp, arc_boost, dot_bias, VOCAB.note0, VOCAB.arc0)
                    for hook in hooks:
                        hook(lg, rules)
                    cut = [xtc > 0 and not done[i] and r.need_note is not None and rng.random() < xtc
                           for i, r in enumerate(rules)]
                    ks = NATIVE.finish(st, a, temp, top_p, min_p, torch.rand(len(rules), generator=gen).tolist(),
                                       cut, XTC_THRESHOLD, VOCAB.time0, VOCAB.swing0, VOCAB.EOS)
                else:
                    ks, bp = draw_python(lg, early or prep())
                for i, t in enumerate(ks):
                    if not done[i]:
                        blocked[i].append(bp[i])
                        if t == VOCAB.EOS:
                            done[i] = True
                        else:
                            r = rules[i]
                            if r.off_times(t):
                                tentative[i] = r.snapshot()
                            r.push(t)
                    for j in range(i, len(rows), n):
                        rows[j].append(t)
                step = last() if mine is None else None
            new = []
            for i in range(n):
                t = rows[i][start:]
                new.append(t[:t.index(VOCAB.EOS)] if VOCAB.EOS in t else t)
            return new, blocked
    finally:
        if mine is not None:
            shared.give(mine)
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


TAIL_DB, TAIL_GRACE = 20.0, 1.0


def song_end(logmel, fps):
    from scipy.ndimage import uniform_filter1d
    power = (10 ** (np.asarray(logmel, np.float32) / 10)).sum(1)
    db = uniform_filter1d(10 * np.log10(power + 1e-10), max(int(fps), 1))
    loud = np.flatnonzero(db >= np.percentile(db, 75) - TAIL_DB)
    return (loud[-1] if len(loud) else len(db)) / fps + TAIL_GRACE


def song_start(logmel, fps):
    from scipy.ndimage import uniform_filter1d
    power = (10 ** (np.asarray(logmel, np.float32) / 10)).sum(1)
    db = uniform_filter1d(10 * np.log10(power + 1e-10), max(int(fps), 1))
    loud = np.flatnonzero(db >= np.percentile(db, 75) - TAIL_DB)
    return (loud[0] if len(loud) else 0) / fps


def read_checkpoint(path):
    path = Path(path)
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        return dict(json.loads(path.with_suffix(".json").read_text()), model=load_file(path))
    return torch.load(path, map_location="cpu")


STEP_BF16 = True


def load_model(path, dev="cuda"):
    torch.backends.cuda.matmul.allow_tf32 = True
    ck = read_checkpoint(path)
    a = ck["args"]
    vocab, max_tokens = ck["model"]["tok.weight"].shape[0], ck["model"]["dec_pos"].shape[0]
    model = Generator(vocab, d=a["d"], enc_layers=a["layers"], dec_layers=a["layers"], heads=a["d"] // 64,
                      conv_blocks=a.get("conv_blocks", 0), global_cond=a.get("global_cond", False),
                      max_tokens=max_tokens, memory=a.get("memory", False), mem_align=a.get("mem_align", False),
                      per_beat=a.get("per_beat", 12), hits=a.get("hits", False))
    model.load_state_dict(ck["model"])
    model.controls = vocab >= VOCAB.controls_end
    model.plan = vocab >= VOCAB.size
    model.planner = None
    model.mappers = ck.get("mappers", [])
    model.swings = not a.get("no_swings", False)
    model = model.to(dev).eval()
    if STEP_BF16 and str(dev).startswith("cuda"):
        model.half_steps()
    return model


def generate(model, logmel, fps, beat_times, offset, n_beats, stars, cfg=1.5, temp=0.9, top_p=0.95,
             dev="cuda", log=None, candidates=1, controls=None, styles=None, max_loop=MAX_LOOP, dry=0.0, xtc=0.0,
             min_p=0.0, arc_boost=0.0, arc_rate=0.0, hooks=(), gen=None, rng=None, progress=None,
             turn=TURN_PENALTY, turn_rate=TURN_RATE, typical_ratings=True, music_fit=False, off_grid=0.0, repeat=0.0,
             steady=True, use_plan=True, plan_cfg=PLAN_CFG, hold=HOLD, times=None):
    if gen is not None and gen.device.type != "cpu":
        gen = torch.Generator().manual_seed(gen.initial_seed())
    plan_nps = None
    if getattr(model, "controls", False):
        c = dict(controls or {})
        if c.get("mapper") is not None:
            name = c["mapper"].strip().lower()
            if name not in model.mappers:
                raise ValueError(f"unknown mapper {c['mapper']!r}; the model knows: {', '.join(model.mappers)}")
            c["mapper"] = model.mappers.index(name)
        c.setdefault("objects", OBJ_WALLS | OBJ_BOMBS | OBJ_ARCS)
        asked = [s for s, w in (styles or {}).items() if w > 0]
        style = asked[0] if len(asked) == 1 else None
        if typical_ratings and stars is not None:
            for k, v in typical_ratings_for(stars, style, typical_ratings == "all").items():
                if c.get(k) is None:
                    c[k] = v
        cond = {f: c.get(f) for f in tokens.COND_FIELDS}
        cond["stars"] = stars
        if getattr(model, "plan", False):
            cond["wnps"] = None
        blend = {s: w / sum(styles.values()) for s, w in styles.items() if w > 0} if styles else {None: 1.0}
        conds = [dict(cond, style=s) for s in blend]
        weights, objects = list(blend.values()), cond["objects"]
        uncond = VOCAB.none_like(cond)
    else:
        conds, weights, objects, uncond = [stars], [1.0], 0, None
    ev, blocked = np.zeros(0, tokens.EVENT_DTYPE), []
    make_rules = native_rules if NATIVE_RULES and NATIVE is not None else Rules
    target = c.get("nps") if getattr(model, "controls", False) else None
    first_sec = None
    cap_stars = stars if stars is not None else BOMB_CAP_STARS[-1]
    bomb_cap = int(np.interp(cap_stars, BOMB_CAP_STARS, BOMB_CAP))
    cell_cap = int(np.interp(cap_stars, BOMB_CAP_STARS, BOMB_CELL_CAP))
    ov, bar = song_overview(logmel, fps, beat_times, offset)
    behind = behind_wall_ticks((c.get("njs") if getattr(model, "controls", False) else None) or 18.0, beat_times)
    double_gap, cell_gap, near_gap = double_gap_ticks(beat_times), cell_gap_ticks(beat_times), near_gap_ticks(beat_times)
    wall_gap = wall_switch_ticks(beat_times)
    swing_gap = swing_gap_ticks(beat_times, stars)
    uncross = stars is not None and stars < UNCROSS_STARS
    swings = getattr(model, "swings", True)
    last_note = 0
    flux = song_flux(logmel) if music_fit or off_grid else None
    families = {}
    starts = range(1 - CONTEXT, n_beats - 2, WINDOW)
    act, _ = window_activity(logmel, fps, beat_times, offset,
                             [(b + CONTEXT, min(b + CONTEXT + WINDOW, n_beats - 2)) for b in starts])
    rates = []
    for w, b in enumerate(starts):
        t0, new0 = b * TICKS, (b + CONTEXT) * TICKS
        end = min(t0 + (CONTEXT + WINDOW) * TICKS, (n_beats - 2) * TICKS)
        pb = getattr(model, "per_beat", 12)
        audio = (beat_mel(logmel, fps, beat_times, b + offset, n=(CONTEXT + WINDOW + AHEAD) * pb, per_beat=pb), ov,
                 overview_rel(bar, b))
        have = ev["tick"][ev["kind"] == NOTE]
        lo = int(have.min()) // TICKS if len(have) else None
        if getattr(model, "memory", False):
            audio += memory_input(ev, lo, b, logmel, fps, beat_times, offset, pb)
        if getattr(model, "hits", False):
            audio += (hit_roll(times, t0),)
        copy = None
        if dry and lo is not None:
            m, sim = memory_match(lo, b, logmel, fps, beat_times, offset)
            copy = (m * TICKS, (m + MEM_BEATS) * TICKS) if m is not None and sim >= REPEAT_SIM[1] else None
        turn_now = turn if turn and off_flow_rate(ev) > turn_rate else 0.0
        window_hooks = tuple(hooks) + ((off_grid_hook(t0, flux, fps, beat_times, offset, off_grid, "cpu"),) if off_grid else ())
        rep = repeat_hook(ev, b, logmel, fps, beat_times, offset, repeat, "cpu", families) if repeat else None
        window_hooks += (rep,) if rep else ()
        planned = None if plan_nps is None else plan_nps(new0, end, beat_times, offset)
        wconds, wweights = ([dict(c, wnps=planned) for c in conds], weights) if getattr(model, "plan", False) else (conds, weights)
        if planned is not None and plan_cfg:
            wconds = wconds + [dict(c, wnps=None) for c in conds]
            wweights = [x * (1 + plan_cfg) for x in weights] + [-x * plan_cfg for x in weights]

        def draw(hide_objects, allow=objects):
            prev = ev[(ev["tick"] >= t0) & (ev["tick"] < new0)]
            if hide_objects:
                prev = prev[(prev["kind"] != WALL) & (prev["kind"] != BOMB)]
            prefixes = [tokens.encode(prev, t0, c, beats=CONTEXT)[:-1] for c in [uncond] + wconds]
            rules = []
            for _ in range(candidates):
                r = make_rules(ev, end, swings, allow, max_loop, behind, dry, turn_now, copy, hold, times,
                               double_gap, cell_gap, swing_gap, uncross, near_gap, wall_gap)
                r.begin(t0, new0 - 1)
                rules.append(r)
            boost = arc_boost
            if arc_rate > 0:
                under = np.sum(ev["kind"] == ARC) < arc_rate * max(np.sum(ev["kind"] == NOTE), 1)
                boost = ARC_PUSH if under else -ARC_PUSH
            new, bl = sample_window(model, audio, prefixes, wweights, rules, cfg, temp, top_p, dev, xtc, min_p, boost,
                                    window_hooks, gen, rng, dot_bias=dot_bias(temp))
            got = [tokens.decode(prefixes[1] + t, t0) for t in new]
            return [g[g["tick"] >= new0] for g in got], bl

        def gap(g):
            ticks = g["tick"][g["kind"] == NOTE]
            return (ticks.min() if len(ticks) else end) - last_note

        limit = (MAX_GAP if first_sec is not None else MAX_INTRO) * TICKS
        quiet = ev[(ev["tick"] > last_note) & (ev["tick"] < new0)]
        if first_sec is not None and np.sum(quiet["kind"] == BOMB) >= BOMB_GAP_RATE * (new0 - last_note) / TICKS:
            limit = min(limit, BOMB_GAP * TICKS)

        got, bl = draw(False)
        note = " [turns]" if turn_now else ""
        if objects & (OBJ_WALLS | OBJ_BOMBS) and all(gap(g) > limit for g in got):
            got, bl = draw(True)
            note = " [gap: redrawn without the lookback walls and bombs]"
        beats = (end - new0) / TICKS
        if planned is not None:
            sec = np.diff(beat_to_sec(np.array([new0, end]) / TICKS + offset, beat_times))[0]
            level = DIP * planned * sec if steady and first_sec is not None else 0
            level = level if level >= FEW_NOTES else 0
        else:
            level = np.median(rates) * DIP * beats if steady and len(rates) >= STEADY_AFTER and act[w] >= BUSY else 0
        held = (lambda g: np.sum(g["kind"] == NOTE)) if planned is not None else hits
        tries = 0
        while level and tries < DIP_TRIES and max(held(g) for g in got) < level:
            more, more_bl = draw(False)
            got, bl, tries = got + more, list(bl) + list(more_bl), tries + 1
        note += f" [dip: {tries} more draws]" if tries else ""
        counts = np.array([np.sum(g["kind"] == NOTE) for g in got])
        walls = np.array([np.sum(g["kind"] == WALL) for g in got])
        bombs = np.array([np.sum(g["kind"] == BOMB) for g in got])
        have_b, have_n = np.sum(ev["kind"] == BOMB), np.sum(ev["kind"] == NOTE)
        stacked = np.array([cell_bombs(g).max() for g in got])
        bomb_ok = ((bombs <= bomb_cap) & (stacked <= cell_cap)
                   & (have_b + bombs <= BOMB_RATIO * (have_n + counts) + bomb_cap))
        if not bomb_ok.any():
            got, bl = draw(False, objects & ~OBJ_BOMBS)
            note += " [bombs: redrawn without]"
            counts = np.array([np.sum(g["kind"] == NOTE) for g in got])
            walls = np.array([np.sum(g["kind"] == WALL) for g in got])
            bombs = stacked = np.zeros(len(got), int); bomb_ok = np.ones(len(got), bool)
        ok = [i for i, g in enumerate(got) if gap(g) <= limit] or list(range(len(got)))
        gap_k = ok[pick_candidate(counts[ok])]
        if len(ok) < len(got) and gap_k != pick_candidate(counts):
            note += " [gap: passed over an empty candidate]"
        few = (counts < FEW_NOTES) & (np.sum(counts >= FEW_NOTES) * 2 > len(got))
        bad = few.astype(int) + (walls > WALL_CAP) + ~bomb_ok
        ok = [i for i in ok if bad[i] == bad[ok].min()]
        if level:
            ok = [i for i in ok if held(got[i]) >= level] or ok
        k = ok[pick_candidate(counts[ok])]
        if music_fit and len(ok) > 1:
            moments = np.array([len(np.unique(g["tick"][g["kind"] == NOTE])) for g in got])
            near = [i for i in ok if all(np.median(x[ok]) / 1.25 <= x[i] <= np.median(x[ok]) * 1.25
                                         or abs(x[i] - np.median(x[ok])) <= 2 for x in (counts, moments))]
            if len(near) > 1:
                k = max(near, key=lambda i: hit_strength(got[i], flux, fps, beat_times, offset))
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
        if stacked[k] > cell_cap:
            g = got[k]
            nth = np.zeros(len(g), int)
            for cell in np.unique(g["cell"][g["kind"] == BOMB]):
                m = (g["kind"] == BOMB) & (g["cell"] == cell)
                nth[m] = np.arange(1, m.sum() + 1)
            got[k] = g[nth <= cell_cap]
            note += f" [bombs cut to {cell_cap} per cell]"
        ev = np.concatenate([ev, got[k]])
        blocked += bl[k]
        if counts[k] >= FEW_NOTES:
            rates.append(hits(got[k]) / ((end - new0) / TICKS))
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


def song_flux(logmel):
    m = logmel.astype(np.float32)
    f = np.maximum(np.diff(m, axis=0), 0).sum(1)
    f = (f - f.mean()) / (f.std() + 1e-6)
    f = np.max([np.roll(f, d) for d in (-2, -1, 0, 1, 2)], 0)
    return f - f.mean()


def hit_strength(events, flux, fps, beat_times, offset):
    ticks = np.unique(events["tick"][events["kind"] == NOTE])
    f = np.round(beat_to_sec(ticks / TICKS + offset, beat_times) * fps).astype(int) - 1
    return float(flux[f[(f >= 0) & (f < len(flux))]].sum())


OFF_GRID_HIT = 1.0


def off_grid_hook(t0, flux, fps, beat_times, offset, amount, dev):
    ticks = t0 + np.arange(VOCAB.swing0 - VOCAB.time0)
    f = np.round(beat_to_sec(ticks / TICKS + offset, beat_times) * fps).astype(int) - 1
    weak = flux[np.clip(f, 0, len(flux) - 1)] < OFF_GRID_HIT
    bias = torch.zeros(VOCAB.size, device=dev)
    bias[VOCAB.time0:VOCAB.swing0] = torch.from_numpy(((ticks % (TICKS // 16) != 0) & weak) * float(amount)).to(dev)

    def hook(lg, rules):
        lg -= bias[:lg.shape[1]]
    return hook


REPEAT_FAMILY = (0.05, 0.85)
STRAIGHT, TRIPLET = 0, 1


def repeat_hook(ev, b, logmel, fps, beat_times, offset, amount, dev, families):
    have = ev["tick"][ev["kind"] == NOTE]
    if not len(have):
        return None
    start, lo = b + CONTEXT, int(have.min()) // TICKS
    first = min(lo, b)
    m, sim = best_match(beat_prints(logmel, fps, beat_times, offset, first, start + MEM_BEATS - first), first, start, lo)
    w = float(np.clip((sim - REPEAT_SIM[0]) / (REPEAT_SIM[1] - REPEAT_SIM[0]), 0, 1))
    if m is None or w == 0:
        return None
    mid = m + MEM_BEATS // 2
    fam = next((f for s0, f in families.items() if s0 <= mid < s0 + WINDOW), None)
    if fam is None:
        ph = np.unique(have[(have >= m * TICKS) & (have < (m + MEM_BEATS) * TICKS)]) % TICKS
        triplet = ph % (TICKS // 16) != 0
        straight = (ph % (TICKS // 16) == 0) & (ph % (TICKS // 2) != 0)
        if len(ph) >= 8 and triplet.mean() <= REPEAT_FAMILY[0]:
            fam = STRAIGHT
        elif triplet.sum() >= 4 and triplet.sum() / (triplet.sum() + straight.sum()) >= REPEAT_FAMILY[1]:
            fam = TRIPLET
        else:
            return None
    families[start] = fam
    t0 = b * TICKS
    p = (t0 + np.arange(VOCAB.swing0 - VOCAB.time0)) % TICKS
    other = (p % (TICKS // 16) != 0) if fam == STRAIGHT else (p % (TICKS // 16) == 0) & (p % (TICKS // 2) != 0)
    other &= np.arange(len(p)) >= start * TICKS - t0
    bias = torch.from_numpy(-amount * w * other.astype(np.float32)).to(dev)

    def hook(lg, rules):
        for i, r in enumerate(rules):
            if r.need_note is not None or r.need_dur:
                continue
            lo = 0 if r.tick is None else r.tick + 1 - r.t0
            hi = r.end - r.t0
            if lo >= hi or not bias[lo:hi].any():
                continue
            seg = lg[i, VOCAB.time0 + lo:VOCAB.time0 + hi]
            moved = seg + bias[lo:hi]
            lg[i, VOCAB.time0 + lo:VOCAB.time0 + hi] = moved - torch.logsumexp(moved, 0) + torch.logsumexp(seg, 0)
    return hook


MAP_SAMPLE_STARS, MAP_TEMP, MAP_TOP_P = [7, 11], [0.9, 1.0], [0.95, 0.98]


DOT_TEMP = 0.2


def dot_bias(temp):
    return DOT_TEMP * max(temp - 0.9, 0.0) / 0.1


def map_sampling(stars):
    s = MAP_SAMPLE_STARS[0] if stars is None else stars
    return float(np.interp(s, MAP_SAMPLE_STARS, MAP_TEMP)), float(np.interp(s, MAP_SAMPLE_STARS, MAP_TOP_P))


def pick_candidate(counts, spread=1.25):
    med = np.median(counts)
    for i, c in enumerate(counts):
        if med / spread <= c <= med * spread or abs(c - med) <= 2:
            return i
    return int(np.argmin(np.abs(counts - med)))
