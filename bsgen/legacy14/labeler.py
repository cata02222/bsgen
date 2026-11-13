import numpy as np
import torch
from torch import nn

JOIN, NEW_FORE, NEW_BACK = 0, 1, 2
N_CONT = 6
SPAN = 512
DOWN, UP = {1, 6, 7}, {0, 4, 5}
BOMB, BOMB_DIR = 2, 9
NO_BOMBS = np.zeros(0, [("beat", "f8"), ("time", "f8"), ("line", "i1"), ("layer", "i1")])


def order(notes):
    return np.lexsort((notes["layer"] * 4 + notes["line"], notes["color"], notes["beat"]))


def features(notes, bombs=None):
    hand = notes["color"].astype(np.int64)
    cell = (notes["layer"] * 4 + notes["line"]).astype(np.int64)
    d = notes["cut_dir"].clip(0, 8).astype(np.int64)
    beat, sec = notes["beat"].astype(np.float64), notes["time"].astype(np.float64)
    if bombs is not None:
        hand = np.concatenate([hand, np.full(len(bombs), BOMB)])
        cell = np.concatenate([cell, bombs["layer"].clip(0, 2).astype(np.int64) * 4 + bombs["line"].clip(0, 3)])
        d = np.concatenate([d, np.full(len(bombs), BOMB_DIR)])
        beat = np.concatenate([beat, bombs["beat"].astype(np.float64)])
        sec = np.concatenate([sec, bombs["time"].astype(np.float64)])
        o = np.lexsort((hand == BOMB, beat))
        hand, cell, d, beat, sec = hand[o], cell[o], d[o], beat[o], sec[o]
    cont = np.zeros((len(hand), N_CONT), np.float32)
    for h in (0, 1) if bombs is None else (0, 1, BOMB):
        i = np.flatnonzero(hand == h)
        if len(i):
            cont[i, 0] = np.diff(beat[i], prepend=beat[i[0]] - 4)
            cont[i, 1] = np.diff(sec[i], prepend=sec[i[0]] - 2)
            cont[i, 2] = np.diff(beat[i], append=beat[i[-1]] + 4)
            cont[i, 3] = np.diff(sec[i], append=sec[i[-1]] + 2)
    cont[:, 4] = np.diff(beat, prepend=beat[0] - 4)
    cont[:, 5] = np.diff(sec, prepend=sec[0] - 2)
    cont = np.log1p(cont.clip(0, 60) * 8).astype(np.float32)
    return (hand, cell, d, cont) if bombs is None else (hand, cell, d, cont, np.flatnonzero(hand != BOMB))


def parity_rule(notes):
    out = np.zeros(len(notes), np.int64)
    last = {0: None, 1: None}
    for j in range(len(notes)):
        h, d, b = int(notes["color"][j]), int(notes["cut_dir"][j]), float(notes["beat"][j])
        if last[h] is not None and b - last[h][0] < 1 / 24:
            out[j] = JOIN
            continue
        fore = True if last[h] is None else not last[h][1]
        if d in DOWN:
            fore = True
        elif d in UP:
            fore = False
        out[j] = NEW_FORE if fore else NEW_BACK
        last[h] = (b, fore)
    return out


class Labeler(nn.Module):
    def __init__(self, d=256, layers=6, heads=4, dropout=0.1, bombs=False):
        super().__init__()
        self.bombs = bombs
        self.hand, self.cell = nn.Embedding(3 if bombs else 2, d), nn.Embedding(12, d)
        self.dir = nn.Embedding(10 if bombs else 9, d)
        self.cont = nn.Sequential(nn.Linear(N_CONT, d), nn.GELU(), nn.Linear(d, d))
        self.pos = nn.Parameter(torch.randn(SPAN, d) * 0.02)
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            d, heads, 4 * d, dropout, activation="gelu", batch_first=True, norm_first=True), layers,
            enable_nested_tensor=False)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 3))

    def forward(self, hand, cell, d, cont, pad):
        x = self.hand(hand) + self.cell(cell) + self.dir(d) + self.cont(cont) + self.pos[: hand.shape[1]]
        return self.out(self.enc(x, src_key_padding_mask=pad))


@torch.no_grad()
def predict(model, notes, dev="cuda", bombs=None):
    if model.bombs:
        hand, cell, d, cont, rows = features(notes, NO_BOMBS if bombs is None else bombs)
    else:
        hand, cell, d, cont = features(notes)
        rows = np.arange(len(notes))
    n = len(hand)
    logits = np.zeros((n, 3), np.float32)
    best = np.full(n, -1)
    starts = list(range(0, max(n - SPAN, 0) + 1, SPAN // 2))
    if starts[-1] + SPAN < n:
        starts.append(n - SPAN)
    for s in starts:
        e = min(s + SPAN, n)
        t = lambda a: torch.as_tensor(a[s:e])[None].to(dev)
        pad = torch.zeros(1, e - s, dtype=torch.bool, device=dev)
        lg = model(t(hand), t(cell), t(d), t(cont), pad)[0].float().cpu().numpy()
        centre = np.minimum(np.arange(e - s), np.arange(e - s)[::-1])
        take = centre > best[s:e]
        logits[s:e][take], best[s:e][take] = lg[take], centre[take]
    return logits[rows]


def label(model, notes, dev="cuda", quick_reset=0.0, bombs=None):
    with torch.autocast("cuda", torch.bfloat16, enabled=dev == "cuda"):
        logits = predict(model, notes, dev, bombs)
    lab = logits.argmax(1)
    rule = parity_rule(notes)
    dots = (notes["cut_dir"] == 8) & (lab != JOIN) & (rule != JOIN)
    lab[dots] = rule[dots]
    return _slow_resets(notes, lab, logits, quick_reset, bombs) if quick_reset else lab


QUICK_RESET_S, QUICK_RESET_COST = 0.5, 4.0


def _slow_resets(notes, lab, logits, quick_reset, bombs=None):
    bt = np.sort(bombs["time"]) if bombs is not None and len(bombs) else np.zeros(0)
    lab = lab.copy()
    for h in (0, 1):
        idx = np.flatnonzero(notes["color"] == h)
        starts = [k for k, i in enumerate(idx) if lab[i] != JOIN]
        if len(starts) < 2:
            continue
        first = idx[starts]
        last = idx[[s - 1 for s in starts[1:]] + [len(idx) - 1]]
        gap = notes["time"][first[1:]] - notes["time"][last[:-1]]
        bomb = np.searchsorted(bt, notes["time"][first[1:]]) > np.searchsorted(bt, notes["time"][last[:-1]], "right")
        gap[bomb] = np.inf
        lg = logits[first][:, [NEW_FORE, NEW_BACK]]
        given = (lab[first] == NEW_BACK).astype(int)
        cost = np.maximum(lg[np.arange(len(first)), given][:, None] - lg, 0.0)
        cost[np.arange(len(first)), given] = 0.0
        total, back = cost[0].copy(), np.zeros((len(first), 2), int)
        for k in range(1, len(first)):
            reset = QUICK_RESET_COST if gap[k - 1] < quick_reset else 0.0
            new = np.empty(2)
            for f in (0, 1):
                via = total + np.where(np.arange(2) == f, reset, 0.0)
                back[k, f] = int(np.argmin(via))
                new[f] = via[back[k, f]] + cost[k, f]
            total = new
        f = int(np.argmin(total))
        for k in range(len(first) - 1, -1, -1):
            lab[first[k]] = NEW_BACK if f else NEW_FORE
            f = back[k, f]
    return lab


def to_swings(notes, lab, grid_offset, ticks=48):
    sw_dtype = [("hand", "i1"), ("forehand", "?")]
    owner = np.full(len(notes), -1, np.int64)
    sw, beat = [], []
    cur = {0: None, 1: None}
    for j in range(len(notes)):
        h, b = int(notes["color"][j]), float(notes["beat"][j])
        if lab[j] == JOIN and cur[h] is not None:
            owner[j] = cur[h][0]
            continue
        fore = lab[j] != NEW_BACK if lab[j] != JOIN else (True if cur[h] is None else not cur[h][1])
        if cur[h] is not None and cur[h][1] == fore:
            mid = (cur[h][2] + b) / 2
            sw.append((h, not fore))
            beat.append(np.round((mid - grid_offset) * ticks) / ticks + grid_offset)
        sw.append((h, fore))
        beat.append(b)
        owner[j] = len(sw) - 1
        cur[h] = (len(sw) - 1, fore, b)
    return owner, np.array(sw, sw_dtype), np.array(beat, np.float64)
