import numpy as np

TICKS_PER_BEAT = 24
PLACES = ("far", "surround", "left", "right", "center")
BASIC_PLACE = (0, 1, 2, 3, 4)
NONE, OFF, ON, FLASH, FADE, TO = range(6)
ACTIONS = ("none", "off", "on", "flash", "fade", "to")
A, B, WHITE = range(3)
COLOURS = ("A", "B", "white")
MAX_SPEED = 15
SECTION_BEATS = 8


def basic_action(value):
    if value == 0:
        return OFF, A
    if not 1 <= value <= 12:
        return NONE, A
    colour = (B, A, WHITE)[(value - 1) // 4]
    return (ON, FLASH, FADE, TO)[(value - 1) % 4], colour


def basic_value(action, colour):
    if action == OFF:
        return 0
    base = {B: 0, A: 4, WHITE: 8}[colour]
    return base + {ON: 1, FLASH: 2, FADE: 3, TO: 4}[action]


def from_basic(events, n_beats, offset=0.0):
    n = int(np.ceil(n_beats * TICKS_PER_BEAT))
    m = dict(act=np.zeros((n, 5), np.int8), col=np.zeros((n, 5), np.int8), spin=np.zeros(n, np.uint8),
             zoom=np.zeros(n, np.uint8), sweep=np.full((n, 2), -1, np.int8))
    for beat, typ, value in events:
        t = int(round((beat - offset) * TICKS_PER_BEAT))
        if not 0 <= t < n:
            continue
        if typ in BASIC_PLACE:
            a, c = basic_action(int(value))
            if a != NONE:
                m["act"][t, typ], m["col"][t, typ] = a, c
        elif typ == 8:
            m["spin"][t] += 1
        elif typ == 9:
            m["zoom"][t] += 1
        elif typ in (12, 13):
            m["sweep"][t, typ - 12] = min(max(int(value), 0), MAX_SPEED)
    return m


def to_basic(m, offset=0.0):
    out = []
    for t in range(len(m["act"])):
        beat = round(t / TICKS_PER_BEAT + offset, 4)
        for p in range(5):
            a = int(m["act"][t, p])
            if a != NONE:
                out.append((beat, BASIC_PLACE[p], basic_value(a, int(m["col"][t, p]))))
        out += [(beat, 8, 0)] * int(m["spin"][t]) + [(beat, 9, 0)] * int(m["zoom"][t])
        for s in range(2):
            if m["sweep"][t, s] >= 0:
                out.append((beat, 12 + s, int(m["sweep"][t, s])))
    return out


def brightness(m, spb=0.5):
    act = m["act"]
    n = len(act)
    out = np.zeros((n, 5), np.float32)
    decay = max(int(1.0 / spb * TICKS_PER_BEAT), 1)
    for p in range(5):
        level, since, kind = 0.0, 0, OFF
        for t in range(n):
            a = act[t, p]
            if a != NONE:
                kind, since = a, 0
            if kind in (ON, TO):
                level = 1.0
            elif kind == FLASH:
                level = 1.0 if since < TICKS_PER_BEAT // 4 else 0.75
            elif kind == FADE:
                level = max(1 - since / decay, 0.0)
            else:
                level = 0.0
            out[t, p] = level
            since += 1
    return out


def sections(m, spb=0.5):
    w = SECTION_BEATS * TICKS_PER_BEAT
    k = len(m["act"]) // w
    br = brightness(m, spb)[:k * w].reshape(k, w, 5).sum(2).mean(1)
    act = m["act"][:k * w].reshape(k, w, 5)
    col = m["col"][:k * w].reshape(k, w, 5)
    lit = (act >= ON)
    busy = (act != NONE).any(2).sum(1) / SECTION_BEATS
    sw = m["sweep"][:k * w].reshape(k, w, 2)
    held = np.zeros(2)
    sweep_mean = np.zeros(k)
    for i in range(k):
        lv = []
        for t in range(w):
            for s in range(2):
                if sw[i, t, s] >= 0:
                    held[s] = sw[i, t, s]
            lv.append(held.mean())
        sweep_mean[i] = np.mean(lv)
    moves = (m["spin"][:k * w].reshape(k, w).sum(1) + m["zoom"][:k * w].reshape(k, w).sum(1)
             + (sw >= 0).sum((1, 2))) / SECTION_BEATS
    lead = np.stack([((col == c) & lit).sum((1, 2)) for c in range(3)], 1).astype(np.float32)
    lead /= np.maximum(lead.sum(1, keepdims=True), 1)
    strobe = (act == FLASH).sum((1, 2)) / SECTION_BEATS
    return dict(level=br, busy=busy, moves=moves, sweep=sweep_mean, lead=lead, strobe=strobe)
