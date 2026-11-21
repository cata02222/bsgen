import numpy as np

ARROW_DEG = np.array([180, 0, 270, 90, 225, 135, 315, 45, 0], float)
DOT = 8
STEP = {0: (0, 1), 1: (0, -1), 2: (-1, 0), 3: (1, 0), 4: (-1, 1), 5: (1, 1), 6: (-1, -1), 7: (1, -1)}
SAME = 0.01
SWING_GAP = 0.07

VIS = {"Easy": (0.025, 1.2), "Normal": (0.05, 1.0), "Hard": (0.08, 0.75), "Expert": (0.1, 0.625),
       "ExpertPlus": (0.1, 0.5)}
WRIST = ((-155, 195), (-195, 155))
WRIST_ERR, WRIST_WARN, WRIST_TURN = 45, 90, 90
FACE_ROT = {0: ([180, 0, -90, 90, -135, 135, -45, 45], [0, 180, 90, -90, 45, -45, 135, -135]),
            1: ([-180, 0, -90, 90, -135, 135, -45, 45], [0, -180, 90, -90, 45, -45, 135, -135])}
FOREHAND = {0: {1, 3, 6, 7}, 1: {1, 2, 6, 7}}


def half_jump(njs, bpm, offset=0.0):
    beat = 60.0 / bpm
    hjd = 4.0
    while njs * beat * hjd > 17.999:
        hjd /= 2
    return max(hjd + offset, 0.25) * beat


def _dist(a, b):
    return np.hypot(a["line"] - b["line"], a["layer"] - b["layer"])


def _turn(a, b):
    return abs((a - b + 180) % 360 - 180)


def points_at(x, y, d, tx, ty, tol):
    if d == DOT:
        return False
    to = np.degrees(np.arctan2(ty - y, tx - x)) + 90
    return _turn(ARROW_DEG[d], to) <= tol


def _points_at(n, x, y, tol):
    return points_at(n["line"], n["layer"], n["cut_dir"], x, y, tol)


def clap(ax, ay, ad, bx, by, bd):
    if ad == DOT or bd == DOT:
        return False
    d = np.hypot(ax - bx, ay - by)
    at = lambda t: points_at(ax, ay, ad, bx, by, t) and points_at(bx, by, bd, ax, ay, t)
    away = lambda t: not points_at(ax, ay, ad, bx, by, t - 1e-3) and not points_at(bx, by, bd, ax, ay, t - 1e-3)
    return bool((d <= 1 and at(45)) or (d <= 1.5 and at(30)) or (d <= 2 and at(15))
                or (d <= 1 and away(135)) or (d <= 1.5 and away(160)))


def in_path(ax, ay, ad, bx, by, bd):
    if np.hypot(ax - bx, ay - by) > 2:
        return False
    return bool((ad != DOT and not points_at(ax, ay, ad, bx, by, 150))
                or (bd != DOT and not points_at(bx, by, bd, ax, ay, 150)))


def swings(notes):
    out = {0: [], 1: []}
    for h in (0, 1):
        idx = np.flatnonzero(notes["color"] == h)
        cur = []
        for i in idx:
            if cur:
                p = notes[cur[-1]]
                n = notes[i]
                gap = n["time"] - p["time"]
                window = _dist(n, p) > 1.8
                same = gap <= (0.08 if window else SWING_GAP)
                turned = gap > 0.005 and n["cut_dir"] != DOT and any(
                    notes[j]["cut_dir"] != DOT and _turn(ARROW_DEG[notes[j]["cut_dir"]], ARROW_DEG[n["cut_dir"]]) > 90
                    for j in cur)
                inline = any(_dist(n, notes[j]) <= 0.5 for j in cur)
                if not same or turned or inline:
                    out[h].append(cur)
                    cur = []
            cur.append(i)
        if cur:
            out[h].append(cur)
    return out


def vision(notes, bombs, difficulty, hjd_s):
    lo, hi = VIS.get(difficulty, VIS["ExpertPlus"])
    hi = min(hi, hjd_s)
    face = notes[(notes["layer"] == 1) & ((notes["line"] == 1) | (notes["line"] == 2))]
    flagged = [set(), set(), set()]
    for f in face:
        left = f["line"] == 1
        for kind, objs in ((0, notes), (2, bombs)):
            dt = objs["time"] - f["time"]
            half = objs["line"] <= 1 if left else objs["line"] >= 2
            for j in np.flatnonzero((dt >= lo) & (dt <= hi) & half):
                k = kind if kind == 2 else (0 if objs[j]["color"] == f["color"] else 1)
                flagged[k].add(j)
    return tuple(len(s) for s in flagged)


def handclap(notes):
    n_flag = 0
    t = notes["time"]
    for a in notes[(notes["color"] == 0) & (notes["cut_dir"] != DOT)]:
        for b in notes[(np.abs(t - a["time"]) <= SAME) & (notes["color"] == 1)]:
            if clap(a["line"], a["layer"], a["cut_dir"], b["line"], b["layer"], b["cut_dir"]):
                n_flag += 1
                break
    return n_flag


def hammer(notes, bombs):
    n_flag = 0
    for n in notes[notes["cut_dir"] != DOT]:
        near = bombs[(bombs["time"] >= n["time"] - 0.125) & (bombs["time"] <= n["time"] + 0.075)]
        for b in near:
            if _dist(n, b) <= 2 and (_points_at(n, b["line"], b["layer"], 15)
                                     or not _points_at(n, b["line"], b["layer"], 165 - 1e-3)):
                n_flag += 1
                break
    return n_flag


def hitbox_path(notes, bombs):
    by_note = by_bomb = 0
    for n in notes[notes["cut_dir"] != DOT]:
        same = notes[(np.abs(notes["time"] - n["time"]) <= SAME) & (notes["color"] != n["color"])]
        if any(in_path(n["line"], n["layer"], n["cut_dir"], o["line"], o["layer"], o["cut_dir"]) for o in same):
            by_note += 1
        bs = bombs[np.abs(bombs["time"] - n["time"]) <= SAME]
        if any(np.hypot(b["line"] - n["line"], b["layer"] - n["layer"]) <= 2
               and not _points_at(n, b["line"], b["layer"], 150) for b in bs):
            by_bomb += 1
    return by_note, by_bomb


def hitbox_stair(notes, sw):
    n_flag = 0
    last = {}
    events = sorted((notes[s[0]]["time"], h, s) for h in (0, 1) for s in sw[h])
    prev_t = {0: None, 1: None}
    for t, h, s in events:
        o = 1 - h
        if o in last:
            lt, x, y, gap = last[o]
            for i in s:
                n = notes[i]
                same_moment = abs(n["time"] - lt) <= SAME
                if 0 < n["time"] - lt < min(0.15, gap) and not same_moment and (n["line"], n["layer"]) == (x, y):
                    n_flag += 1
                    break
        end = notes[s[-1]]
        d = next((notes[i]["cut_dir"] for i in reversed(s) if notes[i]["cut_dir"] != DOT), DOT)
        gap = t - prev_t[h] if prev_t[h] is not None else 1.0
        if d != DOT:
            last[h] = (notes[s[-1]]["time"], end["line"] + STEP[d][0], end["layer"] + STEP[d][1], gap)
        else:
            last.pop(h, None)
        prev_t[h] = t
    return n_flag


def hitbox_inline(notes, njs):
    lim = 1.425 / max(njs, 1.0)
    n_flag = 0
    for i in range(len(notes)):
        n = notes[i]
        prev = notes[(notes["time"] < n["time"] - 1e-4) & (notes["time"] >= n["time"] - lim) & (notes["color"] != n["color"])]
        if len(prev) and np.any(np.hypot(prev["line"] - n["line"], prev["layer"] - n["layer"]) <= 0.5):
            n_flag += 1
    return n_flag


def wrist(notes, bombs, sw):
    err = warn = 0
    for h in (0, 1):
        state, rot = None, 0.0
        lo, hi = WRIST[h]
        prev_start = -1e9
        for s in sw[h]:
            t0 = notes[s[0]]["time"]
            st = state
            for b in bombs[(bombs["time"] > prev_start) & (bombs["time"] <= t0 + 1e-3)]:
                if b["line"] == (2 if h else 1) and b["layer"] in (0, 2):
                    st, rot = (1 if b["layer"] == 0 else 0), 0.0
            dirs = [notes[i]["cut_dir"] for i in s if notes[i]["cut_dir"] != DOT]
            prev_start = t0
            if not dirs:
                if st is not None:
                    state = 1 - st
                continue
            d = dirs[-1]
            if st is None:
                face = 0 if d in FOREHAND[h] else 1
                state, rot = face, FACE_ROT[h][face][d]
                continue
            face = 1 - st
            need = FACE_ROT[h][face][d]
            turn = abs(rot - need)
            if turn > 180 or need < lo + WRIST_ERR or need > hi - WRIST_ERR:
                err += 1
                state, rot = st, FACE_ROT[h][st][d]
                continue
            if need < lo + WRIST_WARN or need > hi - WRIST_WARN or turn > WRIST_TURN:
                warn += 1
            state, rot = face, need
    return err, warn


def bombs_near(notes, bombs, beat_s):
    n_flag = 0
    q = beat_s / 4
    for b in bombs:
        near = notes[np.abs(notes["time"] - b["time"]) <= q]
        if len(near) and np.any(np.hypot(near["line"] - b["line"], near["layer"] - b["layer"]) <= 1.0):
            n_flag += 1
    return n_flag


def fast_swings(notes, sw, bpm):
    lim = min(350.0, 2 * bpm)
    n_flag = 0
    for h in (0, 1):
        starts = np.array([notes[s[0]]["time"] for s in sw[h]])
        if len(starts) > 1:
            ebpm = 60.0 / (2 * np.maximum(np.diff(starts), 1e-6))
            n_flag += int(np.sum(ebpm > lim))
    return n_flag


def doubles(notes):
    t = np.round(notes["time"] / SAME)
    hit, both = 0, 0
    for k in np.unique(t):
        c = notes["color"][t == k]
        hit += 1
        both += (0 in c) and (1 in c)
    return both, hit


def wide_inward(notes):
    n_flag = 0
    for i in np.flatnonzero(notes["color"] == 0):
        a = notes[i]
        bs = notes[(np.abs(notes["time"] - a["time"]) <= SAME) & (notes["color"] == 1)]
        for b in bs:
            if a["line"] == 0 and b["line"] == 3 and a["cut_dir"] in (3, 5, 7) and b["cut_dir"] in (2, 4, 6):
                n_flag += 1
                break
    return n_flag


def face_punch(notes):
    top = notes[notes["layer"] == 2]
    left = top[(top["color"] == 0) & (top["line"] == 3) & np.isin(top["cut_dir"], (3, 5))]
    right = top[(top["color"] == 1) & (top["line"] == 0) & np.isin(top["cut_dir"], (2, 4))]
    return len(left) + len(right)


def crouches(walls):
    w = walls[(walls["layer"] >= 2) & (walls["line"] <= 1) & (walls["line"] + walls["width"] >= 3)
              & (walls["end"] - walls["time"] > 0)]
    spans = sorted(zip(w["time"], w["end"]))
    out = []
    for a, b in spans:
        if out and a <= out[-1][1] + 0.05:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def crouch_checks(walls, notes):
    cr = crouches(walls)
    soon = sum(1 for (a, _), (_, pe) in zip(cr[1:], cr[:-1]) if a - pe < 2.0)
    long_ = sum(1 for a, b in cr if b - a > 3.0)
    top_after = 0
    for _, b in cr:
        top_after += int(np.sum((notes["layer"] == 2) & (notes["time"] > b) & (notes["time"] <= b + 0.5)))
    return len(cr), soon, long_, top_after


def short_walls(walls):
    return int(np.sum((walls["end"] - walls["time"] > 0) & (walls["end"] - walls["time"] < 0.015)))


def all_checks(notes, bombs, walls, difficulty, njs, bpm, beat_s, njs_offset=0.0):
    notes = np.sort(notes, order="time")
    bombs = np.sort(bombs, order="time")
    sw = swings(notes)
    hjd = half_jump(njs, bpm, njs_offset) if njs > 0 else 0.5
    vs, vo, vb = vision(notes, bombs, difficulty, hjd)
    we, ww = wrist(notes, bombs, sw)
    both, hit = doubles(notes)
    nc, soon, long_, top_after = crouch_checks(walls, notes)
    hp = hitbox_path(notes, bombs)
    return {
        "notes": len(notes), "swings": len(sw[0]) + len(sw[1]),
        "vision_same": vs, "vision_other": vo, "vision_bomb": vb,
        "handclap": handclap(notes), "hammer": hammer(notes, bombs),
        "hitbox_path": hp[0], "bomb_path": hp[1], "hitbox_stair": hitbox_stair(notes, sw),
        "hitbox_inline": hitbox_inline(notes, njs or 16.0),
        "wrist_error": we, "wrist_warn": ww, "bomb_near": bombs_near(notes, bombs, beat_s),
        "fast_swing": fast_swings(notes, sw, bpm), "wide_inward": wide_inward(notes),
        "face_punch": face_punch(notes), "double_share": both / max(hit, 1),
        "crouches": nc, "crouch_soon": soon, "crouch_long": long_, "top_after_crouch": top_after,
        "short_walls": short_walls(walls), "first_s": float(min(notes["time"].min() if len(notes) else 99,
                                                                 bombs["time"].min() if len(bombs) else 99)),
        "njs": njs, "rt": hjd,
    }
