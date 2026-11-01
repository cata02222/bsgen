import numpy as np

BACK, RING, LEFT, RIGHT, CENTER = 0, 1, 2, 3, 4
SPIN, ZOOM, LEFT_SPEED, RIGHT_SPEED = 8, 9, 12, 13
OFF = 0
ON, FLASH, FADE = 0, 1, 2
BASE = {0: 5, 1: 1}
PHRASE = 16
BREAK = 4


def light(color, kind):
    return BASE[color] + kind


def phrase_levels(n_beats, beat_times, logmel, fps):
    frames = np.clip(np.round(np.asarray(beat_times[:n_beats + 1]) * fps).astype(int), 0, len(logmel) - 1)
    loud = logmel.astype(np.float32).mean(1)
    per_beat = np.array([loud[a:max(b, a + 1)].mean() for a, b in zip(frames[:-1], frames[1:])])
    n_phrases = int(np.ceil(n_beats / PHRASE))
    per_phrase = np.array([per_beat[i * PHRASE:(i + 1) * PHRASE].mean() for i in range(n_phrases)])
    lo, hi = np.percentile(per_phrase, [35, 70])
    return np.where(per_phrase >= hi, 2, np.where(per_phrase <= lo, 0, 1))


def lightshow(notes, beat_times, logmel, fps, seed=0):
    rng = np.random.default_rng(seed)
    notes = notes[np.argsort(notes["beat"], kind="stable")]
    if not len(notes):
        return []
    beats = notes["beat"].astype(np.float64)
    end = float(beats[-1])
    levels = phrase_levels(min(int(np.ceil(end)) + 1, len(beat_times) - 1), beat_times, logmel, fps)
    level = lambda b: int(levels[min(int(b // PHRASE), len(levels) - 1)])
    ev = []

    changed = np.r_[True, levels[1:] != levels[:-1]]
    leads = (int(rng.integers(2)) + np.cumsum(changed)) % 2

    for p, lv in enumerate(levels):
        s = p * PHRASE
        if s > end:
            break
        if changed[p]:
            ev.append((s, ZOOM, 0))
            if lv >= 1:
                ev += [(s, g, light(int(leads[p]), FLASH)) for g in (BACK, RING, CENTER)]
        for k in range(0, PHRASE, 4 if lv else PHRASE):
            speed = (1, 3, 6)[lv] + int(rng.integers(3))
            ev += [(s + k, LEFT_SPEED, speed), (s + k, RIGHT_SPEED, max(speed + int(rng.integers(-1, 2)), 0))]
        step = {0: 4, 1: 2, 2: 1}[lv]
        ev += [(s + k, SPIN, 0) for k in range(0, PHRASE, step) if s + k <= end]

    moments, colors = np.unique(np.round(beats, 4), return_inverse=True)
    sides = [set(notes["color"][colors == i].tolist()) for i in range(len(moments))]
    for i, (b, side) in enumerate(zip(moments, sides)):
        lv = level(b)
        lead = int(leads[min(int(b // PHRASE), len(levels) - 1)])
        last = i + 1 == len(moments)
        gap_next = 1.0 if last else moments[i + 1] - b
        whole = abs(b - round(b)) < 1e-3
        half = abs(2 * b - round(2 * b)) < 1e-3
        bar = whole and round(b) % 4 == 0
        kind = ON if (lv == 2 and gap_next <= 0.5) else FADE
        for c in (0, 1):
            group = LEFT if c == 0 else RIGHT
            if c in side:
                ev.append((b, group, light(c, kind)))
            elif kind == ON:
                ev.append((b, group, OFF))
        if half and (lv >= 1 or whole):
            c = lead if whole else 1 - lead
            ev.append((b, CENTER, light(c, FLASH if bar and lv == 2 else kind)))
        if whole or (half and lv == 2):
            ev.append((b, RING, light(lead if whole else 1 - lead, FADE)))
        if whole and lv >= 1:
            ev.append((b, BACK, light(lead, FLASH if bar and lv == 2 else FADE)))
        elif half and lv == 2:
            ev.append((b, BACK, OFF))
        if gap_next >= BREAK:
            ev += [(b + 1, g, OFF) for g in (LEFT, RIGHT, CENTER, BACK)]
            ev += [(float(k), RING, light(lead, FADE)) for k in np.arange(np.ceil(b) + 2, b + gap_next - 0.5, 2)]

    ev += [(end + 1, g, OFF) for g in (BACK, RING, LEFT, RIGHT, CENTER)]
    ev.sort(key=lambda e: (e[0], e[1]))
    return [(round(float(b), 4), int(t), int(v)) for b, t, v in ev]
