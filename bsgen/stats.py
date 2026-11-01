import numpy as np


ANGLE = {0: 90, 1: 270, 2: 180, 3: 0, 4: 135, 5: 45, 6: 225, 7: 315}


def off_flow(n):
    turns = total = 0
    for h in (0, 1):
        m = n[n["color"] == h]
        m = m[np.argsort(m["beat"], kind="stable")]
        prev = None
        for g in np.split(m, np.flatnonzero(np.diff(m["beat"]) > 1 / 12 + 1e-6) + 1) if len(m) else []:
            d = [x for x in g["cut_dir"].tolist() if x != 8]
            if d and prev is not None:
                total += 1
                turns += abs((ANGLE[d[0]] - ANGLE[prev] + 180) % 360 - 180) <= 90
            prev = d[-1] if d else None
    return turns / max(total, 1)
