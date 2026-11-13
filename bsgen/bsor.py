import hashlib
import struct
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

MAGIC = 0x442D3D69

FRAME_DTYPE = np.dtype([("time", "<f4"), ("fps", "<i4"), ("pose", "<f4", (3, 7))])

NOTE_DTYPE = np.dtype([
    ("scoring_type", "i4"), ("line", "i1"), ("layer", "i1"), ("color", "i1"), ("cut_dir", "i1"),
    ("event_time", "f4"), ("spawn_time", "f4"), ("event_type", "i1"),
    ("speed_ok", "?"), ("dir_ok", "?"), ("saber_ok", "?"), ("too_soon", "?"),
    ("saber_speed", "f4"), ("saber_dir", "f4", 3), ("saber_type", "i1"),
    ("time_dev", "f4"), ("cut_dir_dev", "f4"),
    ("cut_point", "f4", 3), ("cut_normal", "f4", 3),
    ("cut_dist", "f4"), ("cut_angle", "f4"), ("before_rating", "f4"), ("after_rating", "f4"),
])

WALL_DTYPE = np.dtype([
    ("line", "i1"), ("type", "i1"), ("width", "i1"),
    ("energy", "f4"), ("time", "f4"), ("spawn_time", "f4"),
])

EVENT_GOOD, EVENT_BAD, EVENT_MISS, EVENT_BOMB = 0, 1, 2, 3


class BsorError(ValueError):
    pass


@dataclass
class Replay:
    info: dict
    frames: np.ndarray
    notes: np.ndarray
    walls: np.ndarray
    heights: np.ndarray
    pauses: np.ndarray
    controller_offsets: np.ndarray | None = None
    custom_data: dict[str, bytes] = field(default_factory=dict, repr=False)


class _Reader:
    def __init__(self, buf: bytes):
        self.b = buf
        self.o = 0

    def unpack(self, fmt: str):
        try:
            v = struct.unpack_from("<" + fmt, self.b, self.o)
        except struct.error as e:
            raise BsorError(f"truncated at {self.o}") from e
        self.o += struct.calcsize("<" + fmt)
        return v[0] if len(v) == 1 else v

    def string(self) -> str:
        n = self.unpack("i")
        while n < 0 or n > 300:
            self.o -= 3
            n = self.unpack("i")
        s = self.b[self.o:self.o + n].decode("utf-8", "replace")
        self.o += n
        return s

    def name(self) -> str:
        n = self.unpack("i")
        extra = 0
        if n > 0:
            while struct.unpack_from("<i", self.b, self.o + n + extra)[0] not in (5, 6, 8):
                extra += 1
        s = self.b[self.o:self.o + n + extra].decode("utf-8", "replace")
        self.o += n + extra
        return s


def _read_info(r: _Reader) -> dict:
    info = {}
    for k in ("mod_version", "game_version", "timestamp", "player_id"):
        info[k] = r.string()
    info["player_name"] = r.name()
    for k in ("platform", "tracking_system", "hmd", "controller", "hash", "song_name", "mapper", "difficulty"):
        info[k] = r.string()
    info["score"] = r.unpack("i")
    for k in ("mode", "environment", "modifiers"):
        info[k] = r.string()
    info["jump_distance"] = r.unpack("f")
    info["left_handed"] = r.unpack("?")
    info["height"] = r.unpack("f")
    info["start_time"], info["fail_time"], info["speed"] = r.unpack("3f")
    return info


def _read_frames(r: _Reader) -> np.ndarray:
    n = r.unpack("i")
    end = r.o + n * FRAME_DTYPE.itemsize
    if n < 0 or end > len(r.b):
        raise BsorError(f"bad frame count {n}")
    f = np.frombuffer(r.b, FRAME_DTYPE, n, r.o).copy()
    r.o = end
    t = f["time"]
    keep = t != 0
    keep[1:] &= t[1:] != t[:-1]
    return f[keep]


def _read_notes(r: _Reader) -> np.ndarray:
    n = r.unpack("i")
    out = np.zeros(n, NOTE_DTYPE)
    for i in range(n):
        nid, et, st, ev = r.unpack("iffi")
        e = out[i]
        e["scoring_type"] = nid // 10000
        e["line"] = nid // 1000 % 10
        e["layer"] = nid // 100 % 10
        e["color"] = nid // 10 % 10
        e["cut_dir"] = nid % 10
        e["event_time"], e["spawn_time"], e["event_type"] = et, st, ev
        if ev in (EVENT_GOOD, EVENT_BAD):
            v = r.unpack("4?f3fiff3f3f4f")
            if v[8] not in (0, 1):
                e["saber_type"] = -1
                e["saber_speed"] = e["time_dev"] = e["cut_dist"] = np.nan
                continue
            e["speed_ok"], e["dir_ok"], e["saber_ok"], e["too_soon"] = v[0:4]
            e["saber_speed"] = v[4]
            e["saber_dir"] = v[5:8]
            e["saber_type"] = v[8]
            e["time_dev"], e["cut_dir_dev"] = v[9:11]
            e["cut_point"] = v[11:14]
            e["cut_normal"] = v[14:17]
            e["cut_dist"], e["cut_angle"], e["before_rating"], e["after_rating"] = v[17:21]
        else:
            e["saber_type"] = -1
    return out


def _read_walls(r: _Reader) -> np.ndarray:
    n = r.unpack("i")
    out = np.zeros(n, WALL_DTYPE)
    for i in range(n):
        wid, energy, t, st = r.unpack("ifff")
        out[i] = (wid // 100, wid // 10 % 10, wid % 10, energy, t, st)
    return out


def read_bsor(buf: bytes) -> Replay:
    r = _Reader(buf)
    if r.unpack("I") != MAGIC:
        raise BsorError("not a bsor file")
    if (v := r.unpack("B")) != 1:
        raise BsorError(f"unsupported bsor version {v}")

    parts: dict = {}
    while r.o < len(buf):
        section = r.unpack("B")
        if section == 0:
            parts["info"] = _read_info(r)
        elif section == 1:
            parts["frames"] = _read_frames(r)
        elif section == 2:
            parts["notes"] = _read_notes(r)
        elif section == 3:
            parts["walls"] = _read_walls(r)
        elif section == 4:
            n = r.unpack("i")
            parts["heights"] = np.array([r.unpack("ff") for _ in range(n)], np.float32).reshape(n, 2)
        elif section == 5:
            n = r.unpack("i")
            parts["pauses"] = np.array([r.unpack("qf") for _ in range(n)], np.float64).reshape(n, 2)
        elif section == 6:
            parts["controller_offsets"] = np.array(r.unpack("14f"), np.float32).reshape(2, 7)
        elif section == 7:
            custom = {}
            for _ in range(r.unpack("i")):
                key = r.string()
                n = r.unpack("i")
                custom[key] = r.b[r.o:r.o + n]
                r.o += n
            parts["custom_data"] = custom
        else:
            raise BsorError(f"unknown section {section} at {r.o - 1}")

    for k in ("info", "frames", "notes"):
        if k not in parts:
            raise BsorError(f"missing {k} section")
    parts.setdefault("walls", np.zeros(0, WALL_DTYPE))
    parts.setdefault("heights", np.zeros((0, 2), np.float32))
    parts.setdefault("pauses", np.zeros((0, 2), np.float64))
    return Replay(**parts)


def load_bsor(path) -> Replay:
    with open(path, "rb") as f:
        return read_bsor(f.read())


class _Writer:
    def __init__(self):
        self.b = bytearray()

    def byte(self, v): self.b += struct.pack("<B", v)
    def int(self, v): self.b += struct.pack("<i", v)
    def float(self, v): self.b += struct.pack("<f", v)
    def bool(self, v): self.b += struct.pack("<?", bool(v))
    def floats(self, vs): self.b += struct.pack(f"<{len(vs)}f", *vs)

    def str(self, s):
        e = s.encode()
        self.int(len(e))
        self.b += e


LANES = (-0.9, -0.3, 0.3, 0.9)


def write_bsor(path, meta, pos, quat, events, fps=60):
    w = _Writer()
    rows = np.array([0.85, 1.4, 1.9]) + min(max((meta["height"] - 1.8) * 0.5, -0.2), 0.6)
    w.int(MAGIC)
    w.byte(1)
    w.byte(0)
    for s in ("0.0.1", "1.40.0", str(int(time.time())), "0", meta["player"], "bsgen", "generated", "generated",
              "generated", meta["hash"], meta["song"], meta["mapper"], meta["difficulty"]):
        w.str(s)
    w.int(meta["score"])
    for s in ("Standard", meta["environment"], ""):
        w.str(s)
    w.float(0.0)
    w.bool(False)
    w.float(meta["height"])
    w.float(0.0)
    w.float(0.0)
    w.float(0.0)
    w.byte(1)
    w.int(len(pos))
    for f in range(len(pos)):
        w.float(f / fps)
        w.int(fps)
        for dev in range(3):
            w.floats(list(pos[f, dev]) + list(quat[f, dev]))
    w.byte(2)
    w.int(len(events))
    for e in sorted(events, key=lambda e: e[6]):
        sec, x, y, c, d, kind, t, saber, before, after, dist, speed, sdir, normal = e[:14]
        w.int((e[14] if len(e) > 14 else 0) * 10000 + x * 1000 + y * 100 + c * 10 + d)
        w.float(t)
        w.float(sec)
        w.int(kind)
        if kind in (0, 1):
            w.bool(True)
            w.bool(kind == 0 or saber == c)
            w.bool(saber == c)
            w.bool(False)
            w.float(speed)
            w.floats(list(sdir))
            w.int(saber)
            w.float(t - sec)
            w.float(0.0)
            w.floats([LANES[min(max(x, 0), 3)], rows[min(max(y, 0), 2)], 0.0])
            w.floats(list(normal))
            w.float(dist)
            w.float(0.0)
            w.float(before)
            w.float(after)
    for section in (3, 4, 5):
        w.byte(section)
        w.int(0)
    Path(path).write_bytes(bytes(w.b))


def map_hash(zip_path, info):
    with zipfile.ZipFile(zip_path) as z:
        names = {n.lower(): n for n in z.namelist()}
        h = hashlib.sha1(z.read(names["info.dat"]))
        for d in info.difficulties:
            h.update(z.read(names[d["filename"].lower()]))
    return h.hexdigest().upper()
