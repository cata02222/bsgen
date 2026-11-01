import io
import json
import zipfile
from dataclasses import dataclass

import numpy as np

NOTE_DTYPE = np.dtype([
    ("beat", "f8"), ("time", "f8"), ("line", "i1"), ("layer", "i1"),
    ("color", "i1"), ("cut_dir", "i1"), ("angle", "i2"),
])
BOMB_DTYPE = np.dtype([("beat", "f8"), ("time", "f8"), ("line", "i1"), ("layer", "i1")])
WALL_DTYPE = np.dtype([
    ("beat", "f8"), ("time", "f8"), ("duration", "f8"), ("line", "i1"), ("layer", "i1"),
    ("width", "i1"), ("height", "i1"),
])
ARC_DTYPE = np.dtype([
    ("beat", "f8"), ("color", "i1"), ("line", "i1"), ("layer", "i1"), ("dir", "i1"), ("mult", "f4"),
    ("tail_beat", "f8"), ("tail_line", "i1"), ("tail_layer", "i1"), ("tail_dir", "i1"), ("tail_mult", "f4"),
    ("mid", "i1"),
])


class MapError(ValueError):
    pass


class TempoMap:

    def __init__(self, bpm: float, changes: list[tuple[float, float]] = ()):
        pts = sorted((b, m) for b, m in changes if m > 0 and b >= 0)
        if not pts or pts[0][0] > 0:
            pts.insert(0, (0.0, bpm))
        self.beats = np.array([b for b, _ in pts])
        self.bpms = np.array([m for _, m in pts])
        seg = np.diff(self.beats) * 60.0 / self.bpms[:-1]
        self.secs = np.concatenate([[0.0], np.cumsum(seg)])

    def to_seconds(self, beat):
        beat = np.asarray(beat, np.float64)
        i = np.clip(np.searchsorted(self.beats, beat, "right") - 1, 0, None)
        return self.secs[i] + (beat - self.beats[i]) * 60.0 / self.bpms[i]

    def to_beats(self, sec):
        sec = np.asarray(sec, np.float64)
        i = np.clip(np.searchsorted(self.secs, sec, "right") - 1, 0, None)
        return self.beats[i] + (sec - self.secs[i]) * self.bpms[i] / 60.0

    @property
    def has_changes(self) -> bool:
        return len(self.bpms) > 1


@dataclass
class Difficulty:
    characteristic: str
    difficulty: str
    version: str
    njs: float
    njs_offset: float
    tempo: TempoMap
    notes: np.ndarray
    bombs: np.ndarray
    walls: np.ndarray
    arcs: np.ndarray
    n_arcs: int
    n_chains: int


@dataclass
class MapInfo:
    song_name: str
    song_author: str
    mapper: str
    bpm: float
    song_offset: float
    audio_file: str
    audio_data_file: str | None
    difficulties: list[dict]


def _member(z: zipfile.ZipFile, name: str) -> str:
    want = name.lower()
    for n in z.namelist():
        if n.lower() == want or n.lower().rsplit("/", 1)[-1] == want:
            return n
    raise MapError(f"{name} not in zip")


def _json(z: zipfile.ZipFile, name: str) -> dict:
    raw = z.read(_member(z, name))
    return json.loads(raw.decode("utf-8-sig"))


def read_info(z: zipfile.ZipFile) -> MapInfo:
    d = _json(z, "Info.dat")
    if "_difficultyBeatmapSets" in d:
        diffs = []
        for s in d["_difficultyBeatmapSets"]:
            for b in s["_difficultyBeatmaps"]:
                diffs.append(dict(
                    characteristic=s["_beatmapCharacteristicName"], difficulty=b["_difficulty"],
                    filename=b["_beatmapFilename"], njs=b.get("_noteJumpMovementSpeed", 0),
                    njs_offset=b.get("_noteJumpStartBeatOffset", 0),
                ))
        return MapInfo(
            d.get("_songName", ""), d.get("_songAuthorName", ""), d.get("_levelAuthorName", ""),
            float(d["_beatsPerMinute"]), float(d.get("_songTimeOffset", 0) or 0),
            d["_songFilename"], None, diffs,
        )
    if "difficultyBeatmaps" in d:
        diffs = [dict(
            characteristic=b["characteristic"], difficulty=b["difficulty"],
            filename=b["beatmapDataFilename"], njs=b.get("noteJumpMovementSpeed", 0),
            njs_offset=b.get("noteJumpStartBeatOffset", 0),
        ) for b in d["difficultyBeatmaps"]]
        song, audio = d.get("song", {}), d["audio"]
        mappers = [m for b in d["difficultyBeatmaps"] for m in b.get("beatmapAuthors", {}).get("mappers", [])]
        return MapInfo(
            song.get("title", ""), song.get("author", ""), ", ".join(dict.fromkeys(mappers)),
            float(audio["bpm"]), 0.0, audio["songFilename"], audio.get("audioDataFilename"), diffs,
        )
    raise MapError("unknown Info.dat format")


def _v4_tempo_changes(audio: dict, bpm: float) -> list[tuple[float, float]]:
    freq = audio.get("songFrequency") or 44100
    out = []
    for r in audio.get("bpmData", []):
        si, ei, sb, eb = r["si"], r["ei"], r["sb"], r["eb"]
        if ei > si and eb > sb:
            out.append((float(sb), (eb - sb) / ((ei - si) / freq) * 60.0))
    return out


def read_difficulty(z: zipfile.ZipFile, info: MapInfo, characteristic: str, difficulty: str) -> Difficulty:
    meta = next((d for d in info.difficulties
                 if d["characteristic"].lower() == characteristic.lower()
                 and d["difficulty"].lower() == difficulty.lower()), None)
    if meta is None:
        raise MapError(f"no {characteristic}/{difficulty} in Info.dat")
    d = _json(z, meta["filename"])
    version = str(d.get("version") or d.get("_version") or "2")

    changes: list[tuple[float, float]] = []
    n_arcs = n_chains = 0
    arcs: list = []
    if version.startswith("2"):
        notes = [(n["_time"], n["_lineIndex"], n["_lineLayer"], n["_type"], n["_cutDirection"], 0)
                 for n in d.get("_notes", [])]
        bombs = [(t, x, y) for t, x, y, c, _, _ in notes if c == 3]
        notes = [n for n in notes if n[3] in (0, 1)]
        walls = []
        for o in d.get("_obstacles", []):
            full = o.get("_type", 0) == 0
            walls.append((o["_time"], o["_duration"], o["_lineIndex"], 0 if full else 2, o["_width"], 5 if full else 3))
        changes = [(e["_time"], e.get("_floatValue", 0)) for e in d.get("_events", []) if e.get("_type") == 100]
        arcs = [(a.get("_headTime", 0), a.get("_colorType", 0), a.get("_headLineIndex", 0), a.get("_headLineLayer", 0),
                 a.get("_headCutDirection", 0), a.get("_headControlPointLengthMultiplier", 1), a.get("_tailTime", 0),
                 a.get("_tailLineIndex", 0), a.get("_tailLineLayer", 0), a.get("_tailCutDirection", 0),
                 a.get("_tailControlPointLengthMultiplier", 1), a.get("_sliderMidAnchorMode", 0))
                for a in d.get("_sliders", [])]
        n_arcs = len(arcs)
    elif version.startswith("3"):
        notes = [(n.get("b", 0), n.get("x", 0), n.get("y", 0), n.get("c", 0), n.get("d", 0), n.get("a", 0))
                 for n in d.get("colorNotes", [])]
        bombs = [(n.get("b", 0), n.get("x", 0), n.get("y", 0)) for n in d.get("bombNotes", [])]
        walls = [(o.get("b", 0), o.get("d", 0), o.get("x", 0), o.get("y", 0), o.get("w", 1), o.get("h", 1))
                 for o in d.get("obstacles", [])]
        changes = [(e.get("b", 0), e.get("m", 0)) for e in d.get("bpmEvents", [])]
        arcs = [(a.get("b", 0), a.get("c", 0), a.get("x", 0), a.get("y", 0), a.get("d", 0), a.get("mu", 1),
                 a.get("tb", 0), a.get("tx", 0), a.get("ty", 0), a.get("tc", 0), a.get("tmu", 1), a.get("m", 0))
                for a in d.get("sliders", [])]
        n_arcs, n_chains = len(arcs), len(d.get("burstSliders", []))
    elif version.startswith("4"):
        nd = d.get("colorNotesData", [])
        notes = []
        for n in d.get("colorNotes", []):
            m = nd[n.get("i", 0)]
            notes.append((n.get("b", 0), m.get("x", 0), m.get("y", 0), m.get("c", 0), m.get("d", 0), m.get("a", 0)))
        bd = d.get("bombNotesData", [])
        bombs = [(n.get("b", 0), bd[n.get("i", 0)].get("x", 0), bd[n.get("i", 0)].get("y", 0))
                 for n in d.get("bombNotes", [])]
        od = d.get("obstaclesData", [])
        walls = []
        for o in d.get("obstacles", []):
            m = od[o.get("i", 0)]
            walls.append((o.get("b", 0), m.get("d", 0), m.get("x", 0), m.get("y", 0), m.get("w", 1), m.get("h", 1)))
        if info.audio_data_file:
            changes = _v4_tempo_changes(_json(z, info.audio_data_file), info.bpm)
        ad = d.get("arcsData", [])
        arcs = []
        for a in d.get("arcs", []):
            h, t, m = nd[a.get("hi", 0)], nd[a.get("ti", 0)], ad[a.get("ai", 0)]
            arcs.append((a.get("hb", 0), h.get("c", 0), h.get("x", 0), h.get("y", 0), h.get("d", 0), m.get("m", 1),
                         a.get("tb", 0), t.get("x", 0), t.get("y", 0), t.get("d", 0), m.get("tm", 1), m.get("a", 0)))
        n_arcs, n_chains = len(arcs), len(d.get("chains", []))
    else:
        raise MapError(f"unknown difficulty version {version}")

    tempo = TempoMap(info.bpm, changes)
    na = np.array(sorted(notes), np.float64).reshape(-1, 6)
    note_arr = np.zeros(len(na), NOTE_DTYPE)
    note_arr["beat"], note_arr["line"], note_arr["layer"] = na[:, 0], na[:, 1], na[:, 2]
    note_arr["color"], note_arr["cut_dir"], note_arr["angle"] = na[:, 3], na[:, 4], na[:, 5]
    note_arr["time"] = tempo.to_seconds(note_arr["beat"]) + info.song_offset

    ba = np.array(sorted(bombs), np.float64).reshape(-1, 3)
    bomb_arr = np.zeros(len(ba), BOMB_DTYPE)
    bomb_arr["beat"], bomb_arr["line"], bomb_arr["layer"] = ba[:, 0], ba[:, 1], ba[:, 2]
    bomb_arr["time"] = tempo.to_seconds(bomb_arr["beat"]) + info.song_offset

    wa = np.array(sorted(walls), np.float64).reshape(-1, 6)
    wall_arr = np.zeros(len(wa), WALL_DTYPE)
    for i, k in enumerate(("beat", "duration", "line", "layer", "width", "height")):
        wall_arr[k] = wa[:, i]
    wall_arr["time"] = tempo.to_seconds(wall_arr["beat"]) + info.song_offset

    arc_arr = np.array([tuple(a) for a in sorted(arcs)], ARC_DTYPE) if arcs else np.zeros(0, ARC_DTYPE)

    return Difficulty(characteristic, difficulty, version, float(meta["njs"]), float(meta["njs_offset"]),
                      tempo, note_arr, bomb_arr, wall_arr, arc_arr, n_arcs, n_chains)


def read_audio(z: zipfile.ZipFile, info: MapInfo) -> tuple[np.ndarray, int]:
    import soundfile as sf
    data, sr = sf.read(io.BytesIO(z.read(_member(z, info.audio_file))), dtype="float32", always_2d=True)
    return data.mean(axis=1), sr


def open_map(path) -> zipfile.ZipFile:
    return zipfile.ZipFile(path)


def grid_offset(beats: np.ndarray) -> tuple[float, float]:
    b = np.unique(np.round(beats, 4))
    if not len(b):
        return 0.0, 0.0
    fine = float(np.angle(np.mean(np.exp(2j * np.pi * 48 * b))) / (2 * np.pi) / 48)
    best = None
    for k in range(-6, 7):
        off = fine + k / 48
        if not -0.125 <= off < 0.125:
            continue
        t = np.round((b - off) * 48).astype(int) % 48
        score = (np.sum((t % 12 == 0) | (t % 16 == 0)), np.sum(t % 24 == 0), np.sum(t == 0))
        if best is None or score > best[0]:
            best = (score, off)
    off = best[1]
    on = float(np.mean(np.abs(((beats - off) * 4 + 0.5) % 1 - 0.5) < 0.08))
    return off, on
