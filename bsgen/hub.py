import os
from pathlib import Path

REPO = "cata121/bsgen-1.0"
VERSIONS = ["1.5", "1.4", "1.3", "1.2", "1.1", "1.0"]
NEWEST = VERSIONS[0]
LEGACY = ("1.4", "1.3", "1.2", "1.1", "1.0")
FIXED_STARS = ("1.1", "1.0")
MAP_NOTES = {"1.5": "newest", "1.4": "previous release", "1.3": "older release", "1.2": "older release", "1.1": "more creative, sometimes weird",
             "1.0": "first release"}


def fetch(*names, version=NEWEST):
    folder = os.environ.get("BSGEN_MODELS")
    if folder and version == NEWEST:
        return [Path(folder) / n for n in names]
    from huggingface_hub import hf_hub_download
    _count_download(hf_hub_download)
    return [Path(hf_hub_download(REPO, n, revision=f"v{version}")) for n in names]


_counted = False


def _count_download(hf_hub_download):
    global _counted
    if _counted:
        return
    _counted = True
    try:
        hf_hub_download(REPO, "config.json")
    except Exception:
        pass


LOCAL_MAP = Path(__file__).resolve().parents[1] / "models" / "map.pt"


def map_model(version=None):
    if version in (None, NEWEST) and LOCAL_MAP.exists():
        return LOCAL_MAP
    return fetch("map.safetensors", "map.json", version=version or NEWEST)[0]


LOCAL_PLAYER = LOCAL_MAP.with_name("player.pt")


def player_model():
    if LOCAL_PLAYER.exists():
        return LOCAL_PLAYER
    return fetch("player.safetensors", "player.json")[0]


LOCAL_LABELER = LOCAL_MAP.with_name("labeler.pt")


def labeler_model():
    if LOCAL_LABELER.exists():
        return LOCAL_LABELER
    return fetch("labeler.safetensors")[0]


LOCAL_LIGHTS = LOCAL_MAP.with_name("lights.pt")


def light_model():
    if LOCAL_LIGHTS.exists():
        return LOCAL_LIGHTS
    return fetch("lights.safetensors", "lights.json")[0]


LOCAL_LIGHTSEQ = LOCAL_MAP.with_name("lightseq.pt")


def lightseq_model():
    if LOCAL_LIGHTSEQ.exists():
        return LOCAL_LIGHTSEQ
    return fetch("lightseq.safetensors", "lightseq.json")[0]


LOCAL_RHYTHM = LOCAL_MAP.with_name("rhythm.pt")


def rhythm_model():
    if LOCAL_RHYTHM.exists():
        return LOCAL_RHYTHM
    return fetch("rhythm.safetensors", "rhythm.json")[0]


LOCAL_RATING = LOCAL_MAP.with_name("rating.npz")


def rating_model():
    return LOCAL_RATING if LOCAL_RATING.exists() else fetch("rating.npz")[0]


LOCAL_SONG_STARS = LOCAL_MAP.with_name("song_stars.npz")


def song_stars_model():
    return LOCAL_SONG_STARS if LOCAL_SONG_STARS.exists() else fetch("song_stars.npz")[0]


LOCAL_PATTERN_PLAN = LOCAL_MAP.with_name("pattern_plan.npz")


def pattern_plan():
    return LOCAL_PATTERN_PLAN if LOCAL_PATTERN_PLAN.exists() else fetch("pattern_plan.npz")[0]


def timing_model():
    local = LOCAL_MAP.parents[1] / "runs" / "timing"
    if (local / "timing.safetensors").exists():
        return local
    return fetch("timing.safetensors", "octave.npz")[0].parent
