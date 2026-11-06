from pathlib import Path

REPO = "cata121/bsgen-1.0"
VERSIONS = ["1.2"]
NEWEST = "1.2"
LEGACY = ()


def fetch(*names, version=NEWEST):
    from huggingface_hub import hf_hub_download
    from .. import hub
    hub._count_download(hf_hub_download)
    return [Path(hf_hub_download(REPO, n, revision="v1.2")) for n in names]


def map_model(version=None):
    return fetch("map.safetensors", "map.json")[0]


def player_model():
    from .. import hub
    return hub.player_model()


def labeler_model():
    return fetch("labeler.safetensors")[0]


def light_model():
    return fetch("lights.safetensors", "lights.json")[0]


def rhythm_model():
    return fetch("rhythm.safetensors", "rhythm.json")[0]


def rating_model():
    return fetch("rating.npz")[0]


def song_stars_model():
    return fetch("song_stars.npz")[0]


def timing_model():
    return fetch("timing.safetensors", "octave.npz")[0].parent
