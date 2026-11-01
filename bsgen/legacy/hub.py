from pathlib import Path

REPO = "cata121/bsgen-1.0"


def fetch(*names, version="1.1"):
    from huggingface_hub import hf_hub_download
    return [Path(hf_hub_download(REPO, n, revision=f"v{version}")) for n in names]


def map_model(version="1.1"):
    return fetch("map.safetensors", "map.json", version=version)[0]


def timing_model():
    return fetch("timing.safetensors", "octave.npz")[0].parent
