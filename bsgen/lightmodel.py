import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

AUDIO_PER_BEAT = 12
TICKS_PER_BEAT = 24
TYPES = (0, 1, 2, 3, 4, 8, 9, 12, 13)
CHANNELS = len(TYPES)
LIGHT_VALUES, SPEED_VALUES = 13, 16
N_VALUES = 16
ENVS = ["DefaultEnvironment", "OriginsEnvironment", "TriangleEnvironment", "NiceEnvironment",
        "BigMirrorEnvironment", "DragonsEnvironment", "KDAEnvironment", "MonstercatEnvironment",
        "CrabRaveEnvironment", "PanicEnvironment", "RocketEnvironment", "GreenDayEnvironment",
        "GreenDayGrenadeEnvironment", "TimbalandEnvironment", "FitBeatEnvironment", "LinkinParkEnvironment",
        "BTSEnvironment", "KaleidoscopeEnvironment", "InterscopeEnvironment", "SkrillexEnvironment",
        "BillieEnvironment", "HalloweenEnvironment", "GagaEnvironment"]
OTHER = len(ENVS)
AUTO_ENVS = ["DefaultEnvironment", "BigMirrorEnvironment", "NiceEnvironment", "PanicEnvironment",
             "TriangleEnvironment", "DragonsEnvironment", "KaleidoscopeEnvironment", "BTSEnvironment",
             "OriginsEnvironment", "FitBeatEnvironment"]


def channel_value(event_type, value):
    if event_type not in TYPES:
        return -1, 0
    c = TYPES.index(event_type)
    if c < 5:
        return (c, int(value)) if 0 <= value < LIGHT_VALUES else (-1, 0)
    if c < 7:
        return c, 0
    return c, int(min(max(value, 0), SPEED_VALUES - 1))


def event_type(channel, value):
    return TYPES[channel], int(value)

WINDOW = 32
CONTEXT = 8
MELS = 80
MASK = N_VALUES + 1
UNKNOWN_ENV = OTHER + 1
BUSY_EDGES = [2, 3, 4.5, 6.5, 9, 12.5, 17]
FADE_EDGES = [0.5, 0.85]
UNKNOWN_BUSY, UNKNOWN_FADE = len(BUSY_EDGES) + 1, len(FADE_EDGES) + 1
NOTES_PER_BEAT, TYPICAL_BUSY = [0.5, 1.5, 2.5, 4.0], [3.7, 6.2, 9.0, 11.8]
MADE_BUSY = [1.5, 2.4, 2.9, 3.9, 6.5, 8.9, 8.4, 9.2]
SUB = TICKS_PER_BEAT // AUDIO_PER_BEAT


def valid_values():
    ok = np.zeros((CHANNELS, N_VALUES + 1), bool)
    ok[:, 0] = True
    ok[:5, 1:1 + LIGHT_VALUES] = True
    ok[5:7, 1] = True
    ok[7:, 1:1 + SPEED_VALUES] = True
    return ok


def note_ticks(beats, colors, n_ticks):
    r = np.zeros(n_ticks, np.uint8)
    t = np.round(np.asarray(beats) * TICKS_PER_BEAT).astype(np.int64)
    ok = (t >= 0) & (t < n_ticks)
    np.bitwise_or.at(r, t[ok], (1 << np.asarray(colors)[ok].astype(np.uint8)).astype(np.uint8))
    return r


class Lighter(nn.Module):

    def __init__(self, d=256, layers=8, heads=4):
        super().__init__()
        self.config = dict(d=d, layers=layers, heads=heads)
        T = WINDOW * TICKS_PER_BEAT
        self.audio = nn.Sequential(nn.Linear(MELS, d), nn.GELU(), nn.Linear(d, d))
        self.cell = nn.Embedding(CHANNELS * (MASK + 1), d)
        self.note = nn.Embedding(4, d)
        self.env = nn.Embedding(UNKNOWN_ENV + 1, d)
        self.busy = nn.Embedding(UNKNOWN_BUSY + 1, d)
        self.fade = nn.Embedding(UNKNOWN_FADE + 1, d)
        self.phase = nn.Embedding(4 * TICKS_PER_BEAT, d)
        self.pos = nn.Parameter(torch.zeros(T, d))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, activation="gelu", batch_first=True,
                                           norm_first=True)
        self.body = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, CHANNELS * (N_VALUES + 1))
        self.register_buffer("valid", torch.from_numpy(valid_values()), persistent=False)
        self.register_buffer("chan", torch.arange(CHANNELS) * (MASK + 1), persistent=False)

    def forward(self, cells, notes, audio, env, beat0, busy, fade):
        B, T, _ = cells.shape
        a = self.audio(audio).repeat_interleave(SUB, 1)
        ticks = torch.arange(T, device=cells.device)
        ph = (beat0[:, None] * TICKS_PER_BEAT + ticks) % (4 * TICKS_PER_BEAT)
        x = a + self.cell(cells + self.chan).sum(2) + self.note(notes) + (self.env(env) + self.busy(busy) + self.fade(fade))[:, None] + self.phase(ph) \
            + self.pos[:T]
        x = self.out(self.norm(self.body(x))).view(B, T, CHANNELS, N_VALUES + 1)
        return x.float().masked_fill(~self.valid, -1e4)


def load(path, dev="cuda"):
    path = Path(path)
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        ck = dict(json.loads(path.with_suffix(".json").read_text()), model=load_file(path))
    else:
        ck = torch.load(path, map_location="cpu")
    model = Lighter(**ck["config"]).to(dev)
    model.load_state_dict(ck["model"])
    return model.eval()


@torch.no_grad()
def lightshow(model, logmel, fps, beat_times, offset, notes, env="DefaultEnvironment", busy=None, fade=None,
              seed=0, steps=12, temp=0.9, order="random", min_p=0.0, noise=0.0, dev="cuda"):
    from .data import beat_mel
    if not len(notes):
        return []
    g = torch.Generator(device=dev).manual_seed(seed)
    n_beats = int(np.ceil(notes["beat"].max() - offset)) + 4
    T, hop = WINDOW * TICKS_PER_BEAT, (WINDOW - CONTEXT) * TICKS_PER_BEAT
    n_ticks = n_beats * TICKS_PER_BEAT
    bt = np.asarray(beat_times, float)
    mel = beat_mel(logmel, fps, bt, offset, n=(n_beats + WINDOW) * AUDIO_PER_BEAT, per_beat=AUDIO_PER_BEAT)
    nt = note_ticks(notes["beat"] - offset, notes["color"], n_ticks + T)
    cells = np.zeros((n_ticks + T, CHANNELS), np.int64)
    e = torch.tensor([ENVS.index(env) if env in ENVS else OTHER], device=dev)
    if busy is None:
        per_beat = len(notes) / max(float(np.ptp(notes["beat"])), 1)
        want = np.interp(per_beat, NOTES_PER_BEAT, TYPICAL_BUSY)
        busy = int(np.argmin(np.abs(np.array(MADE_BUSY) - want)))
    fade = len(FADE_EDGES) if fade is None else fade
    kind = [torch.tensor([busy], device=dev), torch.tensor([fade], device=dev)]
    for t0 in range(0, n_ticks, hop):
        b = t0 // TICKS_PER_BEAT
        x = torch.full((1, T, CHANNELS), MASK, dtype=torch.long, device=dev)
        if t0:
            x[0, :CONTEXT * TICKS_PER_BEAT] = torch.from_numpy(cells[t0:t0 + CONTEXT * TICKS_PER_BEAT])
        a = torch.from_numpy(mel[b * AUDIO_PER_BEAT:(b + WINDOW) * AUDIO_PER_BEAT])[None].to(dev)
        n = torch.from_numpy(nt[t0:t0 + T].astype(np.int64))[None].to(dev)
        beat0 = torch.tensor([b % 4], device=dev)
        masked = x == MASK
        total = int(masked.sum())
        for k in range(steps):
            with torch.autocast("cuda", torch.bfloat16, enabled=str(dev).startswith("cuda")):
                logits = model(x, n, a, e, beat0, *kind)
            p = torch.softmax(logits / temp, -1)
            if min_p:
                p = torch.where(p >= min_p * p.amax(-1, keepdim=True), p, torch.zeros_like(p))
            pick = torch.multinomial(p.view(-1, N_VALUES + 1), 1, generator=g).view(p.shape[:-1])
            if order == "sure":
                conf = torch.gather(p, -1, pick[..., None])[..., 0].clamp(min=1e-9).log()
                if noise:
                    u = torch.rand(conf.shape, device=dev, generator=g).clamp(1e-9, 1 - 1e-9)
                    conf = conf + noise * (1 - k / steps) * -torch.log(-torch.log(u))
            else:
                conf = torch.rand(p.shape[:-1], device=dev, generator=g)
            conf = torch.where(masked, conf, torch.full_like(conf, float("inf")))
            left = int(total * math.cos(math.pi / 2 * (k + 1) / steps))
            if left > 0:
                cut = torch.topk(conf.view(-1), conf.numel() - left).values[-1]
                fix = masked & (conf >= cut)
            else:
                fix = masked
            x = torch.where(fix, pick, x)
            masked &= ~fix
        cells[t0:t0 + T] = x[0].cpu().numpy()
    t, c = np.nonzero(cells[:n_ticks])
    v = cells[t, c] - 1
    order = np.lexsort((c, t))
    return [(round(float(t[i] / TICKS_PER_BEAT + offset), 4), *event_type(int(c[i]), int(v[i]))) for i in order]
