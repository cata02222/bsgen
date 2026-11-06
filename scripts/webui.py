import argparse
import contextlib
import functools
import html
import io
import os
import queue
import re
import subprocess
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
import gradio as gr
from PIL import Image

import generate
from bsgen import artwork, beatmap, export, hub, lightmodel, rhythm

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "web"
STYLES = {"Auto": None, "Acc": "acc", "Tech": "tech", "Midspeed": "midspeed", "Speed": "speed"}
ARCS = {"Off": None, "Rare": 0.0, "Normal": 0.025, "Lots": 0.08}
DRY, XTC = 1.0, 0.3
RHYTHM = [f"Auto ({rhythm.AUTO_STARS:g}★ and up)" if rhythm.AUTO_STARS > 0 else "Auto (every difficulty)", "Always", "Never"]
DIFFS = {"Easy": (3.5, "Easy"), "Normal": (4.5, "Normal"), "Hard": (5.5, "Hard"), "Expert": (7.5, "Expert"),
         "Expert+": (10.0, "ExpertPlus")}
MAX_LEVELS = 5
SCENES = {"Auto": "auto"} | dict(zip(["The First (default)", "Origins", "Triangle", "Nice", "Big Mirror", "Dragons", "K/DA", "Monstercat",
                   "Crab Rave", "Panic", "Rocket League", "Green Day", "Green Day Grenade", "Timbaland", "FitBeat",
                   "Linkin Park", "BTS", "Kaleidoscope", "Interscope", "Skrillex", "Billie Eilish", "Spooky",
                   "Lady Gaga"], lightmodel.ENVS))
START_STARS = (10, 7.5, 5.5, 12.5, 4)
SHOWN = {"ExpertPlus": "Expert+"}
HIGH_STAR = 12.0
LEAD_CHECK, LEAD_IN, LEAD_DB = 1.0, 1.0, -45.0
RED, BLUE = "#e8394b", "#2b8fd8"
GITHUB = "https://github.com/cata02222/bsgen"
GITHUB_MARK = ('<svg viewBox="0 0 16 16"><path d="M8 0c4.42 0 8 3.58 8 8a8.013 8.013 0 0 1-5.45 7.59c-.4.08-.55-.17-.55-.38 '
               '0-.27.01-1.13.01-2.2 0-.75-.25-1.23-.54-1.48 1.78-.2 3.65-.88 3.65-3.95 0-.88-.31-1.59-.82-2.15.08-.2.36-1.02-.08-2.12 '
               '0 0-.67-.22-2.2.82-.64-.18-1.32-.27-2-.27-.68 0-1.36.09-2 .27-1.53-1.03-2.2-.82-2.2-.82-.44 1.1-.16 1.92-.08 2.12-.51.56'
               '-.82 1.28-.82 2.15 0 3.06 1.86 3.75 3.64 3.95-.23.2-.44.55-.51 1.07-.46.21-1.61.55-2.33-.66-.15-.24-.6-.83-1.23-.82-.67.01'
               '-.27.38.01.53.34.19.73.9.82 1.13.16.45.68 1.31 2.69.94 0 .67.01 1.3.01 1.49 0 .21-.15.45-.55.38A7.995 7.995 0 0 1 0 8c0-4.42 '
               '3.58-8 8-8Z"/></svg>')
STAR_ASK = ('<div class="star"><span>Enjoying bsgen? A star on GitHub helps.</span>'
            f'<a href="{GITHUB}" target="_blank" rel="noopener" onclick="try{{localStorage.setItem(\'bsgen-starred\',\'1\')}}'
            f'catch(e){{}};setTimeout(()=>document.body.classList.add(\'starred\'),400)">{GITHUB_MARK}Star on GitHub</a></div>')
DISCORD = "https://discord.gg/DCFdQYAQV4"
DISCORD_MARK = ('<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M20.317 4.37a19.79 19.79 0 0 0-4.885-1.515'
                'c-.211.375-.457.88-.626 1.282a18.27 18.27 0 0 0-5.417 0 12.64 12.64 0 0 0-.635-1.282'
                'A19.736 19.736 0 0 0 3.87 4.374C.78 8.953-.058 13.418.361 17.82a19.9 19.9 0 0 0 5.993 3.03'
                'c.483-.66.914-1.36 1.285-2.096a12.87 12.87 0 0 1-2.024-.964c.17-.124.336-.254.496-.388'
                ' 3.903 1.825 8.143 1.825 12 0 .162.134.328.264.496.388-.644.378-1.322.702-2.026.966'
                '.371.735.801 1.437 1.285 2.095a19.84 19.84 0 0 0 5.997-3.03c.49-5.103-.838-9.527-3.546-13.451'
                'ZM8.02 15.123c-1.172 0-2.133-1.076-2.133-2.388s.94-2.389 2.133-2.389 2.154 1.076 2.133 2.389'
                'c0 1.312-.94 2.388-2.133 2.388Zm7.88 0c-1.172 0-2.133-1.076-2.133-2.388s.94-2.389 2.133-2.389'
                ' 2.154 1.076 2.133 2.389c0 1.312-.94 2.388-2.133 2.388Z"/></svg>')
DISCORD_ASK = ('<div class="community discord"><span>Join our community.</span>'
               f'<a href="{DISCORD}" target="_blank" rel="noopener noreferrer">{DISCORD_MARK}Join our Discord</a></div>')
MAP_LINE = re.compile(r"(?:(\w+): )?([\d.]+)★, NJS ([\d.]+): (\d+) notes, ([\d.]+) notes/s")
RATED_LINE = re.compile(r"(?:(\w+): )?(?:estimated from the notes: |kept the one made at [\d.]+★, rated )([\d.]+)★")
REPLAY_LINE = re.compile(r"(\d+) of (\d+) notes cut, (\d+) bad cuts?, (\d+) missed, score (\d+)")

CSS = f"""
.gradio-container {{ width: 100% !important; max-width: 580px !important; margin: auto; }}
#title {{ text-align: center; padding: 28px 0 4px; }}
#title h1 {{ font-size: 44px; font-weight: 800; letter-spacing: -1px; margin: 0; }}
#title h1 .r {{ color: {RED}; }}
#title h1 .b {{ color: {BLUE}; }}
#title p {{ opacity: .6; margin: 6px 0 0; }}
#no-gpu {{ color: {RED}; text-align: center; padding: 8px 0; }}
#go, #play {{ background: linear-gradient(90deg, {RED}, {BLUE}); border: none; color: white; font-weight: 700; }}
footer {{ display: none !important; }}
#map-zip {{ border-style: solid !important; }}
.status .bar {{ height: 8px; border-radius: 4px; background: rgba(127,127,127,.2); overflow: hidden; }}
.status .fill {{ height: 100%; background: linear-gradient(90deg, {RED}, {BLUE}); transition: width .4s; }}
.status .busy .fill {{ width: 35% !important; animation: slide 1.2s ease-in-out infinite alternate; }}
@keyframes slide {{ from {{ margin-left: 0; }} to {{ margin-left: 65%; }} }}
.status .head {{ display: flex; justify-content: space-between; font-weight: 600; margin: 4px 0 8px; }}
.status .row {{ display: grid; grid-template-columns: minmax(120px, max-content) 1fr 48px; gap: 10px;
               align-items: center; font-size: 14px; margin: 6px 0; }}
.status .row span:first-child {{ white-space: nowrap; }}
.status .row span:last-child {{ text-align: right; opacity: .7; font-variant-numeric: tabular-nums; }}
.status .sum {{ font-size: 14px; opacity: .8; margin: 2px 0; }}
.status .err {{ color: {RED}; }}
.status .meta {{ font-size: 14px; opacity: .65; margin: 8px 0 0; }}
.status .lvs {{ display: grid; grid-template-columns: max-content 1fr; gap: 3px 14px; margin: 10px 0 0;
               font-size: 14px; font-variant-numeric: tabular-nums; }}
.status .lvs .dim {{ opacity: .55; }}
.status .star, .status .community {{ display: flex; align-items: center; justify-content: space-between; gap: 12px;
                 margin: 14px 0 0; font-size: 14px; }}
.status .star > span, .status .community > span {{ opacity: .65; }}
.status .star a, .status .community a {{ display: inline-flex; align-items: center; gap: 7px; padding: 4px 11px; border-radius: 6px;
    border: 1px solid rgba(127,127,127,.35); color: inherit !important; text-decoration: none !important;
    font-size: 13px; white-space: nowrap; }}
.status .star a:hover {{ border-color: rgba(127,127,127,.75); }}
.status .star svg, .status .community svg {{ width: 16px; height: 16px; fill: currentColor; }}
.status .discord a {{ background: #5865f2; border-color: #5865f2; color: white !important;
    transition: background .15s, border-color .15s; }}
.status .discord a:hover {{ background: #4752c4; border-color: #4752c4; }}
.status .community a:focus-visible {{ outline: 2px solid #5865f2; outline-offset: 3px; }}
body.starred .status .star {{ display: none; }}
#youtube-row {{ flex-wrap: nowrap; align-items: center; gap: 6px; padding: 5px 5px 5px 12px; min-height: 0;
               background: var(--block-background-fill); border: 1px solid var(--block-border-color);
               border-radius: var(--radius-md); }}
#youtube-row .form, #youtube-link, #youtube-link * {{ background: transparent !important; border: none !important; box-shadow: none !important; }}
#youtube-link, #youtube-link .block, #youtube-link label {{ background: transparent !important; border: none !important;
               box-shadow: none !important; padding: 0 !important; min-width: 0 !important; }}
#youtube-link textarea, #youtube-link input {{ background: transparent !important; border: none !important;
               box-shadow: none !important; padding: 6px 0 !important; height: auto !important; }}
#youtube-go {{ flex: 0 0 auto !important; min-width: 0 !important; width: auto !important; height: 32px; padding: 0 14px;
               font-size: var(--text-md); font-weight: 500; line-height: 1; }}
#youtube-go:hover {{ border-color: {BLUE}; color: {BLUE}; }}
#song-details {{ flex-wrap: nowrap; align-items: center; }}
#cover-picker {{ flex: 0 0 132px; min-width: 132px !important; max-width: 132px; }}
#cover-picker .upload-container {{ min-height: 0; }}
#cover-picker .prompt, #cover-picker .prompt p, #cover-picker .prompt span {{
    white-space: normal; text-overflow: clip; overflow: visible; font-size: 13px; text-align: center;
}}
"""
SCROLL = """
(...args) => {
  document.getElementById('status')?.scrollIntoView({behavior: 'smooth', block: 'center'});
  return args;
}
"""
DARK = """
() => {
  // at the bottom of the page, stay there while the status and the map files are added below
  let stuck = false;
  const atBottom = () => window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 80;
  window.addEventListener('scroll', () => { stuck = atBottom(); }, {passive: true});
  new ResizeObserver(() => { if (stuck) window.scrollTo(0, document.documentElement.scrollHeight); })
    .observe(document.body);
  try { if (localStorage.getItem('bsgen-starred')) document.body.classList.add('starred'); } catch (e) {}
  // audio files the browser gives no type (.egg: a game's .ogg) go to the song box as audio, so it
  // takes them from the file picker and from a drop
  const typed = (files) => {
    if (![...files].some((f) => !f.type)) return null;
    const dt = new DataTransfer();
    for (const f of files) {
      const ext = (f.name.match(/\\.(\\w+)$/) || [, ''])[1].toLowerCase();
      dt.items.add(f.type ? f : new File([f], ext === 'egg' ? f.name.replace(/\\.egg$/i, '.ogg') : f.name,
                                         {type: 'audio/' + (ext === 'egg' ? 'ogg' : ext)}));
    }
    return dt;
  };
  const inSong = (el) => el instanceof Element && el.closest('#song');
  document.addEventListener('drop', (e) => {
    if (!inSong(e.target) || !e.dataTransfer || e.bsgen) return;
    const dt = typed(e.dataTransfer.files);
    if (!dt) return;
    e.stopImmediatePropagation(); e.preventDefault();
    const again = new DragEvent('drop', {bubbles: true, cancelable: true, dataTransfer: dt});
    again.bsgen = true;
    e.target.dispatchEvent(again);
  }, true);
  document.addEventListener('change', (e) => {
    if (inSong(e.target) && e.target.files) { const dt = typed(e.target.files); if (dt) e.target.files = dt.files; }
  }, true);
  document.addEventListener('click', (e) => {
    if (inSong(e.target)) document.querySelectorAll('#song input[type=file]').forEach((i) => {
      if (!i.accept.includes('.egg')) i.accept += (i.accept ? ',' : '') + '.egg';
    });
  }, true);
  const url = new URL(window.location);
  if (url.searchParams.get('__theme') !== 'dark') {
    url.searchParams.set('__theme', 'dark');
    window.location.href = url.href;
  }
}
"""

state = {"model": None}
lock = threading.Lock()
stopping = threading.Event()


class Stopped(Exception):
    pass


def lead_in(song, folder):
    import librosa
    import numpy as np
    head, _ = librosa.load(song, sr=22050, mono=True, duration=LEAD_CHECK)
    if not len(head) or 20 * np.log10(librosa.feature.rms(y=head).max() + 1e-10) < LEAD_DB:
        return None
    title, artist, _, picture = artwork.song_info(song)
    out = folder / "lead_in" / f"{Path(song).stem}.ogg"
    out.parent.mkdir()
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(song), "-vn", "-af", f"adelay={LEAD_IN * 1000:g}:all=1",
                    "-c:a", "libvorbis", "-q:a", "6", "-map_metadata", "0", str(out)], check=True)
    cover = None
    if picture:
        cover = out.parent / "cover"
        cover.write_bytes(picture)
    return out, title, artist, cover


def number(text, name):
    text = (text or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        raise gr.Error(f"{name} should be a number, like 174.")


class Status:

    NAMES = {"ExpertPlus": "Expert+"}
    REPLAY_RATE = 0.06

    def __init__(self, names=None, levels=1, replay=False, doing="Making the map", stars=None):
        self.t0 = time.time()
        self.stars = stars or {}
        self.doing = doing
        self.phase, self.beats, self.found, self.scene = "Starting", {}, "", ""
        self.notes = []
        self.rated, self.maps, self.replays = {}, {}, {}
        for n in names or []:
            self.beats[n] = (0, 1)
        self.started = {}
        self.levels, self.replay, self.song = levels, replay, None

    def update(self, msg):
        if not msg:
            return
        if msg[0] == "phase":
            self.phase = msg[1]
        elif msg[0] == "beat":
            _, name, done, total = msg
            self.beats[name] = (done, total)
            self.phase = "Playing the map" if name.endswith(" replay") else self.doing
            self.started.setdefault(name, (time.time(), done / total))
        elif msg[0] == "log":
            line = msg[1]
            if line.startswith("timing:"):
                self.found = line.split(":", 1)[1].split(",")[0].strip()
            elif line.startswith("scene:"):
                env = line.split(":", 1)[1].strip()
                self.scene = next((name for name, e in SCENES.items() if e == env), env).replace(" (default)", "")
            elif line == "making the lights":
                self.phase = "Making the lights"
            elif line.startswith("song:"):
                self.phase = "Making the map"
                self.song = float(line.split()[1])
            elif m := re.match(r"(\w+): remaking, rated ([\d.]+)★ of ([\d.]+) asked", line):
                how = "hard" if float(m[2]) > float(m[3]) else "easy"
                self.notes.append(f"{self.NAMES.get(m[1], m[1])} came out too {how} ({float(m[2]):g}★ for "
                                  f"{float(m[3]):g}★). Making it again.")
                self.phase = self.doing = "Making it again"
            elif m := RATED_LINE.match(line):
                self.rated[m[1] or self.only()] = float(m[2])
            elif m := re.match(r"(\w+) replay: (.*)", line):
                self.replays[m[1]] = m[2]
            elif m := MAP_LINE.match(line):
                self.maps[m[1] or self.only()] = m

    def only(self):
        names = {n.split(" ")[0] for n in self.beats} or set(self.stars)
        return names.pop() if len(names) == 1 else "map"

    def label(self, name):
        base, _, replay = name.partition(" ")
        slot = self.NAMES.get(base, base)
        if base in self.stars:
            slot = f"{slot} {self.stars[base]:g}★"
        return f"{slot} {replay}".strip()

    def fraction(self):
        parts = self.levels * (2 if self.replay else 1)
        return sum(d / max(t, 1e-9) for d, t in self.beats.values()) / max(parts, len(self.beats), 1)

    def left(self):
        now, worst = time.time(), 0.0
        rate = self.REPLAY_RATE * (1 + 0.4 * (self.levels - 1))
        for name, (d, total) in self.beats.items():
            if name.endswith(" replay"):
                continue
            ahead = []
            for part in (name, f"{name} replay") if self.replay else (name,):
                if part not in self.beats:
                    if self.song is None:
                        return None
                    ahead.append(self.song * rate)
                    continue
                if part not in self.started:
                    return None
                t, f0 = self.started[part]
                d, total = self.beats[part]
                f = d / total
                if f >= 1:
                    continue
                if f <= f0:
                    return None
                ahead.append((now - t) * (1 - f) / (f - f0))
            worst = max(worst, sum(ahead))
        return worst

    def html(self, done=False, error=None):
        took = time.time() - self.t0
        f = 1.0 if done else self.fraction()
        if error:
            head = f'<span class="err">Stopped: {html.escape(error)}</span>'
        elif done:
            head = f"<span>Done in {took:.0f} s</span>"
        else:
            left = None if self.phase == "Making the lights" else self.left()
            left = f" · about {left:.0f} s left" if left is not None else ""
            head = f"<span>{self.phase}…</span><span>{took:.0f} s{left}</span>"
        busy = not done and not error and not any(d for d, _ in self.beats.values())
        out = [f'<div class="head">{head}</div>',
               f'<div class="bar{" busy" if busy else ""}"><div class="fill" style="width:{f:.0%}"></div></div>']
        if done and not error:
            return "".join(out + self.result())
        if self.found:
            out.append(f'<div class="sum">Song: {html.escape(self.found)}</div>')
        if self.scene:
            out.append(f'<div class="sum">Scene: {html.escape(self.scene)}</div>')
        if len(self.beats) > 1 or (self.beats and not done):
            for name, (d, t) in self.beats.items():
                g = 1.0 if done else d / max(t, 1)
                out.append(f'<div class="row"><span>{html.escape(self.label(name))}</span>'
                           f'<div class="bar"><div class="fill" style="width:{g:.0%}"></div></div><span>{g:.0%}</span></div>')
        return "".join(out)

    def result(self):
        out, meta = [], []
        if m := re.match(r"([\d.]+) BPM", self.found):
            meta.append(f"{round(float(m[1]), 1):g} BPM")
        elif self.found:
            meta.append(self.found)
        if self.scene:
            meta.append(self.scene)
        if meta:
            out.append(f'<div class="meta">{html.escape(" · ".join(meta))}</div>')
        lines = []
        for base in dict.fromkeys([*self.maps, *self.replays]):
            m, r = self.maps.get(base), REPLAY_LINE.match(self.replays.get(base, ""))
            parts = []
            stars = self.stars.get(base, float(m[2]) if m else None)
            if stars is not None:
                parts.append(f"{stars:g}★" + (f' <span class="dim">rated {self.rated[base]:.1f}</span>' if base in self.rated else ""))
            if m:
                notes = int(r[2]) if r else int(m[4])
                parts.append(f"{notes:,} notes · {float(m[5]) * notes / int(m[4]):.1f}/s · NJS {float(m[3]):g}")
            lines.append(f"<span>{html.escape(self.NAMES.get(base, base))}</span><span>{' · '.join(parts)}</span>")
            if r:
                lines.append(f'<span></span><span class="dim">replay {int(r[1]):,}/{int(r[2]):,} cut · score {int(r[5]):,}</span>')
            elif base in self.replays:
                lines.append(f'<span></span><span class="dim">replay {html.escape(self.replays[base])}</span>')
        out.append(f'<div class="lvs">{"".join(lines)}</div>')
        return out + [STAR_ASK, DISCORD_ASK]


def star_labels(n, *values):
    names = export.difficulty_names(list(values[:n]))
    out = []
    for i, x in enumerate(values):
        if i >= n:
            out.append(gr.update())
            continue
        label = SHOWN.get(names[i], names[i])
        out.append(gr.update(label=f"{label} · high star" if x >= HIGH_STAR and names[i] == "ExpertPlus" else label))
    return out


def diff_choices(by_name, version):
    if version.split()[0] in hub.FIXED_STARS:
        return [(f"{d} · {x:g}★", d) for d, (x, _) in DIFFS.items()]
    if by_name:
        return [(f"{d} · {by_name[name]:g}★", d) for d, (_, name) in DIFFS.items()]
    return list(DIFFS)


def prefetch(device):
    from bsgen import songstars

    def models():
        with lock:
            if state["model"] is None:
                state["model"] = generate.load_map_model(hub.NEWEST, None, device)
                state["version"] = hub.NEWEST
            if state.get("lighter") is None:
                state["lighter"] = lightmodel.load(hub.light_model(), device)

    jobs = [models, hub.rhythm_model, hub.timing_model, hub.player_model, songstars.load]
    jobs += [functools.partial(generate.legacy_files, v) for v in hub.LEGACY]
    for job in jobs:
        try:
            job()
        except Exception as e:
            print(f"prefetch: {e}", flush=True)
    print("models ready", flush=True)


NO_GPU = ("No GPU here, so maps take many minutes and land less close to the stars asked. "
          "On Colab: Runtime → Change runtime type → T4 GPU, then run both cells again.")


def show_link(local, public):
    print(f"\nbsgen is running, open {public or local}", flush=True)
    if public:
        print(f"(works on any device while this runs; on this computer also {local})", flush=True)
    print(flush=True)


def player(device):
    if state.get("player") is None:
        from bsgen import botplay
        state["player"] = botplay.load_player(hub.player_model(), device)
    return state["player"]


def follow(q, view):
    yield gr.update(value=None, visible=False), view.html()
    shown = 0.0
    while True:
        try:
            msg = q.get(timeout=0.5)
        except queue.Empty:
            msg = ()
        if msg is None:
            return
        view.update(msg)
        while view.notes:
            gr.Info(view.notes.pop(0), duration=5)
        if time.time() - shown > 0.25:
            shown = time.time()
            yield gr.update(value=None, visible=False), view.html()


def map_levels(path):
    try:
        with beatmap.open_map(path) as z:
            info = beatmap.read_info(z)
    except Exception:
        raise gr.Error("That doesn't look like a map zip (no readable Info.dat).")
    rank = [name for _, name in DIFFS.values()]
    names = [d["difficulty"] for d in info.difficulties if d["characteristic"] == "Standard"]
    names.sort(key=lambda n: rank.index(n) if n in rank else len(rank))
    return [(Status.NAMES.get(n, n), n) for n in names]


def build(ckpt, device):
    def pick_levels(path):
        if not path:
            return gr.update(choices=[], value=[], visible=False)
        levels = map_levels(path)
        if not levels:
            raise gr.Error("This map has no Standard difficulty to play.")
        return gr.update(choices=levels, value=[levels[-1][1]], visible=True)

    def play(path, names, height, seed):
        if not path:
            raise gr.Error("Add a map zip first.")
        if not names:
            raise gr.Error("Pick at least one difficulty.")
        seed = number(seed, "Seed")
        folder = OUT / uuid.uuid4().hex[:10]
        folder.mkdir(parents=True)
        stem = Path(path).stem
        q, result = queue.Queue(), {}

        def one(name):
            import torch
            from bsgen import botplay
            stream = torch.cuda.Stream() if str(device).startswith("cuda") else None
            with torch.cuda.stream(stream) if stream else contextlib.nullcontext():
                motion = botplay.play_motion(state["player"], path, name, seed=int(seed or 0), height=height, dev=device,
                                             progress=lambda done, total: q.put(("beat", name, done, total)))
            out = folder / f"{stem} {name}.bsor"
            q.put(("log", f"{name} replay: {botplay.describe(botplay.save_replay(path, name, motion, out))}"))
            return out

        def work():
            try:
                with lock:
                    q.put(("phase", "Loading the player"))
                    player(device)
                    q.put(("phase", "Playing the map"))
                    with ThreadPoolExecutor(len(names)) as ex:
                        result["files"] = [str(f) for f in ex.map(one, names)]
            except Exception as e:
                result["error"] = str(e)
            q.put(None)

        threading.Thread(target=work, daemon=True).start()
        view = Status(names, len(names), doing="Playing the map")
        yield from follow(q, view)
        if "error" in result:
            yield gr.update(value=None, visible=False), view.html(error=result["error"])
            raise gr.Error(result["error"])
        yield gr.update(value=result["files"], visible=True), view.html(done=True)

    def from_youtube(link):
        from bsgen import youtube
        if not (link or "").strip():
            raise gr.Error("Paste a YouTube link first.")
        gr.Info("Getting the audio from YouTube…")
        try:
            path, note = youtube.download(link, OUT / "youtube" / uuid.uuid4().hex[:10])
            if note:
                gr.Warning(note)
            return str(path)
        except youtube.YouTubeError as e:
            raise gr.Error(str(e)) from None

    def make(song, mode, diff, count, *rest):
        stars, (style, arcs, extras, replay, njs, bpm, first_ms, seed, mapper, fmt, title, artist, cover, version, scene, rhythm,
                song_stars) = rest[:MAX_LEVELS], rest[MAX_LEVELS:]
        if not song:
            raise gr.Error("Add a song first.")
        if mode == "Stars":
            levels = [(x, None) for x in sorted(set(stars[:count]))]
        else:
            legacy = version.split()[0] in hub.FIXED_STARS and not ckpt
            levels = [(x if legacy else (song_stars or {}).get(name, x), name) for d, (x, name) in DIFFS.items() if d in diff]
            if not levels:
                raise gr.Error("Pick at least one difficulty.")
        folder = OUT / uuid.uuid4().hex[:10]
        folder.mkdir(parents=True)
        padded = lead_in(song, folder)
        if padded:
            song, tagged_title, tagged_artist, tagged_cover = padded
            title, artist, cover = title or tagged_title, artist or tagged_artist, cover or tagged_cover
        argv = ["--audio", str(song), "--stars", *(f"{x:g}" for x, _ in levels),
                "--format", fmt, "--device", device, "--out", str(folder)]
        if mode != "Stars":
            argv += ["--difficulty", *(name for _, name in levels)]
        version = version.split()[0]
        if ckpt:
            argv += ["--ckpt", ckpt]
        else:
            argv += ["--model", version]
        argv += ["--environment", SCENES[scene]]
        for flag, value in (("--title", title), ("--artist", artist), ("--cover", cover)):
            if value:
                argv += [flag, str(value)]
        if (seed := number(seed, "Seed")) is not None:
            argv += ["--seed", str(int(seed))]
        if (njs := number(njs, "Note speed")) is not None:
            argv += ["--njs", f"{njs:g}"]
        if STYLES[style]:
            argv += ["--style", STYLES[style]]
        argv += ["--no-arcs"] if ARCS[arcs] is None else ["--arc-rate", f"{ARCS[arcs]:g}"]
        for part in ("Walls", "Bombs", "Lights"):
            if part not in extras:
                argv.append(f"--no-{part.lower()}")
        argv += ["--dry", f"{DRY:g}", "--xtc", f"{XTC:g}"]
        if version == hub.NEWEST and not ckpt and rhythm != RHYTHM[0]:
            argv.append("--rhythm" if rhythm == RHYTHM[1] else "--no-rhythm")
        bpm, first_ms = number(bpm, "Song BPM"), number(first_ms, "First beat")
        if bpm:
            argv += ["--bpm", f"{bpm:g}"]
            if first_ms is not None:
                argv += ["--offset-ms", f"{first_ms + (LEAD_IN * 1000 if padded else 0):g}"]
        if mapper and mapper != "Any":
            argv += ["--mapper", mapper]
        if replay:
            argv.append("--replay")
        args = generate.parser().parse_args(argv)

        q, result = queue.Queue(), {}
        if padded:
            q.put(("log", f"the song starts right away: {LEAD_IN:g} s of silence put in front"))

        def check():
            if stopping.is_set():
                raise Stopped()

        def progress(name, done, total):
            check()
            q.put(("beat", name, done, total))
        progress.check = check

        def log(m):
            check()
            q.put(("log", m))

        def work():
            try:
                with lock:
                    if stopping.is_set():
                        raise Stopped()
                    if state["model"] is None or state.get("version") != version:
                        q.put(("phase", "Loading the model"))
                        state["model"] = None
                        state["model"] = generate.load_map_model(version, ckpt, device)
                        state["version"] = version
                    if replay:
                        player(device)
                    if "Lights" in extras and state.get("lighter") is None and version not in hub.LEGACY:
                        state["lighter"] = lightmodel.load(hub.light_model(), device)
                    q.put(("phase", "Reading the song"))
                    result["zip"] = generate.run(args, state["model"], log=log,
                                                 progress=progress,
                                                 player=state.get("player"), lighter=state.get("lighter"))
            except Stopped:
                result["error"] = "you pressed Stop"
            except Exception as e:
                result["error"] = str(e)
            q.put(None)

        stopping.clear()
        threading.Thread(target=work, daemon=True).start()
        slots = export.difficulty_names([x for x, _ in levels]) if mode == "Stars" else [name for _, name in levels]
        view = Status([name for _, name in levels] if mode != "Stars" else None, len(levels), bool(replay),
                      stars=dict(zip(slots, (x for x, _ in levels))))
        for files, shown in follow(q, view):
            yield files, shown, gr.update(visible=True)
        if "error" in result:
            yield gr.update(value=None, visible=False), view.html(error=result["error"]), gr.update(visible=False)
            if not stopping.is_set():
                raise gr.Error(result["error"])
            return
        z = Path(result["zip"])
        files = [str(z)] + sorted(str(p) for p in z.parent.iterdir()
                                  if p.suffix == ".bsor" and p.name.startswith(z.stem + " "))
        yield gr.update(value=files, visible=True), view.html(done=True), gr.update(visible=False)

    mappers = ["Any"]
    try:
        from bsgen import hub, sample
        mappers += sorted(sample.read_checkpoint(ckpt or hub.map_model()).get("mappers") or [])
    except Exception:
        pass

    with gr.Blocks(title="bsgen", analytics_enabled=False) as demo:
        gr.HTML('<div id="title"><h1><span class="r">bs</span><span class="b">gen</span></h1>'
                "<p>Beat Saber maps from any song</p></div>")
        if str(device) == "cpu":
            gr.HTML(f'<div id="no-gpu">{NO_GPU}</div>')
        with gr.Tabs():
            with gr.Tab("Make a map"):
                song = gr.Audio(label="Song", type="filepath", format=None, sources=["upload"], elem_id="song")
                with gr.Row(elem_id="youtube-row"):
                    youtube_link = gr.Textbox(show_label=False, scale=1, min_width=0, max_lines=1,
                                              placeholder="…or paste a YouTube link", elem_id="youtube-link")
                    youtube_go = gr.Button("Get the song", scale=0, min_width=0, elem_id="youtube-go")
                with gr.Row(visible=False, elem_id="song-details") as names:
                    cover = gr.Image(label="Cover", type="filepath", sources=["upload"],
                                     height=132, width=132, min_width=132, scale=0, show_label=False,
                                     buttons=[], placeholder="No cover found · click to pick one",
                                     elem_id="cover-picker")
                    with gr.Column(scale=1, min_width=140):
                        title = gr.Textbox(label="Song title", show_label=False, placeholder="Song title")
                        artist = gr.Textbox(label="Artist", show_label=False, placeholder="Artist")
                auto_cover = gr.State(None)
                mode = gr.Radio(["Difficulty", "Stars"], value="Difficulty", show_label=False, elem_id="mode")
                diff = gr.CheckboxGroup(list(DIFFS), value=["Expert+"], show_label=False)
                song_stars = gr.State(None)
                with gr.Group(visible=False) as star_box:
                    stars = [gr.Slider(1, 15, value=v, step=0.5, label=star_labels(1, v)[0]["label"] if i == 0 else "",
                                       visible=i == 0) for i, v in enumerate(START_STARS)]
                    with gr.Row():
                        more = gr.Button("+ another difficulty", size="sm")
                        fewer = gr.Button("− remove", size="sm", visible=False)
                count = gr.State(1)
                with gr.Accordion("Options", open=False):
                    style = gr.Radio(list(STYLES), value="Auto", label="Style")
                    arcs = gr.Radio(list(ARCS), value="Off", label="Arcs")
                    extras = gr.CheckboxGroup(["Walls", "Bombs", "Lights"], value=["Walls", "Bombs", "Lights"], show_label=False)
                    replay = gr.Checkbox(False, label="Also make replays")
                    scene = gr.Dropdown(list(SCENES), value="Auto", label="Scene")
                with gr.Accordion("Expert", open=False):
                    with gr.Row():
                        bpm = gr.Textbox(label="Song BPM", placeholder="detected")
                        first_ms = gr.Textbox(label="First beat (ms)", placeholder="detected")
                    with gr.Row():
                        njs = gr.Textbox(label="Note speed (NJS)", placeholder="auto")
                        seed = gr.Textbox(label="Seed", placeholder="random")
                    with gr.Row():
                        fmt = gr.Radio(["v2", "v3"], value="v2", label="Format")
                        mapper = gr.Dropdown(mappers, value="Any", label="Map like", visible=len(mappers) > 1)
                    version = gr.Dropdown([f"{v} · {hub.MAP_NOTES[v]}" for v in hub.VERSIONS], value=f"{hub.NEWEST} · "
                                          f"{hub.MAP_NOTES[hub.NEWEST]}", label="Map model", visible=not ckpt)
                    rhythm = gr.Dropdown(RHYTHM, value=RHYTHM[0], label="Rhythm model", visible=not ckpt)
                go = gr.Button("Generate map", elem_id="go", size="lg")
                stop = gr.Button("Stop", size="sm", visible=False)
                status = gr.HTML('<div class="sum">Add a song and press Generate.</div>', elem_id="status", elem_classes=["status"], padding=True)
                out = gr.File(label="Map (.zip) and replays (.bsor)", file_count="multiple", height=140, visible=False)
            with gr.Tab("Replay a map"):
                map_zip = gr.File(label="Map (.zip)", file_types=[".zip"], type="filepath", height=240, elem_id="map-zip")
                play_diffs = gr.CheckboxGroup([], label="Difficulties", visible=False)
                with gr.Row():
                    height = gr.Slider(1.4, 2.1, value=1.7, step=0.05, label="Player height (m)")
                    play_seed = gr.Textbox(label="Seed", placeholder="0")
                play_go = gr.Button("Make replay", elem_id="play", size="lg")
                play_status = gr.HTML('<div class="sum">Add a map and press Make replay.</div>',
                                      elem_classes=["status"], padding=True)
                play_out = gr.File(label="Replays (.bsor)", file_count="multiple", height=140, visible=False)
        def resize(n):
            n = max(1, min(MAX_LEVELS, n))
            return [n, *(gr.update(visible=i < n) for i in range(MAX_LEVELS)),
                    gr.update(visible=n < MAX_LEVELS), gr.update(visible=n > 1)]

        def find_cover(t, a, duration=None, picture=None):
            from bsgen import artwork
            found = None
            if not picture and t:
                picture, found = artwork.find(t, a, duration)
            if not picture:
                return None, a or found or ""
            path = OUT / "covers" / f"{uuid.uuid4().hex[:10]}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with Image.open(io.BytesIO(picture)) as im:
                    im.convert("RGB").save(path)
            except Exception:
                return None, a or found or ""
            return str(path), a or found or ""

        def song_stars_of(path):
            import librosa
            from bsgen import sample, songstars
            y, sr = librosa.load(path, sr=None, mono=True)
            return songstars.stars(*sample.song_mel(y, sr))

        def song_names(path, version):
            if not path:
                return gr.update(visible=False), "", "", None, None, gr.update(choices=diff_choices(None, version)), None
            from bsgen import artwork
            with ThreadPoolExecutor(1) as ex:
                found = ex.submit(song_stars_of, path)
                t, a, duration, picture = artwork.song_info(path)
                c, a = find_cover(t, a, duration, picture)
                try:
                    by_name = found.result()
                except Exception:
                    by_name = None
            return gr.update(visible=True), t, a, c, c, gr.update(choices=diff_choices(by_name, version)), by_name

        @functools.lru_cache(64)
        def found_cover(t, a):
            return find_cover(t, a)

        def renamed(t, a, c, auto):
            if c and c != auto:
                return c, auto, a
            c, a = found_cover(t.strip(), a.strip())
            return c, c, a

        for ev in (youtube_go.click, youtube_link.submit):
            ev(from_youtube, youtube_link, song, concurrency_limit=1)
        song.change(song_names, [song, version], [names, title, artist, cover, auto_cover, diff, song_stars])
        for box in (title, artist):
            box.blur(renamed, [title, artist, cover, auto_cover], [cover, auto_cover, artist])
        mode.change(lambda m: (gr.update(visible=m == "Difficulty"), gr.update(visible=m == "Stars")), mode, [diff, star_box])
        more.click(lambda n: resize(n + 1), count, [count, *stars, more, fewer])
        fewer.click(lambda n: resize(n - 1), count, [count, *stars, more, fewer])
        for ev in [count.change] + [x.change for x in stars]:
            ev(star_labels, [count, *stars], stars, concurrency_limit=None, show_progress="hidden")
        go.click(make, [song, mode, diff, count, *stars, style, arcs, extras, replay, njs, bpm, first_ms, seed, mapper, fmt, title, artist, cover, version, scene, rhythm, song_stars],
                 [out, status, stop], js=SCROLL)
        stop.click(stopping.set)
        version.change(lambda v: gr.update(visible=v.split()[0] == hub.NEWEST), version, rhythm)
        version.change(lambda v, b: gr.update(choices=diff_choices(b, v)), [version, song_stars], diff)
        map_zip.change(pick_levels, map_zip, play_diffs)
        play_go.click(play, [map_zip, play_diffs, height, play_seed], [play_out, play_status])
    return demo


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", help="map model checkpoint (default: the released model, downloaded once)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--share", action="store_true", help="also make a public link (on Colab always)")
    args = ap.parse_args()
    theme = gr.themes.Base(primary_hue=gr.themes.colors.blue, secondary_hue=gr.themes.colors.red,
                           neutral_hue=gr.themes.colors.zinc, radius_size=gr.themes.sizes.radius_md,
                           font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"])
    if str(args.device) == "cpu":
        print(NO_GPU, flush=True)
    demo = build(args.ckpt, args.device)
    share = args.share or "COLAB_RELEASE_TAG" in os.environ
    said = io.StringIO()
    with contextlib.redirect_stdout(said):
        _, local, public = demo.queue(default_concurrency_limit=1).launch(
            server_name=args.host, server_port=args.port, share=share, theme=theme, css=CSS, js=DARK,
            allowed_paths=[str(OUT)], quiet=True, prevent_thread_lock=True)
    if share and not public:
        print(said.getvalue(), end="", flush=True)
    show_link(local, public)
    if not args.ckpt:
        print("loading the models in the background...", flush=True)
        threading.Thread(target=prefetch, args=(args.device,), daemon=True).start()
    demo.block_thread()


if __name__ == "__main__":
    main()
