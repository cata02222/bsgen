import argparse
import contextlib
import io
import json
import subprocess
import sys
import tempfile
from itertools import product
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bsgen import artwork, beatmap, botplay, export, hub, lightmodel, lights, lightseq, pattern, rating, rhythm, sample, songstars, timing, tokens
from bsgen.tokens import SWING

ROOT = Path(__file__).resolve().parents[1]
GB_PER_DIFFICULTY = 1.0
STAR_TRIES = 3
STAR_ROUNDS = 2
STAR_SPREAD = 2.5
HITS_PER_NOTE = 0.85
STAR_CLOSE = 0.75
STAR_MAX = 16.0
STAR_CLIMB = 2.5
RHYTHM_TRIES = 16
RHYTHM_BATCH = 3
STYLE_AIM = True


def first_asks(star, k, around=False):
    if around and k > 1:
        return [max(star + PLAN_SPREAD * (2 * j / (k - 1) - 1), 0.5) for j in range(k)]
    return [star] if k == 1 else [min(star + STAR_SPREAD * j / (k - 1), STAR_MAX) for j in range(k)]


def next_asks(star, made, k, climb=STAR_CLIMB):
    if k == 1 or any(lv["est"] is None for lv in made) or min(abs(lv["est"] - star) for lv in made) <= STAR_CLOSE:
        return []
    a, e = np.array([lv["ask"] for lv in made]), np.array([lv["est"] for lv in made])
    slope, icept = np.polyfit(a, e, 1) if np.ptp(a) > 0 else (0.0, 0.0)
    best = made[int(np.argmin(np.abs(e - star)))]
    aim = (star - icept) / slope if slope > 0.2 else best["ask"] + star - best["est"]
    top = min(star + climb, STAR_MAX)
    aim = float(np.clip(aim, max(star - STAR_SPREAD, 0.5), top))
    asks = [float(np.clip(aim + d, 0.5, top)) for d in np.linspace(-0.4, 0.4, k)]
    asks = sorted({round(x, 2) for x in asks if np.min(np.abs(a - x)) > 0.2})
    return asks


STAR_ORDER_GAP = 0.5
PICK_OFF = 10.0
PLAN_LIFT = 2.5
PLAN_SPREAD = 0.8
STYLE_PLAN_LIFT = {"tech": 3.2, "speed": -2.4, "midspeed": 0.0, "acc": 0.0}
RHYTHM_AT_WANTED = False


def pick_levels(stars, made):
    near = lambda s, lv: abs(lv["est"] - s) + PICK_OFF * (1 - lv.get("on", 1.0)) if lv["est"] else 0.0
    nearest = [min(m, key=lambda lv: near(s, lv)) for s, m in zip(stars, made)]
    if any(lv["est"] is None for m in made for lv in m):
        return nearest, True
    order = sorted(range(len(stars)), key=lambda i: stars[i])

    def rising(pick):
        return all(stars[b] == stars[a] or pick[b]["est"] - pick[a]["est"] >= min(STAR_ORDER_GAP, (stars[b] - stars[a]) / 2)
                   for a, b in zip(order, order[1:]))
    fits = [p for p in product(*made) if rising(p)]
    if not fits:
        return nearest, False
    return list(min(fits, key=lambda p: sum(near(s, lv) for s, lv in zip(stars, p)))), True


def ogg_vorbis(path):
    data = Path(path).read_bytes()
    if data[:4] == b"OggS" and b"vorbis" in data[:64]:
        return data
    return subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-c:a", "libvorbis", "-q:a", "6", "-f", "ogg", "-"],
                          capture_output=True, check=True).stdout


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", help="map model checkpoint (default: the released model of --model, downloaded once)")
    ap.add_argument("--model", choices=hub.VERSIONS, default=hub.NEWEST, help="which release's map model")
    ap.add_argument("--map", help="take song + timing from this map zip")
    ap.add_argument("--diff", default="ExpertPlus", help="difficulty to read from --map")
    ap.add_argument("--difficulty", nargs="+", choices=list(export.DIFF_RANK),
                    help="one per --stars value (default: from the stars); without --stars each one's "
                         "stars come from the song (bsgen/songstars.py)")
    ap.add_argument("--audio")
    ap.add_argument("--youtube", metavar="URL", help="a YouTube link to take the song from (instead of --audio)")
    ap.add_argument("--title", help="song title for artwork lookup (default: audio tags or filename)")
    ap.add_argument("--artist", help="song artist for artwork lookup (default: audio tags or filename)")
    ap.add_argument("--cover", help="cover image to use instead of an automatic lookup")
    ap.add_argument("--bpm", type=float, help="default: detected from the audio")
    ap.add_argument("--offset-ms", type=float, help="song time of the first beat (with --bpm; default: found in the audio)")
    ap.add_argument("--stars", type=float, nargs="+",
                    help="one difficulty per value, all in one level (e.g. --stars 4.5 7.5 10)")
    ap.add_argument("--cfg", type=float, default=1.5)
    ap.add_argument("--temp", type=float, default=None, help="default: by the stars (sample.MAP_TEMP)")
    ap.add_argument("--top-p", type=float, default=None, help="default: by the stars (sample.MAP_TOP_P)")
    ap.add_argument("--candidates", type=int, default=1,
                    help="samples per window; with several, one far from the others in density, or with a gap or "
                         "too many walls or bombs, is passed over (fewer odd spots, but maps rate ~0.4 stars easier)")
    ap.add_argument("--music-fit", action="store_true",
                    help="of those, take the one whose notes land on the strongest hits in the music")
    ap.add_argument("--off-grid", type=float, default=0.0,
                    help="penalty on notes off the 1/16-beat grid (triplets) where the music doesn't hit (0: off; ~3)")
    ap.add_argument("--copy", type=float, default=0.0,
                    help="logits toward the notes of an earlier part that sounds like this one, as human mappers repeat them (0: off)")
    ap.add_argument("--repeat", type=float, default=4.0,
                    help="keep a part that sounds like an earlier one to that part's off-beat family, straight or triplet, as human mappers do (0: off)")
    ap.add_argument("--njs", type=float, nargs="+",
                    help="note speed, one for every difficulty or one per difficulty (default: the usual one for its stars)")
    ap.add_argument("--max-loop", type=float, default=sample.MAX_LOOP,
                    help="beats a pattern may repeat back to back before its continuation is penalized; 0: off")
    ap.add_argument("--dry", type=float, default=1.0,
                    help="penalty on a note that carries on a stretch of notes played earlier (0: off; ~1)")
    ap.add_argument("--human-times", action="store_true",
                    help="put notes only and always at the hit times of the --map difficulty (a test of the rest of the model)")
    ap.add_argument("--rhythm", nargs="?", const="on", default=None,
                    help="a rhythm model writes the whole song's hit times first and the map "
                         "model puts the notes on them, at every difficulty; a checkpoint, or no value for the released "
                         "one (default: with the newest map model or one that takes a pattern plan, the released one from rhythm.AUTO_STARS up)")
    ap.add_argument("--no-rhythm", action="store_true", help="no rhythm model: the map model times the notes itself")
    ap.add_argument("--song-aim", action="store_true",
                    help="hits per second, tech and pass as human mappers map songs like this one at those stars "
                         "(bsgen/songstars.py), not from the stars alone")
    ap.add_argument("--rhythm-temp", type=float, default=None,
                    help="the rhythm model's sampling temperature (default: by the stars, see bsgen/rhythm.py)")
    ap.add_argument("--rhythm-top-p", type=float, default=None,
                    help="the rhythm model's top-p cut (default: by the stars, see bsgen/rhythm.py)")
    ap.add_argument("--rhythm-cfg", type=float, default=None,
                    help="the rhythm model's push toward the stars asked (1: none; default: by the stars, see CFG)")
    ap.add_argument("--rhythm-audio-cfg", type=float, default=None,
                    help="the rhythm model's push toward the music over its own last beats (0: none; default "
                         "by the stars, rhythm.AUDIO_CFG)")
    ap.add_argument("--star-tries", type=int, default=None,
                    help="maps made at once per difficulty, each asked for more or fewer stars, to land on the "
                         f"stars asked: the one rated nearest them is kept (1: one map, made at the stars asked; "
                         f"default: {STAR_TRIES} on a GPU, 1 without one, where each map takes minutes)")
    ap.add_argument("--star-climb", type=float, default=STAR_CLIMB,
                    help="the most stars above the stars asked a try may be made at (default %(default)s)")
    ap.add_argument("--rhythm-tries", type=int, default=RHYTHM_TRIES,
                    help="rhythms written per difficulty; the one that keeps up with the music best is kept")
    ap.add_argument("--hold", type=float, default=sample.HOLD,
                    help="penalty on carrying on one gap between hits held too long (0: off; see HOLD)")
    ap.add_argument("--turn", type=float, default=sample.TURN_PENALTY,
                    help="penalty on a swing that turns 90° or more off the way back from the last one, "
                         f"while the map has over {sample.TURN_RATE * 100:.0f}%% of them (0: off)")
    ap.add_argument("--xtc", type=float, default=0.3, help="how often the likeliest note choices are excluded (0: off)")
    ap.add_argument("--arc-boost", type=float, default=0.0, help="logits added to arcs where one is allowed")
    ap.add_argument("--arc-rate", type=float, default=0.0,
                    help="arcs per note to steer toward (human maps that have arcs: ~0.025); 0: off")
    ap.add_argument("--walls", type=float, default=None,
                    help="walls a minute to steer toward (default: by the stars, sample.WALL_RATE; 0: the model's own)")
    ap.add_argument("--min-p", type=float, default=0.0, help="keep tokens at least this share of the likeliest's (0: top-p)")
    ap.add_argument("--style-lift", type=float, default=1.0, help=argparse.SUPPRESS)
    ap.add_argument("--pattern-plan", help=argparse.SUPPRESS)
    ap.add_argument("--style", help="acc, tech, midspeed or speed, or a blend like tech=0.7,speed=0.3. A song whose beat could be counted at 200-245 BPM or half is mapped at the fast count for speed and midspeed (and with no style at 7★+), at half for tech")
    ap.add_argument("--mapper", help="map like this mapper (see --list-mappers)")
    ap.add_argument("--list-mappers", action="store_true")
    ap.add_argument("--nps", type=float, help="notes per second over the whole map")
    ap.add_argument("--no-plan", action="store_true",
                    help="no density plan for map models that have one: each window guesses its own density")
    ap.add_argument("--pass-rating", type=float)
    ap.add_argument("--acc-rating", type=float)
    ap.add_argument("--tech-rating", type=float)
    ap.add_argument("--no-walls", action="store_true")
    ap.add_argument("--no-bombs", action="store_true")
    ap.add_argument("--no-arcs", action="store_true")
    for kind in ("wall", "bomb", "arc", "chain"):
        ap.add_argument(f"--{kind}-amount", choices=tokens.AMOUNTS, default=None,
                        help=f"how many {kind}s for the stars (default: normal" + (", or none with --no-" + kind + "s)"
                                                                                  if kind != "chain" else "; chains only when asked for)")
                        + ", for map models trained with the amounts")
    ap.add_argument("--no-lights", action="store_true", help="leave the lightshow out")
    ap.add_argument("--lights", choices=["model", "old", "rules"], default="model",
                    help="model: the lighting model (the whole song heard at once, all difficulties' shows written "
                         "together); old: the earlier lighting model; rules: lights that follow the notes and "
                         "loudness by fixed rules")
    ap.add_argument("--environment", choices=["auto", *lightmodel.ENVS], default="auto",
                    help="the game scene the map is set in; the lighting model lights it like human maps set there "
                         "(auto: one of the scenes it knows best, by the seed)")
    ap.add_argument("--format", choices=["v2", "v3"], default="v2",
                    help="v2: read by every viewer, two wall shapes; v3: exact wall shapes")
    ap.add_argument("--replay", action="store_true",
                    help="also play each difficulty with the player model and write its .bsor replay next to the map zip")
    ap.add_argument("--seed", type=int, help="the same seed and settings make the same map (default: random)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--parallel", type=int, help="difficulties made at the same time (default: as many as fit, up to 5)")
    ap.add_argument("--verbose", action="store_true", help="log every window")
    ap.add_argument("--out", default=str(ROOT / "out"))
    return ap


def load_map_model(version=hub.NEWEST, ckpt=None, dev="cuda"):
    if version in FROZEN and not ckpt:
        old = FROZEN[version]()
        return old.sample.load_model(old.hub.map_model(), dev)
    if version in hub.LEGACY and not ckpt:
        from bsgen.legacy import hub as old_hub, sample as old_sample
        return old_sample.load_model(old_hub.map_model(version), dev)
    return sample.load_model(ckpt or hub.map_model(version), dev)


def legacy_files(version):
    if version in FROZEN:
        return FROZEN[version]().hub.map_model()
    from bsgen.legacy import hub as old_hub
    return old_hub.map_model(version)


def legacy_run(args, model=None, log=print, progress=None, player=None):
    if args.model in FROZEN:
        return frozen_run(args, model, log, progress, player)
    from bsgen.legacy import generate as old
    new = vars(parser().parse_args([]))
    la = old.parser().parse_args(["--stars", "1"])
    for k, v in vars(args).items():
        if hasattr(la, k) and v != new.get(k):
            setattr(la, k, v)
    la.stars = args.stars or [songstars.MEDIAN[n] for n in args.difficulty or ["ExpertPlus"]]
    la.replay = False
    title, artist, picture = None, args.artist or "", None
    if not args.map:
        title, found, _, picture = artwork.song_info(args.audio)
        artist = artist or found or ""
    title = args.title or title
    if args.cover:
        picture = Path(args.cover).read_bytes()
    job = artwork.start(title, artist, None, picture) if title else None
    found = {}

    def cover():
        if job is None:
            return None
        found["cover"], found["artist"] = artwork.finish(job, log)
        return found["cover"]

    env = "DefaultEnvironment" if args.environment == "auto" else args.environment
    old_model = model if isinstance(model, old.sample.Generator) else None
    zip_path = old.run(la, old_model, log, progress, version=args.model, title=title, cover=cover,
                       artist=artist, environment=env)
    if zip_path is None or not args.replay:
        return zip_path
    player = player or botplay.load_player(hub.player_model(), args.device)
    names = la.difficulty or old.export.difficulty_names(la.stars)
    for star, name in zip(la.stars, names):
        tag = f" {star:g}star" if len(names) > 1 else ""
        tick = (lambda done, total, name=name: progress(f"{name} replay", done, total)) if progress else None
        s = botplay.play_map(player, zip_path, name, zip_path.with_name(f"{zip_path.stem}{tag} {name}.bsor"),
                             seed=args.seed or 0, dev=args.device, progress=tick)
        log(f"{name} replay: {botplay.describe(s)}")
    return zip_path


def _legacy12():
    from bsgen.legacy12 import generate
    return generate


def _legacy13():
    from bsgen.legacy13 import generate
    return generate


FROZEN = {"1.2": _legacy12, "1.3": _legacy13}


def frozen_run(args, model=None, log=print, progress=None, player=None):
    old = FROZEN[args.model]()
    new = vars(parser().parse_args([]))
    la = old.parser().parse_args([])
    for k, v in vars(args).items():
        if hasattr(la, k) and k != "model" and v != new.get(k):
            setattr(la, k, v)
    la.model, la.out = args.model, args.out
    if la.replay and player is None:
        player = botplay.load_player(hub.player_model(), args.device)
    old_model = model if isinstance(model, old.sample.Generator) else None
    return old.make_map(la, old_model, log, progress, player=player)


def run(args, model=None, log=print, progress=None, player=None, lighter=None):
    if not getattr(args, "youtube", None):
        return make_map(args, model, log, progress, player, lighter)
    from bsgen import youtube
    with tempfile.TemporaryDirectory(prefix="bsgen_youtube_") as folder:
        log("downloading the YouTube audio")
        try:
            path, note = youtube.download(args.youtube, folder, say=log)
            args.audio = str(path)
            if note:
                log(note)
        except youtube.YouTubeError as e:
            raise SystemExit(str(e)) from None
        return make_map(args, model, log, progress, player, lighter)


AUTO_FAST_STARS = 7.0
STYLE_OCTAVE = {"speed": "fast", "midspeed": "fast", "tech": "half"}


def style_name(args):
    if not args.style:
        return None
    parts = [p.partition("=") for p in args.style.split(",")]
    return max(parts, key=lambda p: float(p[2] or 1))[0].strip()


def octave_want(args):
    style = style_name(args)
    if style:
        return STYLE_OCTAVE.get(style)
    if args.stars:
        return "fast" if max(args.stars) >= AUTO_FAST_STARS else None
    return "fast" if set(args.difficulty or ()) & {"Expert", "ExpertPlus"} else None


def make_map(args, model=None, log=print, progress=None, player=None, lighter=None):
    if args.model in hub.LEGACY and not args.ckpt:
        return legacy_run(args, model, log, progress, player)
    if args.seed is None:
        args.seed = int(np.random.default_rng().integers(1_000_000))
    log(f"seed {args.seed}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.environment == "auto":
        args.environment = lightmodel.auto_env(args.seed, Path(args.audio or args.map or "").stem)
        log(f"scene: {args.environment}")

    import librosa
    picture, artist = None, ""
    if args.map:
        with beatmap.open_map(args.map) as z:
            info = beatmap.read_info(z)
            diff = beatmap.read_difficulty(z, info, "Standard", args.diff)
            y, sr = beatmap.read_audio(z, info)
            audio_bytes = z.read(beatmap._member(z, info.audio_file))
            artist = info.song_author
            raw_info = json.loads(z.read(beatmap._member(z, "Info.dat")).decode("utf-8-sig"))
            cover_name = raw_info.get("_coverImageFilename") or raw_info.get("coverImageFilename")
            if cover_name:
                try:
                    picture = z.read(beatmap._member(z, cover_name))
                except (KeyError, beatmap.MapError):
                    pass
        tempo, bpm, title = diff.tempo, info.bpm, info.song_name
        offset, _ = beatmap.grid_offset(diff.notes["beat"])
    else:
        audio_bytes = ogg_vorbis(args.audio)
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=True)
        title, artist, _, picture = artwork.song_info(args.audio)
        bpm, first = args.bpm, (args.offset_ms or 0) / 1000
        tempo = None
        if not bpm or args.offset_ms is None:
            logmel, fps = sample.song_mel(y, sr)
            root = hub.timing_model()
            if bpm:
                bpm, first = timing.detect(logmel, fps, root, args.device, bpm=bpm)
                log(f"timing: {bpm:.3f} BPM (given), first beat at {first * 1000:.0f} ms")
            else:
                tempo, offset, secs = timing.detect_tempo(logmel, fps, root, args.device)
                want = octave_want(args)
                secs, changed = timing.choose_octave(secs, want)
                if changed:
                    tempo, offset = timing.tempo_map(secs)
                    log(f"timing: {want} count for the {style_name(args) or 'stars asked'}: "
                        + ", ".join(f"{a:.3f} -> {b:.3f} BPM" for _, a, b in changed))
                bpm = secs[0][1]
                log("timing: " + ", then ".join(f"{b:.3f} BPM" + (f" from {s:.1f} s" if s else "")
                                                for s, b, _ in secs) + f", first beat at {secs[0][2] * 1000:.0f} ms")
        if tempo is None:
            tempo = beatmap.TempoMap(bpm)
            offset = first * bpm / 60
    dur = len(y) / sr
    title, artist = args.title or title, args.artist or artist
    if args.cover:
        picture = Path(args.cover).read_bytes()
    cover_job = artwork.start(title, artist, dur, picture)
    n_beats = int(tempo.to_beats(np.array([dur]))[0])
    log(f"song: {dur:.0f} s, {n_beats} beats")
    beat_times = tempo.to_seconds(np.arange(0, n_beats + 16, dtype=float))
    grid_times = tempo.to_seconds(np.arange(0, n_beats + 16, dtype=float) + offset)
    logmel, fps = sample.song_mel(y, sr)
    end_beat = float(tempo.to_beats(np.array([sample.song_end(logmel, fps)]))[0])
    model = model or load_map_model(args.model, args.ckpt, args.device)
    if args.list_mappers:
        log("\n".join(model.mappers) or "this model has no mapper control")
        return None
    stars = args.stars
    if not stars:
        if not args.difficulty:
            raise ValueError("give --stars, or --difficulty to take each one's stars from the song")
        by_song = songstars.stars(logmel, fps)
        stars = [by_song[name] for name in args.difficulty]
        log("stars from the song: " + ", ".join(f"{n} {s:g}★" for n, s in zip(args.difficulty, stars)))
    if len(stars) > len(export.DIFF_RANK):
        raise ValueError(f"at most {len(export.DIFF_RANK)} difficulties in one map")
    names = args.difficulty or export.difficulty_names(stars)
    njs = args.njs or [export.typical_njs(x) for x in stars]
    if len(names) != len(stars) or len(njs) not in (1, len(stars)):
        raise ValueError("give one --difficulty per --stars value, and one --njs or one per --stars value")
    if len(set(names)) < len(names):
        raise ValueError(f"two difficulties would share a name ({', '.join(names)}); name them with --difficulty")
    njs = njs * len(stars) if len(njs) == 1 else njs
    objects = (0 if args.no_walls else tokens.OBJ_WALLS) | (0 if args.no_bombs else tokens.OBJ_BOMBS) \
        | (0 if args.no_arcs else tokens.OBJ_ARCS)
    styles = None
    if args.style:
        styles = {}
        for part in args.style.split(","):
            name, _, w = part.partition("=")
            if name.strip() not in tokens.STYLES:
                raise ValueError(f"unknown style {name!r}; one of {', '.join(tokens.STYLES)}")
            styles[name.strip()] = float(w or 1)
    if not getattr(model, "controls", False) and (styles or args.mapper or args.nps):
        log("note: this model only takes a star rating; the other controls are ignored")

    auto_rhythm = args.rhythm is None and (getattr(model, "pattern", False) or (args.model == hub.NEWEST and not args.ckpt))
    rpath = hub.rhythm_model() if args.rhythm == "on" or (auto_rhythm and max(stars) >= rhythm.AUTO_STARS) \
        else args.rhythm
    rhythm_model = rhythm.loaded(rpath, args.device) if rpath and not args.no_rhythm and not args.human_times else None
    changes = list(zip(tempo.beats[1:], tempo.bpms[1:])) if tempo.has_changes else []

    def plan(ask, speed):
        aim = songstars.how(logmel, fps, ask) if args.song_aim else None
        amounts = {f"{k}s": tokens.AMOUNTS.index(v) for k in ("wall", "bomb", "arc", "chain")
                   if (v := getattr(args, f"{k}_amount")) is not None}
        controls = dict(njs=speed, nps=args.nps, mapper=args.mapper, objects=objects, acc=args.acc_rating, amounts=amounts,
                        tech=args.tech_rating if args.tech_rating is not None or not aim else aim["tech"],
                        **{"pass": args.pass_rating if args.pass_rating is not None or not aim else aim["pass"]})
        return aim, controls

    kept_rhythms = {}
    style_lift = args.style_lift * sum(w * STYLE_PLAN_LIFT[s] for s, w in styles.items()) / sum(styles.values()) if styles else 0.0

    def rhythms(jobs):
        if not (planned_model or RHYTHM_AT_WANTED):
            return write_rhythms(jobs)
        new = [j for j in dict.fromkeys(jobs) if j not in kept_rhythms]
        kept_rhythms.update(zip(new, write_rhythms(new)))
        return [kept_rhythms[j] for j in jobs]

    def write_rhythms(jobs):
        if rhythm_model is None or args.human_times:
            return [None] * len(jobs)
        out = [None] * len(jobs)
        use = [j for j, (ask, *_) in enumerate(jobs) if not auto_rhythm or ask >= rhythm.AUTO_STARS]
        style = max(styles, key=styles.get) if styles else None
        for c in range(0, len(use), RHYTHM_BATCH):
            part = use[c:c + RHYTHM_BATCH]
            plans = [plan(*jobs[j][:2]) for j in part]
            asks = [jobs[j][0] for j in part]
            wanted = [jobs[j][2] for j in part]
            levels = [aim and aim["hps"] for aim, _ in plans]
            ratings = [None if k["tech"] is None and k["pass"] is None else (k["tech"], k["pass"]) for _, k in plans]
            for i, ask in enumerate(asks):
                if style is not None and STYLE_AIM:
                    typ = sample.typical_ratings_for(ask, style)
                    levels[i] = levels[i] or typ["nps"] * HITS_PER_NOTE
                    ratings[i] = ratings[i] or (typ["tech"], typ["pass"])
            with sample.GATE.reading():
                rgen = torch.Generator(device=args.device).manual_seed(args.seed)
            one = len(part) == 1
            got = rhythm.best_of(rhythm_model, logmel, fps, grid_times, 0.0, n_beats, asks[0] if one else asks,
                                 args.rhythm_tries, level=levels[0] if one else levels, style=style,
                                 temp=args.rhythm_temp, top_p=args.rhythm_top_p, cfg=args.rhythm_cfg, dev=args.device,
                                 gen=rgen, ratings=ratings[0] if one else ratings, audio_cfg=args.rhythm_audio_cfg,
                                 wanted=wanted[0] if one else wanted, check=getattr(progress, "check", None))
            for j, t in zip(part, [got] if one else got):
                out[j] = t
        return out

    def level(star, name, speed, ask, times=None, progress=progress, say=None):
        say = say or ((lambda m: log(f"{name}: {m}")) if len(stars) > 1 else log)
        with sample.GATE.reading():
            gen = torch.Generator(device=args.device).manual_seed(args.seed)
            stream = torch.cuda.Stream() if str(args.device).startswith("cuda") else None
        rng = np.random.default_rng(args.seed)
        planned = getattr(model, "pattern", False)
        aim, controls = plan(star if planned else ask, speed)
        if args.human_times:
            if not args.map:
                raise SystemExit("--human-times needs --map")
            times = np.unique(np.round((diff.notes["beat"] - offset) * tokens.TICKS).astype(np.int64))
        elif times is not None:
            say(f"rhythm: {len(times)} hits" + (f" (aim {aim['hps']:.1f}/s, tech {aim['tech']:.1f}, pass {aim['pass']:.1f})" if aim else ""))
        made_at, patterns = ask, None
        if planned and times is not None:
            made_at = star
            patterns = pattern.make_plan(pattern_plan, times, grid_times, 0.0, star, logmel, fps, song_summary,
                                         lift=(ask - star) * PLAN_LIFT + (0.0 if pattern_plan.get("style") else style_lift),
                                         rng=np.random.default_rng(args.seed), styles=styles)
        hooks = []
        with torch.cuda.stream(stream) if stream else contextlib.nullcontext():
            temp, top_p = sample.map_sampling(made_at)
            temp, top_p = temp if args.temp is None else args.temp, top_p if args.top_p is None else args.top_p
            ev, b = sample.generate(model, logmel, fps, grid_times, 0.0, n_beats, made_at, args.cfg, temp,
                                    top_p, dev=args.device, log=say if args.verbose else None,
                                    candidates=args.candidates, controls=controls, styles=styles,
                                    max_loop=args.max_loop, dry=args.dry, xtc=args.xtc, min_p=args.min_p,
                                    turn=args.turn, hold=args.hold, times=times, pattern=patterns, arc_boost=args.arc_boost, arc_rate=args.arc_rate, walls=args.walls, music_fit=args.music_fit, off_grid=args.off_grid, repeat=args.repeat, copy_notes=args.copy,
                                    use_plan=not args.no_plan, hooks=hooks, gen=gen, rng=rng, floor=sample.xtc_floor(star), top=sample.top_bias(star), fast=star,
                                    progress=lambda beat: progress(name, beat, n_beats) if progress else None)
        bombs, walls, arcs = tokens.to_objects(ev, offset)
        chains = tokens.to_chains(ev, offset)
        notes = tokens.to_notes(ev, offset)
        late = int((notes["beat"] > end_beat).sum())
        notes, bombs, walls = (x[x["beat"] <= end_beat] for x in (notes, bombs, walls))
        walls["duration"] = np.minimum(walls["duration"], end_beat - walls["beat"])
        arcs = arcs[arcs["tail_beat"] <= end_beat]
        chains = chains[chains["beat"] <= end_beat]
        if not args.no_bombs:
            bombs = sample.reset_bombs(notes, bombs, walls, tempo.to_seconds, args.seed or 0)
        if late:
            say(f"{late} notes in the song's quiet ending left out")
        secs = tempo.to_seconds(notes["beat"])
        span = secs.max() - secs.min() if len(notes) else 1
        say(f"{ask:g}★, NJS {speed:g}: {len(notes)} notes, {len(notes) / span:.2f} notes/s, "
            f"swings {np.sum(ev['kind'] == SWING)} ({np.mean(ev['air'][ev['kind'] == SWING]):.0%} air), "
            f"bombs {len(bombs)}, walls {len(walls)}, arcs {len(arcs)}" + (f", chains {len(chains)}" if len(chains) else ""))
        say(f"probability the model put on moves the rules blocked: mean {b.mean():.1%}, "
            f"steps where it was over half: {np.mean(b > 0.5):.1%}")
        est = rating.estimate(notes, speed, tempo.to_seconds(notes["beat"]))
        if est:
            say(f"estimated from the notes: {est[0]:.1f}★, tech {est[1]:.1f}, pass {est[2]:.1f}"
                + (f" (made at {ask:g}★)" if ask != star else ""))
        return dict(difficulty=name, stars=star, ask=ask, est=est and round(est[0], 1), notes=notes,
                    label=f"bsgen {star:g}★" + (f" (est {est[0]:.1f})" if est else ""), njs=speed, bombs=bombs,
                    walls=walls, arcs=arcs, chains=chains, on=rhythm.on_sound(np.unique(secs), marks))

    marks = rhythm.accents(logmel, fps)
    planned_model = getattr(model, "pattern", False)
    pattern_plan = pattern.load_plan(args.pattern_plan or hub.pattern_plan()) if planned_model else None
    song_summary = songstars.features(logmel, fps) if planned_model else None

    def finish(lv):
        name, notes = lv["difficulty"], lv["notes"]
        say = (lambda m: log(f"{name}: {m}")) if len(stars) > 1 else log
        with sample.GATE.reading():
            stream = torch.cuda.Stream() if str(args.device).startswith("cuda") else None
        if args.no_lights:
            show = []
        elif "lights" in lv:
            show = lv["lights"]
        elif args.lights == "rules":
            show = lights.lightshow(notes, beat_times, logmel, fps, args.seed)
        else:
            show = lightmodel.lightshow(lighter, logmel, fps, beat_times, offset, notes, args.environment,
                                        seed=args.seed, dev=args.device, same_colour=lighter.song_in is not None,
                                        repeat=lightmodel.REPEAT_SIM, calm=lightmodel.CALM, calm_from=lightmodel.CALM_FROM)
        if show:
            say(f"lights: {len(show)} events, {len(show) / max(notes['beat'].max() - notes['beat'].min(), 1):.1f} per beat")
        lv["lights"] = show
        if args.replay:
            with tempfile.TemporaryDirectory() as tmp:
                one = export.write_pack(Path(tmp) / "level", b"", bpm, title, [lv], tempo_changes=changes,
                                        fmt=args.format)
                tick = (lambda done, total: progress(f"{name} replay", done, total)) if progress else None
                with torch.cuda.stream(stream) if stream else contextlib.nullcontext():
                    lv["motion"] = botplay.play_motion(player, one.parent / (one.name + ".zip"), name,
                                                       seed=args.seed or 0, dev=args.device, progress=tick)
        return lv

    if args.replay:
        player = player or botplay.load_player(hub.player_model(), args.device)
    if not args.no_lights and args.lights == "old":
        lighter = lighter if isinstance(lighter, lightmodel.Lighter) else lightmodel.load(hub.light_model(), args.device)
    elif not args.no_lights and args.lights == "model":
        lighter = lighter if isinstance(lighter, lightseq.LightSeq) else lightseq.load(hub.lightseq_model(), args.device)
    parallel = args.parallel
    if parallel is None:
        free = torch.cuda.mem_get_info()[0] / 1e9 if str(args.device).startswith("cuda") else len(stars)
        parallel = int(free // GB_PER_DIFFICULTY)
    tries = args.star_tries if args.star_tries is not None else STAR_TRIES if str(args.device).startswith("cuda") else 1
    k = max(1, tries) if rating.available() else 1
    workers = max(1, min(parallel, len(stars) * k))
    with ThreadPoolExecutor(workers) as ex, sample.together(workers):
        made = [[] for _ in stars]
        asks = [first_asks(s, k, planned_model) for s in stars]
        rounds = STAR_ROUNDS if k > 1 else 1
        for round_ in range(rounds):
            todo = [(i, j, a) for i in range(len(stars)) for j, a in enumerate(asks[i])]
            if not todo:
                break
            quiet = (lambda m: None) if k > 1 and not args.verbose else None
            times = rhythms([(stars[i] if RHYTHM_AT_WANTED or planned_model else a, njs[i], stars[i]) for i, _, a in todo])
            beats = {}

            def tick(i, j):
                def on_beat(name, beat, total):
                    beats[i, j] = beat
                    n = len(asks[i])
                    progress(name, sum(beats.get((i, x), 0) for x in range(n)), total * n)
                return on_beat if progress else None

            for (i, _, _), lv in zip(todo, ex.map(lambda t, h: level(stars[t[0]], names[t[0]], njs[t[0]], t[2], h,
                                                                     tick(t[0], t[1]),
                                                                     None if t[1] == 0 else quiet), todo, times)):
                made[i].append(lv)
            asks = [next_asks(s, m, k, args.star_climb) for s, m in zip(stars, made)]
        levels, in_order = pick_levels(stars, made)
        if not in_order:
            log("the difficulties' ratings don't rise with their stars on this song; try another seed")
        for s, lv, m in zip(stars, levels, made):
            if len(m) > 1:
                log(f"{lv['difficulty']}: kept the one made at {lv['ask']:g}★, rated {lv['est']:.1f}★ of {s:g} asked"
                    f" ({len(m)} made: " + ", ".join(f"{x['ask']:g}→{x['est']}" for x in m) + ")")
        if not args.no_lights:
            log("making the lights")
        if not args.no_lights and args.lights == "model":
            shows, _ = lightseq.lightshows(lighter, logmel, fps, beat_times, offset, [lv["notes"] for lv in levels], bpm,
                                           seed=args.seed or 0, dev=args.device)
            for lv, show in zip(levels, shows):
                lv["lights"] = show
        levels = list(ex.map(finish, levels))
    tag = ",".join(f"{x:g}" for x in stars)
    tag += "star"
    if all(lv["est"] for lv in levels):
        tag += " (est " + ",".join(f"{lv['est']:.1f}" for lv in levels) + ")"
    folder = Path(args.out) / export.safe_name(f"{title[:40]} {tag}")
    for n in range(2, 1000):
        if not folder.exists() and not folder.with_name(folder.name + ".zip").exists():
            break
        folder = folder.with_name(export.safe_name(f"{title[:40]} {tag} {n}"))
    cover, found_artist = artwork.finish(cover_job, log)
    artist = artist or found_artist or ""
    export.write_pack(folder, audio_bytes, bpm, f"{title} (bsgen {stars[0]:g}★)" if len(stars) == 1 else f"{title} (bsgen)",
                      levels, tempo_changes=changes, fmt=args.format, cover=cover, artist=artist,
                      environment=args.environment)
    log(f"wrote {folder}")
    zip_path = folder.parent / (folder.name + ".zip")
    for lv in levels:
        if "motion" in lv:
            name = lv["difficulty"]
            stars = f" {lv['stars']:g}star" if len(levels) > 1 else ""
            out = zip_path.with_name(f"{zip_path.stem}{stars} {name}.bsor")
            s = botplay.save_replay(zip_path, name, lv["motion"], out)
            log(f"{name} replay: {botplay.describe(s)}")
    return zip_path


def main():
    run(parser().parse_args(), log=lambda s: print(s, flush=True))


if __name__ == "__main__":
    main()
