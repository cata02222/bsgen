import io
import json
import re
import subprocess
import threading
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import Future
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

HEADERS = {"User-Agent": "bsgen/1.2 (https://github.com/cata02222/bsgen)"}
MAX_IMAGE = 12 * 1024 * 1024
_lock = threading.Lock()
_last_request = 0.0
_workers = threading.BoundedSemaphore(2)


def start(title, artist="", duration=None, picture=None):
    job = Future()
    if not _workers.acquire(blocking=False):
        job.set_result(None)
        return job
    job.set_running_or_notify_cancel()

    def work():
        try:
            job.set_result(prepare(title, artist, duration, picture, lambda message: None))
        except Exception as error:
            job.set_exception(error)
        finally:
            _workers.release()

    threading.Thread(target=work, name="cover", daemon=True).start()
    return job


def finish(job, log=print):
    if not job.done():
        job.cancel()
        log("cover: lookup still running; exporting with the plain cover")
        return plain(), None
    try:
        found = job.result() or {}
    except Exception:
        log("cover: couldn't load artwork; exporting with the plain cover")
        return plain(), None
    cover = found.get("cover")
    log("cover: ready" if cover else "cover: no artwork; using the plain cover")
    return cover or plain(), found.get("artist")


def song_info(path):
    title, artist, duration, picture = Path(path).stem, "", None, None
    tagged_title = False
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)],
            capture_output=True, check=True, timeout=15)
        data = json.loads(result.stdout)
        streams = [s for s in data.get("streams", []) if s.get("codec_type") == "audio"]
        tags = {k.lower(): v for s in streams[:1] for k, v in s.get("tags", {}).items()}
        tags |= {k.lower(): v for k, v in data.get("format", {}).get("tags", {}).items()}
        title, artist = tags.get("title") or title, tags.get("artist") or ""
        if re.fullmatch(r"\s*\d{1,3}\s*", artist):
            artist = ""
        tagged_title = bool(tags.get("title"))
        duration = float(data.get("format", {}).get("duration", 0)) or None
        stream = next((s for s in data.get("streams", [])
                       if s.get("disposition", {}).get("attached_pic")), None)
        if stream is not None:
            picture = subprocess.run(
                ["ffmpeg", "-v", "error", "-i", str(path), "-map", f"0:{stream['index']}",
                 "-frames:v", "1", "-vf", "scale=512:512:force_original_aspect_ratio=decrease",
                 "-f", "image2pipe", "-c:v", "png", "-"],
                capture_output=True, check=True, timeout=15).stdout
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    if not tagged_title:
        title = re.sub(r"^\s*(?:\d{1,3}\s*[-.)]|0\d\s)\s*(?=\D)", "", title)
    if not artist and " - " in title:
        artist, title = title.split(" - ", 1)
    if not tagged_title:
        title = re.sub(r"[_ ]\d{2,4}(?:kbps|k)\b", "", title, flags=re.I)
    return clean_title(title), artist.strip(), duration, picture


JUNK = r"official(?: (?:audio|video|music video|visualizer))?|lyrics?|lyric video|explicit|clean|audio|hq|hd|" \
       r"unreleased|best version|\d{2,4}k|(?:19|20)\d\d|(?:ft\.?|feat\.?|featuring) [^)\]]*"


def clean_title(title):
    title = re.sub(rf"[(\[]\s*(?:{JUNK})\s*[)\]]", "", title, flags=re.I)
    title = re.sub(r"\s+(?:ft\.?|feat\.?)\s+.*$", "", title, flags=re.I)
    return " ".join(title.split()).strip(" -")


def _normal(text):
    text = unicodedata.normalize("NFKD", text).casefold()
    return " ".join(re.findall(r"\w+", text))


def _similar(a, b):
    return SequenceMatcher(None, _normal(a), _normal(b)).ratio()


def _artist_matches(artist, credits):
    names = [c.get("name", c.get("artist", {}).get("name", ""))
             for c in credits if isinstance(c, dict)]
    joined = "".join(c.get("name", c.get("artist", {}).get("name", "")) + c.get("joinphrase", "")
                     for c in credits if isinstance(c, dict))
    return _similar(artist, joined) >= 0.9 or bool(names and _similar(artist, names[0]) >= 0.9)


def _get(url, params=None):
    if params:
        url += "?" + urllib.parse.urlencode(params)
    return urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=5)


def _json(url, params=None):
    with _get(url, params) as response:
        return json.loads(response.read(MAX_IMAGE))


def _download(url):
    with _get(url) as response:
        data = response.read(MAX_IMAGE + 1)
    if len(data) > MAX_IMAGE:
        raise ValueError("Cover image is too large")
    return data


def _quoted(text):
    return '"' + re.sub(r'([+\-!(){}\[\]^"~*?:\\/|&])', r'\\\1', text) + '"'


def _catalog(entity, params):
    global _last_request
    with _lock:
        for attempt in range(2):
            time.sleep(max(0, (2.5 if attempt else 1.1) - (time.monotonic() - _last_request)))
            _last_request = time.monotonic()
            try:
                return _json(f"https://musicbrainz.org/ws/2/{entity}/", {"fmt": "json", **params})
            except urllib.error.HTTPError as error:
                if not attempt and error.code in (429, 503):
                    continue
                raise


def _bare(title):
    return re.sub(r"[(\[][^)\]]*[)\]]", "", title).strip() or title


NO_ARTIST_S = 1.0


def pick(found, title, artist="", duration=None):
    want = _bare(title)
    ok = [f for f in found if _similar(want, _bare(f[0])) >= 0.85 and f[3]]
    if duration and not artist:
        ok = [f for f in ok if f[2] and abs(f[2] - duration) <= NO_ARTIST_S]
    if artist:
        ok = [f for f in ok if _similar(artist, f[1]) >= 0.85 or _normal(f[1]) in _normal(artist)
              or _normal(artist) in _normal(f[1])]
    elif not duration or len({_normal(f[1]) for f in ok}) > 1:
        return None
    if duration:
        ok.sort(key=lambda f: abs((f[2] or duration) - duration))
    return ok[0] if ok else None


def search(title, artist="", duration=None):
    q = f"{artist} {_bare(title)}".strip()
    try:
        data = _json("https://api.deezer.com/search", {"q": q, "limit": 10}).get("data", [])
        found = [(d.get("title", ""), d.get("artist", {}).get("name", ""), d.get("duration"),
                  d.get("album", {}).get("cover_xl")) for d in data]
        if hit := pick(found, title, artist, duration):
            return hit
    except (OSError, ValueError, AttributeError):
        pass
    try:
        data = _json("https://itunes.apple.com/search", {"term": q, "entity": "song", "limit": 10})
        found = [(d.get("trackName", ""), d.get("artistName", ""), (d.get("trackTimeMillis") or 0) / 1000 or None,
                  (d.get("artworkUrl100") or "").replace("100x100bb", "600x600bb") or None)
                 for d in data.get("results", [])]
        return pick(found, title, artist, duration)
    except (OSError, ValueError, AttributeError):
        return None


def lookup(title, artist="", duration=None):
    if not _normal(title):
        return None
    query = f"recording:{_quoted(title)}"
    if artist:
        query += f" AND artist:{_quoted(artist)}"
    if duration:
        query += f" AND dur:[{max(0, int((duration - 8) * 1000))} TO {int((duration + 8) * 1000)}]"
    try:
        matches = []
        for recording in _catalog("recording", {"query": query, "limit": 10}).get("recordings", []):
            if recording.get("video"):
                continue
            credits = recording.get("artist-credit", [])
            who = "".join(c.get("name", c.get("artist", {}).get("name", "")) + c.get("joinphrase", "")
                          for c in credits if isinstance(c, dict))
            if _similar(title, recording.get("title", "")) < 0.9:
                continue
            if artist and not _artist_matches(artist, credits):
                continue
            if duration and recording.get("length") and abs(duration - recording["length"] / 1000) > 8:
                continue
            matches.append((recording, who))
        if not artist and len({_normal(who) for _, who in matches}) > 1:
            return None
        releases = [r for rec, _ in matches for r in rec.get("releases", [])]
        releases = [r for r in releases if r.get("status", "Official") == "Official"]
        releases = [r for r in releases if not set(r.get("release-group", {}).get("secondary-types", []))
                    & {"Compilation", "DJ-mix"}]
        expected = artist or (matches[0][1] if matches else "")
        releases = [r for r in releases if not r.get("artist-credit") or
                    _artist_matches(expected, r["artist-credit"])]
        releases.sort(key=lambda r: (_similar(title, r.get("title", "")) < 0.9,
                                     bool(r.get("release-group", {}).get("secondary-types")),
                                     r.get("date") or "9999"))
        urls = []
        for release in releases:
            group = release.get("release-group", {}).get("id")
            url = (f"https://coverartarchive.org/release-group/{group}/front-500" if group else
                   f"https://coverartarchive.org/release/{release['id']}/front-500")
            if url not in urls:
                urls.append(url)
        for url in urls[:3]:
            try:
                picture = _download(url)
                with Image.open(io.BytesIO(picture)) as image:
                    image.verify()
                return picture
            except (OSError, ValueError, Image.DecompressionBombError):
                continue
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


def badge(data):
    with Image.open(io.BytesIO(data)) as source:
        source = ImageOps.exif_transpose(source)
        source.thumbnail((1024, 1024))
        image = ImageOps.fit(source.convert("RGB"), (512, 512), method=Image.Resampling.LANCZOS).convert("RGBA")
    with Image.open(Path(__file__).parents[1] / "assets" / "logo.png") as source:
        logo = source.convert("RGBA")
        logo = logo.crop(logo.getbbox())
        logo.thumbnail((180, 54), Image.Resampling.LANCZOS)
    x, y = 30, 512 - 30 - logo.height
    overlay = Image.new("RGBA", image.size)
    ImageDraw.Draw(overlay).rounded_rectangle(
        (x - 14, y - 11, x + logo.width + 14, y + logo.height + 11), radius=11, fill=(0, 0, 0, 180))
    overlay.alpha_composite(logo, (x, y))
    image = Image.alpha_composite(image, overlay).convert("RGB")
    out = io.BytesIO()
    image.save(out, "JPEG", quality=92)
    return out.getvalue()


def plain():
    x = np.linspace(-1, 1, 512)
    xx, yy = np.meshgrid(x, x)
    base = np.array([16, 16, 20], float)
    red = np.exp(-((xx + 0.9) ** 2 + yy ** 2) / 0.5)[..., None] * np.array([90, 10, 20])
    blue = np.exp(-((xx - 0.9) ** 2 + yy ** 2) / 0.5)[..., None] * np.array([10, 30, 100])
    image = Image.fromarray(np.clip(base + red + blue, 0, 255).astype(np.uint8)).convert("RGBA")
    with Image.open(Path(__file__).parents[1] / "assets" / "logo.png") as source:
        logo = source.convert("RGBA")
        logo = logo.crop(logo.getbbox())
        logo.thumbnail((340, 340), Image.Resampling.LANCZOS)
    image.alpha_composite(logo, ((512 - logo.width) // 2, (512 - logo.height) // 2))
    out = io.BytesIO()
    image.convert("RGB").save(out, "JPEG", quality=92)
    return out.getvalue()


def find(title, artist="", duration=None):
    hit = search(title, artist, duration)
    for get in ((lambda: _download(hit[3])) if hit else None, lambda: lookup(title, artist, duration)):
        if get is None:
            continue
        try:
            picture = get()
            if picture:
                Image.open(io.BytesIO(picture)).verify()
                return picture, hit[1] if hit else None
        except (OSError, ValueError, SyntaxError, Image.DecompressionBombError):
            pass
    return None, hit[1] if hit else None


def prepare(title, artist="", duration=None, picture=None, log=print):
    if picture:
        hit = search(title, artist, duration)
        try:
            return {"cover": badge(picture), "artist": hit[1] if hit else None}
        except (OSError, ValueError, Image.DecompressionBombError):
            log("cover: couldn't read the supplied image; trying a song lookup")
    log("cover: looking for song artwork")
    picture, found_artist = find(title, artist, duration)
    if picture:
        try:
            return {"cover": badge(picture), "artist": found_artist}
        except (OSError, ValueError, Image.DecompressionBombError):
            pass
    log("cover: no confident match; using the plain cover (add an image in Options for another)")
    return {"cover": plain(), "artist": found_artist}
