import os
import re
import subprocess
from pathlib import Path
from urllib.parse import urlparse

MAX_MINUTES = 15
HOSTS = ("youtube.com", "youtu.be", "youtube-nocookie.com")


class _Silent:
    debug = warning = error = staticmethod(lambda msg: None)


class YouTubeError(Exception):
    pass


def video_url(text):
    text = (text or "").strip()
    if not re.match(r"^https?://", text, re.I):
        text = "https://" + text if re.match(r"^(?:[\w-]+\.)?(?:youtube\.com|youtu\.be)/", text, re.I) else text
    u = urlparse(text)
    host = (u.hostname or "").lower()
    if u.scheme not in ("http", "https") or not any(host == h or host.endswith("." + h) for h in HOSTS):
        return None
    return text


def fresh_visitor():
    import requests
    try:
        page = requests.get("https://www.youtube.com/", timeout=15, headers={
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:130.0) Gecko/20100101 Firefox/130.0",
            "Accept-Language": "en-US,en;q=0.9"}).text
    except requests.RequestException:
        return None
    m = re.search(r'"VISITOR_DATA":"([^"]+)"', page)
    return m.group(1) if m else None


CLIENTS = [None, ["web", "visionos"], ["tv", "android_vr"], ["mweb", "web_safari"]]
SEARCHES = ["scsearch1:{}", "ytsearch1:{} audio"]
GIVE_UP = r"unavailable|privat\w+|removed|copyright|members|age"


def _title(url):
    import requests
    try:
        r = requests.get("https://www.youtube.com/oembed", params={"url": url, "format": "json"}, timeout=15)
        r.raise_for_status()
        j = r.json()
        return j.get("title"), j.get("author_name")
    except (requests.RequestException, ValueError):
        return None


def thumbnail(path):
    import io
    import requests
    from PIL import Image
    m = re.fullmatch(r"youtube_([\w-]{11})\.\w+", Path(path).name)
    if not m:
        return None
    for size in ("maxresdefault", "hqdefault"):
        try:
            r = requests.get(f"https://i.ytimg.com/vi/{m.group(1)}/{size}.jpg", timeout=15)
            if r.status_code != 200:
                continue
            with Image.open(io.BytesIO(r.content)) as im:
                im = im.convert("RGB")
                w, h = im.size
                if h > w * 0.7:
                    h = round(w * 9 / 16)
                    im = im.crop((0, (im.size[1] - h) // 2, w, (im.size[1] + h) // 2))
                s = min(w, h)
                im = im.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
                out = io.BytesIO()
                im.save(out, "PNG")
                return out.getvalue()
        except (requests.RequestException, OSError):
            continue
    return None


def song_tags(info):
    title = (info.get("track") or info.get("title") or "").strip()
    artist = (info.get("artist") or info.get("creator") or "").split(",")[0].strip()
    if not artist and not info.get("track"):
        channel = re.sub(r"\s*-\s*Topic$|VEVO$", "", info.get("uploader") or info.get("channel") or "").strip()
        title = re.sub(r"^\s*\d{1,3}\s*[-.)]\s*(?=\D)", "", title)
        left, dash, right = title.partition(" - ")
        if dash and right.strip():
            artist, title = left.strip(), right.strip()
        else:
            artist = channel
    title = re.sub(r"[(\[][^()\[\]]*(?:remaster|version|video|audio|lyric|mv|hd|4k|visuali[sz]er)[^()\[\]]*[)\]]", "",
                   title, flags=re.I)
    return " ".join(title.split()).strip(" -"), artist


def _retag(path, info):
    title, artist = song_tags(info)
    if not title:
        return
    tmp = path.with_name("tagged_" + path.name)
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(path), "-vn", "-c:a", "copy", "-map_metadata", "-1",
           "-metadata:s:a:0", f"title={title}"] + (["-metadata:s:a:0", f"artist={artist}"] if artist else []) + [str(tmp)]
    try:
        if subprocess.run(cmd, capture_output=True, timeout=60).returncode == 0:
            tmp.replace(path)
    except (OSError, subprocess.SubprocessError):
        pass
    tmp.unlink(missing_ok=True)


def _fetch(yt_dlp, target, folder, extra):
    def reject(info, incomplete=False):
        if info.get("is_live") or info.get("live_status") in ("is_live", "is_upcoming"):
            return "a live stream"
        if (info.get("duration") or 0) > MAX_MINUTES * 60:
            return f"longer than {MAX_MINUTES} minutes"
        return None

    opts = {
        "format": "bestaudio/best", "noplaylist": True, "logger": _Silent(), "socket_timeout": 20,
        "outtmpl": str(folder / "%(extractor)s_%(id)s.%(ext)s"), "match_filter": reject, "restrictfilenames": True,
        "js_runtimes": {"deno": {}, "node": {}},
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "vorbis", "preferredquality": "6"},
                           {"key": "FFmpegMetadata", "add_metadata": True}],
    } | extra
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(target, download=True)
    except yt_dlp.utils.DownloadError as e:
        return None, re.sub(r"\x1b\[[0-9;]*m", "", str(e))
    if info and "entries" in info:
        info = next(iter(info["entries"]), None)
    if not info:
        return None, "rejected: a live stream, or longer than the limit"
    files = sorted(folder.glob(f"*{info['id']}*.ogg"))
    if files:
        _retag(files[0], info)
    return (files[0], info) if files else (None, "no audio file came out")


def download(text, folder, say=None):
    url = video_url(text)
    if url is None:
        raise YouTubeError("That doesn't look like a YouTube link.")
    try:
        import yt_dlp
    except ImportError:
        raise YouTubeError("YouTube links need yt-dlp: uv sync (or pip install yt-dlp).") from None
    say = say or (lambda msg: None)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    browser = os.environ.get("BSGEN_COOKIES_FROM_BROWSER")
    routes = [("fresh visitor", c) for c in CLIENTS] + ([("browser login", None)] if browser else [])
    errors = []
    for name, clients in routes:
        extra = {}
        args = {}
        visitor = None if name == "browser login" else fresh_visitor()
        if visitor:
            args["visitor_data"] = [visitor]
        if clients:
            args["player_client"] = clients
        if args:
            extra["extractor_args"] = {"youtube": args}
        if name == "browser login":
            extra["cookiesfrombrowser"] = (browser,)
        path, got = _fetch(yt_dlp, url, folder, extra)
        if path:
            return path, None
        errors.append(got)
        say(f"YouTube route {len(errors)} of {len(routes)} failed")
        if re.search(GIVE_UP, got, re.I) and not re.search(r"sign in|bot|cookies|format", got, re.I):
            raise YouTubeError("That video isn't available.")
    found = _title(url)
    if found and found[0]:
        title = re.sub(r"\s+", " ", found[0]).strip()
        for pattern in SEARCHES:
            say("looking for the song by its title")
            path, got = _fetch(yt_dlp, pattern.format(title), folder, {})
            if path:
                return path, (f"YouTube wouldn't give out that video's audio, so this is another upload of "
                              f"\u201c{title}\u201d (from {got.get('extractor', 'another site')}). Check it's the right version.")
    if any(re.search(r"sign in|bot|cookies", e, re.I) for e in errors):
        raise YouTubeError("YouTube asked for a sign-in and every route failed. YouTube blocks some automatic "
                           "downloads, so upload the audio file instead (or set BSGEN_COOKIES_FROM_BROWSER=firefox "
                           "to use a browser's login).")
    raise YouTubeError("Couldn't get that video's audio from YouTube. YouTube blocks some automatic downloads, "
                       "and every route failed: upload the audio file instead. Last error: " + (errors[-1] if errors else "").replace("ERROR: ", "")[:200])
