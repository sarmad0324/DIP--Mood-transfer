"""Fetch freely-licensed candidate images (with author/license metadata) from Wikimedia Commons.

Only files whose license is CC0, Public Domain, CC BY or CC BY-SA are kept.
Metadata is written next to the images as ``meta.json`` so that CREDITS.md can be
generated from it (see ``scripts/make_credits.py``).

Usage:
    python scripts/fetch_commons.py --query "golden hour landscape" --out cand/golden --n 8
    python scripts/fetch_commons.py --article "Blue hour" --out cand/blue --n 10
    python scripts/fetch_commons.py --title "File:Some_name.jpg" --out data/moods --name golden_hour
"""
from __future__ import annotations

import argparse
import json
import re
import time
import urllib.parse
import subprocess
from pathlib import Path

# HTTP is done with the system `curl` (with IPv4 forced): on the build network Python's
# urllib stalled for ~20 s per request on a black-holed IPv6 route, curl did not.


def _curl(url: str, timeout: int = 60) -> bytes:
    r = subprocess.run(["curl", "-4", "-sSfL", "-m", str(timeout), "-A", UA, url],
                       capture_output=True, check=False)
    if r.returncode != 0:
        raise IOError(f"curl failed ({r.returncode}) for {url}: {r.stderr.decode()[:200]}")
    return r.stdout

# Commons file metadata is served by every Wikimedia wiki for "shared" files; we use the
# en.wikipedia endpoint because commons.wikimedia.org was unreachable from the build network.
API = "https://en.wikipedia.org/w/api.php"
SEARCH_API = "https://commons.wikimedia.org/w/api.php"
UA = "MoodTransferCourseProject/1.0 (educational; python-urllib)"
OK_LICENSES = re.compile(r"^(cc0|public domain|pd|cc by(-sa)? ?\d|cc-by(-sa)?-\d)", re.I)


def _get(params: dict, api: str = API) -> dict:
    params = {**params, "format": "json"}
    url = api + "?" + urllib.parse.urlencode(params)
    return json.loads(_curl(url))


def _clean(html: str) -> str:
    return re.sub(r"<[^>]+>", "", html or "").strip()


def _info_from_page(page: dict) -> dict | None:
    ii = (page.get("imageinfo") or [{}])[0]
    md = ii.get("extmetadata", {})
    lic = _clean(md.get("LicenseShortName", {}).get("value", ""))
    if not OK_LICENSES.match(lic):
        return None
    return {
        "title": page["title"],
        "page_url": ii.get("descriptionurl", ""),
        "download_url": ii.get("thumburl") or ii.get("url"),
        "author": _clean(md.get("Artist", {}).get("value", "unknown")),
        "license": lic,
        "license_url": md.get("LicenseUrl", {}).get("value", ""),
        "width": ii.get("width"), "height": ii.get("height"),
    }


def search(query: str, n: int, width: int = 1280) -> list[dict]:
    data = _get({"action": "query", "generator": "search", "gsrsearch": f"{query} filetype:bitmap",
                 "gsrnamespace": 6, "gsrlimit": min(50, n * 4), "prop": "imageinfo",
                 "iiprop": "url|extmetadata|size", "iiurlwidth": width}, api=SEARCH_API)
    pages = sorted(data.get("query", {}).get("pages", {}).values(), key=lambda p: p.get("index", 0))
    out = []
    for p in pages:
        info = _info_from_page(p)
        if info and (info["width"] or 0) >= 1000:
            out.append(info)
        if len(out) >= n:
            break
    return out


def from_article(article: str, n: int, width: int = 1280) -> list[dict]:
    """Freely-licensed photos (JPEG, >=1000 px wide) used in a Wikipedia article."""
    data = _get({"action": "query", "generator": "images", "titles": article, "gimlimit": 100,
                 "prop": "imageinfo", "iiprop": "url|extmetadata|size", "iiurlwidth": width})
    out = []
    for p in data.get("query", {}).get("pages", {}).values():
        if not p["title"].lower().endswith((".jpg", ".jpeg")):
            continue
        info = _info_from_page(p)
        if info and (info["width"] or 0) >= 1000:
            out.append(info)
    return out[:n]


def by_title(title: str, width: int = 1280) -> dict | None:
    data = _get({"action": "query", "titles": title, "prop": "imageinfo",
                 "iiprop": "url|extmetadata|size", "iiurlwidth": width})
    page = next(iter(data["query"]["pages"].values()))
    return _info_from_page(page)


def download(info: dict, dst: Path) -> None:
    dst.write_bytes(_curl(info["download_url"], timeout=60))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query"); ap.add_argument("--title"); ap.add_argument("--article")
    ap.add_argument("--out", required=True); ap.add_argument("--name")
    ap.add_argument("--n", type=int, default=8); ap.add_argument("--width", type=int, default=1280)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    meta_path = out / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    if a.title:
        infos = [by_title(a.title, a.width)]
    elif a.article:
        infos = from_article(a.article, a.n, a.width)
    else:
        infos = search(a.query, a.n, a.width)
    for i, info in enumerate(infos):
        if info is None:
            print("skipped: license not in allow-list"); continue
        ext = Path(urllib.parse.urlparse(info["download_url"]).path).suffix.lower() or ".jpg"
        name = (a.name or f"cand_{len(meta):03d}") + ext
        try:
            download(info, out / name)
        except Exception as e:  # throttled / timed out: skip this file
            print(f"skip {info['title']}: {e}")
            continue
        meta[name] = info
        print(f"{name}: {info['title']} | {info['license']} | {info['author'][:60]}")
        time.sleep(1.0)
    meta_path.write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
