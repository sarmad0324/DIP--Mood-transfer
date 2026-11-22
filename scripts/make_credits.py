"""Copy selected candidate images into data/ and write CREDITS.md from their metadata.

Selection file (JSON): {"content": {"dest_stem": "cand_dir/cand_xxx.jpg", ...},
                        "moods":   {"golden_hour": "cand_dir/cand_xxx.jpg", ...}}
Each candidate directory contains the ``meta.json`` written by fetch_commons.py.
Images are re-encoded (max 1600 px long side, JPEG q=92) which also strips metadata.

Usage: python scripts/make_credits.py --cand <candidate root> --selection selection.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402

from src.utils import load_image, to_uint8  # noqa: E402

HEADER = {
    "content": "# Content images — credits & licences\n\n"
               "Test-set content photographs (structure to preserve). All files are from Wikimedia Commons "
               "under free licences; metadata was retrieved with `scripts/fetch_commons.py` from the "
               "Wikimedia API at download time. Images were resized to ≤1600 px and re-encoded as JPEG "
               "(a modification, as required to state by CC BY/BY-SA).\n\n",
    "moods": "# Mood reference images — credits & licences\n\n"
             "Mood/atmosphere references used as presets. All files are from Wikimedia Commons under free "
             "licences (metadata retrieved via the Wikimedia API). Images were resized to ≤1600 px and "
             "re-encoded as JPEG (modified).\n\n",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True)
    ap.add_argument("--selection", required=True)
    a = ap.parse_args()
    cand = Path(a.cand)
    sel = json.loads(Path(a.selection).read_text())
    for kind in ("content", "moods"):
        out_dir = ROOT / "data" / kind
        out_dir.mkdir(parents=True, exist_ok=True)
        lines = [HEADER[kind], "| File | Original title | Author | Licence | Source |", "|---|---|---|---|---|"]
        for stem, rel in sel[kind].items():
            src = cand / rel
            meta = json.loads((src.parent / "meta.json").read_text())[src.name]
            img = load_image(src, max_side=1600)
            cv2.imwrite(str(out_dir / f"{stem}.jpg"), to_uint8(img)[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])
            author = " ".join(meta["author"].split()).split(" NOTE:")[0][:120].replace("|", "/")
            lic = meta["license"] + (f" ({meta['license_url']})" if meta.get("license_url") else "")
            title = meta["title"].replace("File:", "").replace("|", "/")
            lines.append(f"| `{stem}.jpg` | {title} | {author} | {lic} | {meta['page_url']} |")
            print(f"{kind}/{stem}.jpg  <- {title} [{meta['license']}]")
        (out_dir / "CREDITS.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
