"""End-to-end GUI check: launch app.py, drive it in a real browser, verify output, take a screenshot.

1. API check with gradio_client (every method through the real event handler).
2. Browser check with Playwright (system Chrome, no download): upload a content image, keep the
   default preset, click "Transfer mood", wait for the output image, save a screenshot to
   report/figures/gui_screenshot.png.

Usage: python app.py &   then   python scripts/check_gui.py [--url http://127.0.0.1:7860]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def api_check(url: str) -> None:
    from gradio_client import Client, handle_file
    client = Client(url, verbose=False)
    content = handle_file(str(ROOT / "data" / "content" / "lucca_rooftops.jpg"))
    for method in ("Reinhard", "Histogram", "AdaIN", "AdaIN + mood enhancements", "Gatys"):
        t0 = time.time()
        res = client.predict(
            content, None, "Golden Hour", None, "(none — use uploaded image)", 0.5, method, 1.0,
            "color-only mood (keep luminance)" if "mood" in method else "full mood",
            "mood" in method, "guided", 384, 60, 0, 0, 0, 0, 0, 0, api_name="/run")
        out_img, file_path, info = res[1], res[2], res[4]
        assert out_img and Path(file_path).exists(), f"{method}: no output"
        print(f"[api] {method:28s} OK in {time.time() - t0:5.1f}s  -> {info.splitlines()[0][:80]}")
    # multi-reference blending + post-processing
    res = client.predict(content, None, "Golden Hour", None, "Cold Blue Dusk", 0.5, "AdaIN + mood enhancements",
                         0.8, "color + tone (photographic)", True, "guided", 384, 60, 0.3, 0.2, 0.5, 0.3, 0.4, 0,
                         api_name="/run")
    assert Path(res[2]).exists()
    print("[api] blending + post-processing OK")
    # error handling: no content image
    try:
        client.predict(None, None, "Golden Hour", None, "(none — use uploaded image)", 0.5, "AdaIN", 1.0,
                       "full mood", False, "guided", 384, 60, 0, 0, 0, 0, 0, 0, api_name="/run")
        raise AssertionError("expected an error for missing content")
    except Exception as e:  # gradio raises AppError with our message
        assert "content image" in str(e).lower(), e
        print("[api] missing-content error handled:", str(e)[:70])


def browser_check(url: str, shot: Path) -> None:
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(viewport={"width": 1500, "height": 1250}, device_scale_factor=1)
        page.goto(url, wait_until="networkidle")
        page.locator("input[type=file]").first.set_input_files(str(ROOT / "data" / "content" / "arezzo_piazza.jpg"))
        page.wait_for_timeout(1500)
        page.get_by_role("button", name="Transfer mood").click()
        page.get_by_text("Output mood:").wait_for(timeout=60000)
        page.wait_for_timeout(1500)
        shot.parent.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(shot), full_page=True)
        browser.close()
    print(f"[browser] upload → transfer → output OK; screenshot {shot.relative_to(ROOT)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:7860")
    a = ap.parse_args()
    api_check(a.url)
    browser_check(a.url, ROOT / "report" / "figures" / "gui_screenshot.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
