"""Export report/report.md to report/report.pdf.

Route (first that works):
  1. pandoc (if installed)                      -> PDF via LaTeX
  2. python-markdown -> HTML -> headless Chrome/Chromium/Edge  --print-to-pdf
  3. python-markdown -> HTML -> WeasyPrint      (needs pango system libs)
If all fail, report.html is still written and the exact manual command is printed.

Usage: python scripts/export_pdf.py [--md report/report.md]
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CSS = """
@page { size: A4; margin: 18mm 16mm 18mm 16mm; }
body { font-family: "Charter", "Georgia", "Times New Roman", serif; font-size: 10.5pt; line-height: 1.45;
       color: #111; max-width: 180mm; margin: auto; }
h1 { font-size: 20pt; text-align: center; margin-bottom: 4pt; }
h2 { font-size: 14pt; border-bottom: 1px solid #999; padding-bottom: 2pt; margin-top: 18pt; page-break-after: avoid; }
h3 { font-size: 11.5pt; margin-top: 12pt; page-break-after: avoid; }
p, li { text-align: justify; }
code { font-family: "Menlo", "Consolas", monospace; font-size: 8.8pt; background: #f3f3f3; padding: 0 2px; }
pre { background: #f6f6f6; border: 1px solid #ddd; padding: 6pt; font-size: 8.6pt; overflow-x: auto;
      white-space: pre-wrap; page-break-inside: avoid; }
table { border-collapse: collapse; margin: 8pt auto; font-size: 8.3pt; page-break-inside: avoid; }
th, td { border: 1px solid #bbb; padding: 2pt 4pt; text-align: center; }
th { background: #eee; }
img { max-width: 100%; display: block; margin: 6pt auto; page-break-inside: avoid; }
blockquote { color: #444; border-left: 3px solid #ccc; margin-left: 0; padding-left: 8pt; }
.caption, em.caption { display: block; text-align: center; font-size: 9pt; color: #333; }
"""

CHROME_CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    "google-chrome", "chromium", "chromium-browser", "msedge",
]


def md_to_html(md_path: Path) -> Path:
    import markdown
    text = md_path.read_text(encoding="utf-8")
    body = markdown.markdown(text, extensions=["tables", "fenced_code", "toc", "attr_list", "sane_lists"])
    html = (f"<!doctype html><html><head><meta charset='utf-8'><title>{md_path.stem}</title>"
            f"<style>{CSS}</style></head><body>{body}</body></html>")
    out = md_path.with_suffix(".html")
    out.write_text(html, encoding="utf-8")
    return out


def try_pandoc(md: Path, pdf: Path) -> bool:
    if not shutil.which("pandoc"):
        return False
    r = subprocess.run(["pandoc", str(md), "-o", str(pdf), "--resource-path", str(md.parent)],
                       capture_output=True)
    return r.returncode == 0 and pdf.exists()


def try_chrome(html: Path, pdf: Path) -> bool:
    for c in CHROME_CANDIDATES:
        exe = c if Path(c).exists() else shutil.which(c)
        if not exe:
            continue
        cmd = [exe, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
               "--allow-file-access-from-files", f"--print-to-pdf={pdf}", html.resolve().as_uri()]
        r = subprocess.run(cmd, capture_output=True, timeout=180)
        if r.returncode == 0 and pdf.exists() and pdf.stat().st_size > 10_000:
            return True
    return False


def try_weasy(html: Path, pdf: Path) -> bool:
    try:
        from weasyprint import HTML
        HTML(filename=str(html)).write_pdf(str(pdf))
        return True
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", default=str(ROOT / "report" / "report.md"))
    a = ap.parse_args()
    md = Path(a.md)
    pdf = md.with_suffix(".pdf")
    if pdf.exists():
        pdf.unlink()
    if try_pandoc(md, pdf):
        print(f"PDF via pandoc: {pdf}"); return 0
    html = md_to_html(md)
    print(f"HTML: {html}")
    if try_chrome(html, pdf):
        print(f"PDF via headless Chrome: {pdf}"); return 0
    if try_weasy(html, pdf):
        print(f"PDF via WeasyPrint: {pdf}"); return 0
    print("Could not create a PDF automatically. Open the HTML in a browser and 'Print → Save as PDF', or run:\n"
          f"  pandoc {md} -o {pdf} --resource-path {md.parent}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
