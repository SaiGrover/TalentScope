"""Capture README screenshots from a running TalentScope server.

    streamlit run app.py --server.port 8521        # in one terminal
    pip install playwright                         # uses your installed Chrome, no browser download
    python scripts/capture_screenshots.py --base http://localhost:8521
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "screenshots"

SHOTS = [
    # (file, path, height, element to scroll to before capturing or None)
    ("overview.png", "/", 1500, None),
    ("attrition.png", "/attrition", 1500, ".st-key-card_ai_job_role"),
    ("performance.png", "/performance", 1500, ".st-key-card_perf_rank"),
    ("heldout.png", "/performance", 1300, ".st-key-card_perf_roc_test"),
    ("explainability.png", "/explainability", 1400, ".st-key-card_xai_shap"),
    ("simulator.png", "/simulator", 1400, None),
    ("model_lab.png", "/model-lab", 1300, None),
    ("data_lab.png", "/data-lab", 1300, None),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8521")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p, tempfile.TemporaryDirectory() as profile:
        ctx = p.chromium.launch_persistent_context(profile, channel="chrome", headless=True,
                                                   viewport={"width": 1440, "height": 1000}, device_scale_factor=1)
        page = ctx.new_page()
        for name, path, height, anchor in SHOTS:
            page.set_viewport_size({"width": 1440, "height": height})
            page.goto(args.base + path, wait_until="networkidle")
            page.wait_for_selector(".ts-title", timeout=90_000)
            page.wait_for_function("() => !document.querySelector('[data-testid=\"stStatusWidget\"]')", timeout=120_000)
            page.wait_for_timeout(2500)
            if anchor:
                page.locator(anchor).first.scroll_into_view_if_needed()
                page.evaluate("window.scrollBy(0, -110)")
                page.wait_for_timeout(1500)
            page.screenshot(path=str(OUT / name))
            print("saved", name)
        ctx.close()


if __name__ == "__main__":
    main()
