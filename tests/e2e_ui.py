"""Optional browser test for the Truth_Rail web page (needs: pip install playwright && playwright install chromium).
Starts the real server with a stand-in AI (no keys, no internet) and clicks through the page.
Run:  python tests/e2e_ui.py   (screenshots go to ./ui_screenshots)"""
import os
import re
import sys
import tempfile
import threading
import time

os.environ["FC_NO_AUTOSTART"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uvicorn
from playwright.sync_api import sync_playwright

from forward_checker.cache import Cache
from forward_checker.evidence import Evidence, EvidenceRetriever
from forward_checker.service import BotService, RateLimiter
from forward_checker.voice import VoiceStore
from server import create_app

OUT = os.path.join(os.getcwd(), "ui_screenshots")
os.makedirs(OUT, exist_ok=True)
EV = [Evidence("Fact check", "https://factcheck.afp.com/doc.afp.com.9JA8ZM", "No evidence.")]


class LoopLLM:
    last_provider, last_key, keys = "gemini", "gemini#1", {"gemini#1": "x"}

    def generate_json(self, system, user):
        time.sleep(0.3)
        if "claim extractor" in system:
            lang = "hi" if re.search(r"[\u0900-\u097F]", user) else "en"
            return {"language": lang, "claims": [{"text": user.strip()[:90], "topic": "other"}],
                    "manipulation_flags": ["chain_forward_request", "fake_urgency"] if "Share" in user or "शेयर" in user else []}
        if "Aadhaar" in user:
            return {"label": "Outdated", "reason": "The deadline was extended to 30 June 2023 and has now passed.", "source_ids": [1]}
        if "lemon" in user.lower() or "नींबू" in user:
            return {"label": "False", "reason": "Health experts say no drink can cure cancer.", "source_ids": [1]}
        return {"label": "Cannot be confirmed", "reason": "No trusted source talks about this.", "source_ids": []}

    def transcribe(self, kind, data, mime, prompt, filename="f"):
        return "Drinking hot water with lemon cures cancer. Share with everyone!"


EMOJI = r"[\u2600-\u27BF\U0001F300-\U0001FAFF\U0001F000-\U0001F2FF]"
SIZES = [(320, 700), (390, 844), (768, 1024), (1024, 768), (1440, 900)]
OVERFLOW_JS = """() => { const W = document.documentElement.clientWidth, bad = [];
  document.querySelectorAll('body *').forEach(e => { const r = e.getBoundingClientRect();
    if (r.width && r.right > W + 1 && getComputedStyle(e).position !== 'fixed' && !e.closest('svg')) bad.push(e.tagName + '.' + e.className + ' right=' + Math.round(r.right)); });
  return {W, scrollW: document.documentElement.scrollWidth, bad: bad.slice(0, 6)}; }"""


def shot(pg, **kw):
    """Scroll back to the top first, so sticky headers are captured in the right place."""
    pg.evaluate("window.scrollTo({top: 0, behavior: 'instant'})")
    pg.wait_for_timeout(300)
    pg.screenshot(**kw)


def main():
    d = tempfile.mkdtemp()
    svc = BotService(router=LoopLLM(), retriever=EvidenceRetriever(search_fn=lambda q: EV), cache=Cache(":memory:"),
                     voice=VoiceStore(d, tts=lambda t, l, p: open(p, "wb").close()), limiter=RateLimiter(500),
                     public_base_url="http://127.0.0.1:8777")
    app = create_app(svc, twilio=None, meta=None, wa_url="https://wa.me/14155238886?text=join%20test-word")
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=8777, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    time.sleep(1.5)
    base, problems = "http://127.0.0.1:8777", []
    sample = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "samples", "lemon_forward.png")

    def check(cond, msg):
        print(("PASS  " if cond else "FAIL  ") + msg)
        if not cond:
            problems.append(msg)

    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream"])
        ctx = browser.new_context(viewport={"width": 390, "height": 844}, device_scale_factor=2,
                                  permissions=["microphone"], is_mobile=True, has_touch=True)
        page = ctx.new_page()
        errors = []
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" and "Failed to load resource" not in m.text else None)
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(base)
        page.wait_for_selector("#go")
        shot(page, path=f"{OUT}/01_mobile_home.png", full_page=True)
        check(page.title().startswith("Truth_Rail"), "page title")
        check(page.locator("#waBtn").is_visible(), "WhatsApp button shown when a number is configured")
        check(page.locator("link[rel=manifest]").count() == 1, "manifest linked")
        check(page.locator("svg.i").count() > 20, "icons are SVG")
        check(not re.search(EMOJI, page.inner_text("body")), "no emoji anywhere on the page")
        check(page.locator("#tabs [role=tab]").count() == 4, "four input tabs")

        # text check (lemon) -> False
        page.click("[data-ex^='Drinking hot water']")
        page.click("#go")
        page.wait_for_selector(".banner")
        check("false" in page.inner_text(".banner h2").lower(), "lemon forward shows 'This is false'")
        check(page.locator(".src").count() >= 1, "source link shown")
        check(page.locator(".warn").count() == 1, "manipulation warning shown")
        check("wa.me" in page.locator("a:has-text('Share on WhatsApp')").get_attribute("href"), "share-on-WhatsApp link")
        check(not re.search(EMOJI, page.inner_text("#result")), "no emoji inside the result")
        shot(page, path=f"{OUT}/02_mobile_result.png", full_page=True)

        # language switch re-renders the result and the page
        page.click("[data-lang=hi]")
        check("झूठ" in page.inner_text(".banner h2") or "झूठ" in page.inner_text("#result"), "Hindi verdict text")
        check("जाँच" in page.inner_text("#checker"), "page text switches to Hindi")
        shot(page, path=f"{OUT}/03_mobile_hindi.png", full_page=True)
        page.click("[data-lang=mr]")
        check("खोटे" in page.inner_text("#result"), "Marathi verdict text")
        page.click("[data-lang=en]")

        # outdated claim
        page.click("[data-ex^='Last date']")
        page.click("#go")
        page.wait_for_selector(".banner h2:has-text('out of date')")
        check(True, "Aadhaar example shows the Outdated message")

        # history, feedback
        check(page.locator("#hist button").count() == 2, "history lists 2 checks")
        page.locator("#hist button").nth(1).click()
        check("false" in page.inner_text(".banner h2").lower(), "clicking history re-opens a result")
        with page.expect_response("**/api/feedback") as fb:
            page.click(".fb button >> nth=0")
        check(fb.value.status == 200, "feedback accepted")

        # tabs: keyboard + panels
        page.click("#tab-image")
        check(page.locator("#panel-image").is_visible() and not page.locator("#panel-text").is_visible(), "image tab shows the drop zone")
        page.keyboard.press("ArrowRight")
        check(page.get_attribute("#tab-voice", "aria-selected") == "true", "arrow keys move between tabs")

        # screenshot upload
        page.reload()
        page.click("#tab-image")
        page.set_input_files("#fileImage", sample)
        check(page.locator("#panel-image .picked img").is_visible(), "screenshot preview shown")
        with page.expect_response("**/api/check") as r:
            page.click("#go")
        check(r.value.json()["kind"] == "image", "server read the screenshot as an image")

        # PDF upload
        page.reload()
        page.click("#tab-pdf")
        pdf = os.path.join(tempfile.mkdtemp(), "doc.pdf")
        open(pdf, "wb").write(b"%PDF-1.4 not a real pdf")
        page.set_input_files("#filePdf", pdf)
        check(page.locator("#panel-pdf .picked").is_visible(), "PDF chip shown")
        with page.expect_response("**/api/check") as r:
            page.click("#go")
        check(r.value.status == 400 and "PDF" in r.value.json()["error"], "unreadable PDF gives a clear message")
        check(page.locator(".err").is_visible(), "error panel shown")

        # voice recording (fake microphone)
        page.reload()
        page.click("#tab-voice")
        page.click("#panel-voice .mic")
        page.wait_for_selector("#panel-voice .mic.on")
        page.wait_for_timeout(1800)
        page.click("#panel-voice .mic.on")
        page.wait_for_selector("#panel-voice audio")
        with page.expect_response("**/api/check") as r:
            page.click("#go")
        check(r.value.json()["kind"] == "audio", "recording is sent and read as audio")

        # empty submit
        page.reload()
        page.click("#go")
        check(page.locator(".err").is_visible(), "empty submit shows a friendly message")

        # settings, QR, stats, PWA, headers
        page.click("#menuBtn")
        check(page.locator("#menu").is_visible(), "settings menu opens")
        page.click("[data-size='2']")
        check(page.evaluate("document.documentElement.style.fontSize") == "130%", "text size changes")
        page.click("[data-theme-set='dark']")
        check(page.get_attribute("html", "data-theme") == "dark", "dark theme switch")
        page.keyboard.press("Escape")
        check(not page.locator("#menu").is_visible(), "Escape closes the menu")
        qr = page.request.get(base + "/qr.svg")
        check(qr.status == 200 and "svg" in qr.headers["content-type"], "QR code endpoint")
        for _ in range(40):
            if ctx.service_workers:
                break
            page.wait_for_timeout(250)
        check(bool(ctx.service_workers), "service worker registered")
        check(page.request.get(base + "/docs").status == 404, "API docs page is not exposed")

        # server error is shown nicely (not a blank page)
        svc.router = type("Broken", (), {"keys": {"x": 1}, "ready": True, "last_key": None, "last_model": None,
                                         "generate_json": lambda self, s, u: (_ for _ in ()).throw(RuntimeError("model exploded")),
                                         "transcribe": lambda *a, **k: "x"})()
        page.reload()
        page.fill("#text", "something to check")
        page.click("#go")
        page.wait_for_selector(".err")
        check("Technical details" in page.inner_text(".err") and "model exploded" in page.text_content(".err"),
              "crash shows a friendly message with technical details (on localhost)")
        check(page.locator(".err button:has-text('Try again')").count() == 1, "retry button offered")
        svc.router = LoopLLM()
        ctx.close()

        # responsive: no sideways scroll at any size, with a result on screen
        for w, h in SIZES:
            c = browser.new_context(viewport={"width": w, "height": h}, device_scale_factor=1)
            pg = c.new_page()
            pg.goto(base)
            pg.wait_for_selector("#go")
            if w == 1440:
                shot(pg, path=f"{OUT}/05_desktop_top.png")
            if w == 768:
                shot(pg, path=f"{OUT}/05_tablet_top.png")
            pg.click("[data-ex^='Drinking hot water']")
            pg.click("#go")
            pg.wait_for_selector(".banner")
            pg.wait_for_timeout(1200)
            res = pg.evaluate(OVERFLOW_JS)
            check(res["scrollW"] <= res["W"] and not res["bad"], f"no sideways scroll at {w}px  {res['bad'][:2]}")
            if w in (768, 1440):
                shot(pg, path=f"{OUT}/04_{'tablet' if w == 768 else 'desktop'}_result.png", full_page=True)
            c.close()

        dctx = browser.new_context(viewport={"width": 1440, "height": 900}, color_scheme="dark")
        dpage = dctx.new_page()
        dpage.goto(base)
        dpage.click("[data-ex^='Drinking hot water']")
        dpage.click("#go")
        dpage.wait_for_selector(".banner")
        dpage.wait_for_timeout(1200)
        shot(dpage, path=f"{OUT}/06_desktop_dark.png", full_page=True)
        check(True, "dark desktop screenshot saved")

        check(not errors, "no console errors: " + "; ".join(errors[:3]))
        browser.close()
    server.should_exit = True
    print("\nALL UI CHECKS PASSED" if not problems else f"\n{len(problems)} UI CHECK(S) FAILED")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
