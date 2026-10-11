"""Tests for the public-access features: PWA files, QR code, feedback, stats, privacy, security headers."""
import json
import os
import sys
import tempfile
import unittest

os.environ["FC_NO_AUTOSTART"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from forward_checker import config as _cfg
_cfg.GEMINI_MODELS = ["m"]        # one model per key: scripted test responses map one-to-one
_cfg.GROQ_MODELS = ["m"]
_cfg.GROQ_VISION_MODELS = ["m"]
_cfg.GROQ_WHISPER_MODELS = ["m"]
_cfg.LEGACY_GEMINI_MODELS = []
sys.path.insert(0, os.path.dirname(__file__))

from fastapi.testclient import TestClient

from forward_checker import config
from forward_checker.cache import Cache
from forward_checker.service import BotService, RateLimiter
from forward_checker.voice import VoiceStore
from server import create_app, whatsapp_url
from test_phase2 import StubLLM, SPLIT, VERDICT, retriever, write_file

WA = "https://wa.me/14155238886?text=join%20test-word"


class PublicAccessTests(unittest.TestCase):
    def make(self, wa=WA, voice=True):
        d = tempfile.mkdtemp()
        v = VoiceStore(d, tts=lambda t, l, p: write_file(p)) if voice else None
        svc = BotService(router=StubLLM(SPLIT, VERDICT), retriever=retriever(), cache=Cache(":memory:"),
                         voice=v, limiter=RateLimiter(50), public_base_url="https://x.test" if voice else "")
        return svc, TestClient(create_app(svc, twilio=None, meta=None, wa_url=wa))

    def test_pwa_files(self):
        _, c = self.make()
        m = c.get("/manifest.webmanifest")
        self.assertEqual(m.status_code, 200)
        self.assertIn("manifest+json", m.headers["content-type"])
        data = m.json()
        self.assertEqual(data["name"], "Truth_Rail")
        self.assertEqual(data["display"], "standalone")
        self.assertTrue(any(i["purpose"] == "maskable" for i in data["icons"]))
        for icon in data["icons"]:
            self.assertEqual(c.get(icon["src"]).status_code, 200, icon["src"])
        sw = c.get("/sw.js")
        self.assertEqual(sw.status_code, 200)
        self.assertIn("javascript", sw.headers["content-type"])
        self.assertIn('"/api"', sw.text)          # the service worker must never cache API calls

    def test_home_page_content(self):
        _, c = self.make()
        html = c.get("/").text
        for needle in ("Truth_Rail", "/manifest.webmanifest", 'data-lang="hi"', 'data-lang="mr"', "/privacy"):
            self.assertIn(needle, html)

    def test_privacy_page(self):
        _, c = self.make()
        r = c.get("/privacy")
        self.assertEqual(r.status_code, 200)
        self.assertIn("do not send private or personal information", r.text)

    def test_config_with_and_without_whatsapp(self):
        _, c = self.make()
        cfg = c.get("/api/config").json()
        self.assertEqual(cfg["whatsapp_url"], WA)
        self.assertTrue(cfg["voice_replies"])
        _, c2 = self.make(wa=None, voice=False)
        cfg2 = c2.get("/api/config").json()
        self.assertIsNone(cfg2["whatsapp_url"])
        self.assertFalse(cfg2["voice_replies"])

    def test_whatsapp_url_builder(self):
        from unittest import mock
        with mock.patch.object(config, "WHATSAPP_NUMBER", "+91 98765-43210"), \
             mock.patch.object(config, "WHATSAPP_JOIN_TEXT", "join blue-tiger"):
            self.assertEqual(whatsapp_url(), "https://wa.me/919876543210?text=join%20blue-tiger")
        with mock.patch.object(config, "WHATSAPP_NUMBER", ""):
            self.assertIsNone(whatsapp_url())

    def test_qr_codes(self):
        _, c = self.make()
        r = c.get("/qr.svg")
        self.assertEqual((r.status_code, r.headers["content-type"]), (200, "image/svg+xml"))
        self.assertIn("<svg", r.text)
        self.assertEqual(c.get("/qr.svg", params={"target": "web"}).status_code, 200)
        _, c2 = self.make(wa=None)
        self.assertEqual(c2.get("/qr.svg").status_code, 404)              # no number set
        self.assertEqual(c2.get("/qr.svg", params={"target": "web"}).status_code, 200)

    def test_feedback_and_stats(self):
        svc, c = self.make()
        self.assertEqual(c.post("/api/feedback", json={"rating": "up", "label": "False", "kind": "text"}).status_code, 200)
        self.assertEqual(c.post("/api/feedback", json={"rating": "down", "label": "<script>", "kind": "evil"}).status_code, 200)
        self.assertEqual(c.post("/api/feedback", json={"rating": "meh"}).status_code, 422)
        self.assertEqual(svc.cache.stats()["feedback"], {"up": 1, "down": 1})
        c.post("/api/check", data={"text": "Lemon cures cancer"})
        self.assertEqual(c.get("/api/stats").json(), {"checks": 1, "served_from_cache": 0})

    def test_security_headers_and_hidden_docs(self):
        _, c = self.make()
        h = c.get("/").headers
        self.assertEqual(h["x-content-type-options"], "nosniff")
        self.assertIn("default-src 'self'", h["content-security-policy"])
        self.assertIn("frame-ancestors 'none'", h["content-security-policy"])
        for path in ("/docs", "/redoc", "/openapi.json"):
            self.assertEqual(c.get(path).status_code, 404, path)

    def test_static_path_traversal_blocked(self):
        _, c = self.make()
        self.assertEqual(c.get("/static/../server.py").status_code, 404)
        self.assertEqual(c.get("/static/%2e%2e/server.py").status_code, 404)


if __name__ == "__main__":
    unittest.main()
