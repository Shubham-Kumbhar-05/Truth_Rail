"""Model chains, discovery, retired models, optional-setting retries, error details and diagnostics."""
import json
import os
import sys
import unittest

os.environ["FC_NO_AUTOSTART"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(__file__))

from fastapi.testclient import TestClient

from forward_checker import config
from forward_checker.cache import Cache
from forward_checker.llm_router import LLMError, LLMRouter, discover_flash_models
from forward_checker.service import BotService, RateLimiter
from server import create_app
from test_core import Resp, gemini_ok, groq_ok, Clock
from test_phase2 import StubLLM, retriever


class ModelPost:
    """Answers by (key, model) so tests can script each combination."""
    def __init__(self, script, default=None):
        self.script, self.default, self.log, self.payloads = script, default, [], []

    def __call__(self, url, **kw):
        h = kw.get("headers", {})
        key = h.get("x-goog-api-key") or h.get("Authorization", "").replace("Bearer ", "")
        model = url.split("/models/")[1].split(":")[0] if "/models/" in url else kw.get("json", {}).get("model", "?")
        self.log.append((key, model))
        self.payloads.append(kw.get("json"))
        q = self.script.get((key, model))
        if q:
            return q.pop(0)
        if self.default is not None:
            return self.default
        raise AssertionError(f"unscripted call {key}/{model}")


QUOTA = lambda: Resp(429, text="quota exceeded PerMinute")
GONE = lambda: Resp(404, text='{"error":{"message":"models/old-model is not found for API version v1beta"}}')

LISTING = {"models": [
    {"name": "models/gemini-3.5-flash", "supportedGenerationMethods": ["generateContent"]},
    {"name": "models/gemini-3.5-flash-lite", "supportedGenerationMethods": ["generateContent"]},
    {"name": "models/gemini-3.1-flash-lite", "supportedGenerationMethods": ["generateContent"]},
    {"name": "models/gemini-2.5-flash", "supportedGenerationMethods": ["generateContent"]},
    {"name": "models/gemini-3.5-flash-image", "supportedGenerationMethods": ["generateContent"]},
    {"name": "models/gemini-3.5-flash-preview-05", "supportedGenerationMethods": ["generateContent"]},
    {"name": "models/gemini-3.10-flash", "supportedGenerationMethods": ["generateContent"]},
    {"name": "models/gemini-embedding-001", "supportedGenerationMethods": ["embedContent"]},
    {"name": "models/gemini-4-flash", "supportedGenerationMethods": ["countTokens"]},
]}


class DiscoveryTests(unittest.TestCase):
    def test_picks_newest_stable_flash_models(self):
        self.assertEqual(discover_flash_models(LISTING), ["gemini-3.10-flash", "gemini-3.5-flash", "gemini-3.5-flash-lite"])
        self.assertEqual(discover_flash_models(LISTING, top=5)[-2:], ["gemini-3.1-flash-lite", "gemini-2.5-flash"])
        self.assertEqual(discover_flash_models({}), [])

    def test_router_uses_discovered_models_then_defaults(self):
        get = lambda url, **kw: Resp(200, LISTING)
        r = LLMRouter(["k1"], "", http_post=ModelPost({}), http_get=get, discover=True, gemini_models=None)
        r._explicit_gemini = []
        chain = r.gemini_models()
        self.assertEqual(chain[:3], ["gemini-3.10-flash", "gemini-3.5-flash", "gemini-3.5-flash-lite"])
        self.assertIn("gemini-2.5-flash-lite", chain)             # defaults follow as backups
        self.assertEqual(len(chain), len(set(chain)))

    def test_discovery_failure_falls_back_to_defaults(self):
        def boom(url, **kw):
            raise OSError("offline")
        r = LLMRouter(["k1"], "", http_post=ModelPost({}), http_get=boom, discover=True)
        r._explicit_gemini = []
        self.assertEqual(r.gemini_models(), config.DEFAULT_GEMINI_MODELS)

    def test_explicit_models_skip_discovery(self):
        calls = []
        r = LLMRouter(["k1"], "", http_post=ModelPost({}), http_get=lambda *a, **k: calls.append(1), discover=True,
                      gemini_models=["only-this"])
        self.assertEqual(r.gemini_models(), ["only-this"])
        self.assertEqual(calls, [])


class ChainTests(unittest.TestCase):
    def test_next_model_on_same_key_before_next_key(self):
        post = ModelPost({("k1", "a"): [QUOTA()], ("k1", "b"): [gemini_ok({"v": 1})]})
        r = LLMRouter(["k1", "k2"], "", http_post=post, gemini_models=["a", "b"])
        self.assertEqual(r.generate_json("s", "u"), {"v": 1})
        self.assertEqual(post.log, [("k1", "a"), ("k1", "b")])
        self.assertEqual(r.last_model, "b")

    def test_rate_limited_model_is_skipped_next_time(self):
        post = ModelPost({("k1", "a"): [QUOTA()], ("k1", "b"): [gemini_ok({}), gemini_ok({})]})
        r = LLMRouter(["k1"], "", http_post=post, gemini_models=["a", "b"])
        r.generate_json("s", "u")
        r.generate_json("s", "u")
        self.assertEqual(post.log, [("k1", "a"), ("k1", "b"), ("k1", "b")])

    def test_retired_model_is_dropped_for_every_key(self):
        post = ModelPost({("k1", "old-model"): [GONE()], ("k1", "new"): [gemini_ok({}), gemini_ok({})],
                          ("k2", "new"): [gemini_ok({})]})
        r = LLMRouter(["k1", "k2"], "", http_post=post, gemini_models=["old-model", "new"])
        r.generate_json("s", "u")
        self.assertIn(("gemini", "old-model"), r.dead_models)
        r.blocked_until["gemini#1"] = 10 ** 12                       # force use of key 2
        r.generate_json("s", "u")
        self.assertNotIn(("k2", "old-model"), post.log)               # never tried on key 2

    def test_bad_key_skips_all_its_models(self):
        bad = Resp(400, text='{"error":{"message":"API key not valid. Please pass a valid API key."}}')
        post = ModelPost({("k1", "a"): [bad], ("k2", "a"): [gemini_ok({"ok": 1})]})
        r = LLMRouter(["k1", "k2"], "", http_post=post, gemini_models=["a", "b"])
        r.generate_json("s", "u")
        self.assertEqual(post.log, [("k1", "a"), ("k2", "a")])        # k1/b was never tried

    def test_groq_is_last_and_has_its_own_chain(self):
        post = ModelPost({("k1", "a"): [QUOTA()], ("gq", "g1"): [Resp(429, text="rate")], ("gq", "g2"): [groq_ok({"g": 2})]})
        r = LLMRouter(["k1"], "gq", http_post=post, gemini_models=["a"], groq_models=["g1", "g2"])
        self.assertEqual(r.generate_json("s", "u"), {"g": 2})
        self.assertEqual((r.last_provider, r.last_key, r.last_model), ("groq", "groq", "g2"))

    def test_attempt_cap(self):
        post = ModelPost({}, default=QUOTA())
        r = LLMRouter(["k1", "k2", "k3"], "", http_post=post, gemini_models=["a", "b", "c"], max_attempts=4)
        with self.assertRaises(LLMError) as ctx:
            r.generate_json("s", "u")
        self.assertEqual(len(post.log), 4)
        self.assertIn("stopped after 4 attempts", str(ctx.exception))

    def test_media_uses_the_same_chain(self):
        post = ModelPost({("k1", "a"): [QUOTA()], ("k1", "b"): [Resp(200, {"candidates": [{"content": {"parts": [{"text": "hello"}]}}]})]})
        r = LLMRouter(["k1"], "", http_post=post, gemini_models=["a", "b"])
        self.assertEqual(r.transcribe("image", b"x", "image/png", "p"), "hello")


class ExtrasTests(unittest.TestCase):
    def test_gemini3_payload_has_thinking_low_and_no_temperature(self):
        post = ModelPost({("k1", "gemini-3.5-flash"): [gemini_ok({})]})
        LLMRouter(["k1"], "", http_post=post, gemini_models=["gemini-3.5-flash"]).generate_json("s", "u")
        cfg = post.payloads[0]["generationConfig"]
        self.assertEqual(cfg["thinkingConfig"], {"thinkingLevel": "low"})
        self.assertNotIn("temperature", cfg)
        self.assertEqual(cfg["responseMimeType"], "application/json")

    def test_gemini25_payload_has_temperature_and_no_thinking_budget_for_lite(self):
        post = ModelPost({("k1", "gemini-2.5-flash"): [gemini_ok({})], ("k1", "gemini-2.5-flash-lite"): [gemini_ok({})]})
        LLMRouter(["k1"], "", http_post=post, gemini_models=["gemini-2.5-flash"]).generate_json("s", "u")
        LLMRouter(["k1"], "", http_post=post, gemini_models=["gemini-2.5-flash-lite"]).generate_json("s", "u")
        self.assertEqual(post.payloads[0]["generationConfig"]["thinkingConfig"], {"thinkingBudget": 0})
        self.assertEqual(post.payloads[0]["generationConfig"]["temperature"], 0)
        self.assertNotIn("thinkingConfig", post.payloads[1]["generationConfig"])

    def test_retries_without_optional_settings_when_rejected(self):
        bad = Resp(400, text='{"error":{"message":"Invalid JSON payload received. Unknown name \\"thinkingConfig\\""}}')
        post = ModelPost({("k1", "gemini-3.5-flash"): [bad, gemini_ok({"ok": 1}), gemini_ok({"ok": 2})]})
        r = LLMRouter(["k1"], "", http_post=post, gemini_models=["gemini-3.5-flash"])
        self.assertEqual(r.generate_json("s", "u"), {"ok": 1})
        self.assertIn("thinkingConfig", post.payloads[0]["generationConfig"])
        self.assertNotIn("thinkingConfig", post.payloads[1]["generationConfig"])
        r.generate_json("s", "u")                                      # remembered: no extras any more
        self.assertNotIn("thinkingConfig", post.payloads[2]["generationConfig"])
        self.assertEqual(len(post.log), 3)

    def test_groq_json_mode_rejected_then_retried_plain(self):
        bad = Resp(400, text='{"error":{"message":"response_format json_object is not supported with this model"}}')
        post = ModelPost({("gq", "g1"): [bad, groq_ok({"ok": 1})]})
        r = LLMRouter([], "gq", http_post=post, groq_models=["g1"])
        self.assertEqual(r.generate_json("s", "u"), {"ok": 1})
        self.assertIn("response_format", post.payloads[0])
        self.assertNotIn("response_format", post.payloads[1])

    def test_thought_parts_are_ignored(self):
        resp = Resp(200, {"candidates": [{"content": {"parts": [{"text": "thinking...", "thought": True}, {"text": '{"a": 1}'}]}}]})
        post = ModelPost({("k1", "a"): [resp]})
        self.assertEqual(LLMRouter(["k1"], "", http_post=post, gemini_models=["a"]).generate_json("s", "u"), {"a": 1})


class DiagnoseTests(unittest.TestCase):
    def test_diagnose_reports_each_key_and_model(self):
        post = ModelPost({("k1", "a"): [gemini_ok({"ok": True})], ("k1", "b"): [QUOTA()], ("k2", "a"): [gemini_ok({"ok": True})],
                          ("gq", "g1"): [groq_ok({"ok": True})]})
        r = LLMRouter(["k1", "k2"], "gq", http_post=post, gemini_models=["a", "b"], groq_models=["g1"])
        rep = r.diagnose()
        by = {x["slot"]: x for x in rep["results"]}
        self.assertTrue(by["gemini#1:a"]["ok"] and by["gemini#2:a"]["ok"] and by["groq:g1"]["ok"])
        self.assertFalse(by["gemini#1:b"]["ok"])
        self.assertIn("QuotaError", by["gemini#1:b"]["error"])
        self.assertNotIn("gemini#2:b", by)                             # other keys: only the model that worked
        self.assertTrue(rep["ready"])
        self.assertEqual(r.blocked_until, {})                          # a test must not put keys on cooldown

    def test_diagnose_without_keys(self):
        rep = LLMRouter([], "", http_post=ModelPost({})).diagnose()
        self.assertFalse(rep["ready"])
        self.assertEqual(rep["results"], [])


class ServerErrorTests(unittest.TestCase):
    def make(self, llm, local=False):
        svc = BotService(router=llm, retriever=retriever(), cache=Cache(":memory:"), limiter=RateLimiter(100))
        app = create_app(svc, twilio=None, meta=None, wa_url=None)
        return svc, TestClient(app, client=("127.0.0.1", 5000) if local else ("203.0.113.9", 5000))

    def test_llm_failure_is_json_with_friendly_message_and_hidden_detail(self):
        _, c = self.make(StubLLM(LLMError("All providers failed: gemini#1:m: quota / rate limit (per minute)")))
        r = c.post("/api/check", data={"text": "some claim"})
        self.assertEqual(r.status_code, 503)
        self.assertIn("busy", r.json()["error"])
        self.assertNotIn("detail", r.json())                           # strangers never see technical details

    def test_local_user_sees_technical_detail(self):
        _, c = self.make(StubLLM(LLMError("All providers failed: gemini#1:m: model not available")), local=True)
        r = c.post("/api/check", data={"text": "some claim"})
        self.assertIn("model not available", r.json()["detail"])

    def test_unexpected_crash_returns_json_not_a_html_page(self):
        _, c = self.make(StubLLM(ZeroDivisionError("boom")), local=True)
        r = c.post("/api/check", data={"text": "some claim"})
        self.assertEqual(r.status_code, 500)
        self.assertTrue(r.headers["content-type"].startswith("application/json"))
        self.assertIn("Something went wrong", r.json()["error"])
        self.assertIn("boom", r.json()["detail"])

    def test_debug_flag_shows_detail_to_everyone(self):
        from unittest import mock
        _, c = self.make(StubLLM(LLMError("secret-ish detail")))
        with mock.patch.object(config, "DEBUG_ERRORS", True):
            self.assertIn("detail", c.post("/api/check", data={"text": "x claim"}).json())

    def test_no_key_gives_setup_message_and_config_flag(self):
        router = LLMRouter([], "", http_post=ModelPost({}))
        _, c = self.make(router, local=True)
        self.assertFalse(c.get("/api/config").json()["llm_ready"])
        r = c.post("/api/check", data={"text": "some claim"})
        self.assertEqual(r.status_code, 503)
        self.assertIn("not fully set up", r.json()["error"])

    def test_diagnose_endpoint_is_local_only(self):
        post = ModelPost({("k1", "a"): [gemini_ok({"ok": True})]})
        router = LLMRouter(["k1"], "", http_post=post, gemini_models=["a"])
        _, remote = self.make(router, local=False)
        self.assertEqual(remote.get("/api/diagnose").status_code, 403)
        _, local = self.make(router, local=True)
        rep = local.get("/api/diagnose").json()
        self.assertTrue(rep["ready"])
        self.assertIn("search", rep)


if __name__ == "__main__":
    unittest.main()
