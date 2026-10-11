import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from forward_checker import config as _cfg
_cfg.GEMINI_MODELS = ["m"]        # one model per key: scripted test responses map one-to-one
_cfg.GROQ_MODELS = ["m"]
_cfg.GROQ_VISION_MODELS = ["m"]
_cfg.GROQ_WHISPER_MODELS = ["m"]
_cfg.LEGACY_GEMINI_MODELS = []

from forward_checker.evidence import Evidence, EvidenceRetriever, is_time_sensitive, is_trusted
from forward_checker.llm_router import LLMError, LLMRouter
from forward_checker.extractors import ExtractionError, detect_kind, extract_text
from forward_checker.pipeline import check_forward, format_card
from forward_checker.verdict import judge


class Resp:
    def __init__(self, code=200, body=None, text="", headers=None):
        self.status_code, self._body, self.text, self.headers = code, body, text, headers or {}

    def json(self):
        return self._body


def gemini_ok(obj):
    return Resp(200, {"candidates": [{"content": {"parts": [{"text": json.dumps(obj)}]}}]})


def groq_ok(obj):
    return Resp(200, {"choices": [{"message": {"content": json.dumps(obj)}}]})


class FakePost:
    """Routes by URL; each provider has a queue of responses."""
    def __init__(self, gemini, groq):
        self.q = {"generativelanguage": list(gemini), "groq": list(groq)}
        self.log = []

    def __call__(self, url, **kw):
        key = "generativelanguage" if "generativelanguage" in url else "groq"
        self.log.append(key)
        return self.q[key].pop(0)


class Clock:
    t = 1000.0
    def __call__(self):
        return self.t


class RouterTests(unittest.TestCase):
    def test_uses_gemini_first(self):
        post = FakePost([gemini_ok({"a": 1})], [])
        r = LLMRouter("g", "q", http_post=post)
        self.assertEqual(r.generate_json("s", "u"), {"a": 1})
        self.assertEqual(r.last_provider, "gemini")

    def test_falls_back_to_groq_on_quota_then_returns_after_cooldown(self):
        clock = Clock()
        quota = Resp(429, text="quota exceeded PerDay")
        post = FakePost([quota, gemini_ok({"back": True})], [groq_ok({"b": 2}), groq_ok({"b": 3})])
        r = LLMRouter("g", "q", http_post=post, now=clock)
        self.assertEqual(r.generate_json("s", "u"), {"b": 2})
        self.assertEqual(r.last_provider, "groq")
        self.assertEqual(r.generate_json("s", "u"), {"b": 3})  # gemini skipped (cooldown)
        self.assertEqual(post.log, ["generativelanguage", "groq", "groq"])
        clock.t += 4000  # daily cooldown over
        self.assertEqual(r.generate_json("s", "u"), {"back": True})
        self.assertEqual(r.last_provider, "gemini")

    def test_falls_back_on_server_error_and_bad_json(self):
        post = FakePost([Resp(500, text="boom")], [groq_ok({"ok": 1})])
        r = LLMRouter("g", "q", http_post=post)
        self.assertEqual(r.generate_json("s", "u"), {"ok": 1})

    def test_only_groq_key(self):
        post = FakePost([], [groq_ok({"x": 1})])
        r = LLMRouter("", "q", http_post=post)
        self.assertEqual(r.generate_json("s", "u"), {"x": 1})

    def test_no_keys_raises(self):
        with self.assertRaises(LLMError):
            LLMRouter("", "", http_post=FakePost([], [])).generate_json("s", "u")

    def test_both_fail_raises(self):
        post = FakePost([Resp(500, text="a")], [Resp(500, text="b")])
        with self.assertRaises(LLMError):
            LLMRouter("g", "q", http_post=post).generate_json("s", "u")

    def test_fenced_json_is_parsed(self):
        fenced = Resp(200, {"candidates": [{"content": {"parts": [{"text": "```json\n{\"k\": 1}\n```"}]}}]})
        r = LLMRouter("g", "", http_post=FakePost([fenced], []))
        self.assertEqual(r.generate_json("s", "u"), {"k": 1})


class FakePost2:
    """Records calls; per-provider queues keyed by a url hint."""
    def __init__(self, gemini, groq_chat=(), groq_audio=()):
        self.q = {"generativelanguage": list(gemini), "chat": list(groq_chat), "audio": list(groq_audio)}
        self.calls = []

    def __call__(self, url, **kw):
        key = ("generativelanguage" if "generativelanguage" in url
               else "audio" if "audio/transcriptions" in url else "chat")
        self.calls.append((key, kw))
        return self.q[key].pop(0)


def gemini_text(t):
    return Resp(200, {"candidates": [{"content": {"parts": [{"text": t}]}}]})


class MediaTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.dir = tempfile.mkdtemp()

    def _file(self, name, data=b"bytes"):
        p = os.path.join(self.dir, name)
        with open(p, "wb") as f:
            f.write(data)
        return p

    def test_detect_kind(self):
        self.assertEqual(detect_kind("a.PNG")[0], "image")
        self.assertEqual(detect_kind("v.opus"), ("audio", "audio/ogg"))
        self.assertEqual(detect_kind("m.txt")[0], "text")
        with self.assertRaises(ExtractionError):
            detect_kind("x.exe")

    def test_image_via_gemini(self):
        post = FakePost2([gemini_text("Lemon water cures cancer")])
        r = LLMRouter("g", "q", http_post=post)
        kind, text = extract_text(r, self._file("s.png"))
        self.assertEqual((kind, text), ("image", "Lemon water cures cancer"))
        sent = post.calls[0][1]["json"]["contents"][0]["parts"][0]
        self.assertEqual(sent["inline_data"]["mime_type"], "image/png")

    def test_image_falls_back_to_groq_vision(self):
        post = FakePost2([Resp(429, text="quota")], groq_chat=[groq_text("Hello text")])
        r = LLMRouter("g", "q", http_post=post)
        self.assertEqual(extract_text(r, self._file("s.jpg"))[1], "Hello text")
        self.assertEqual(r.last_provider, "groq")
        content = post.calls[1][1]["json"]["messages"][0]["content"]
        self.assertTrue(content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,"))

    def test_audio_falls_back_to_whisper(self):
        post = FakePost2([Resp(429, text="quota")], groq_audio=[Resp(200, {"text": "namaste yeh sach hai"})])
        r = LLMRouter("g", "q", http_post=post)
        kind, text = extract_text(r, self._file("v.ogg"))
        self.assertEqual((kind, text), ("audio", "namaste yeh sach hai"))
        self.assertEqual(post.calls[1][0], "audio")
        self.assertIn("file", post.calls[1][1]["files"])

    def test_text_file_needs_no_llm(self):
        kind, text = extract_text(LLMRouter("", "", http_post=FakePost2([])), self._file("m.txt", b" hi there "))
        self.assertEqual((kind, text), ("text", "hi there"))

    def test_missing_and_all_failed(self):
        r = LLMRouter("g", "q", http_post=FakePost2([Resp(500, text="a")], groq_chat=[Resp(500, text="b")]))
        with self.assertRaises(ExtractionError):
            extract_text(r, self._file("s.png"))
        with self.assertRaises(ExtractionError):
            extract_text(r, os.path.join(self.dir, "nope.png"))

    def test_empty_transcription_counts_as_failure(self):
        post = FakePost2([gemini_text("   ")], groq_chat=[groq_text("real text")])
        r = LLMRouter("g", "q", http_post=post)
        self.assertEqual(extract_text(r, self._file("s.png"))[1], "real text")


def groq_text(t):
    return Resp(200, {"choices": [{"message": {"content": t}}]})


class KeyPost:
    """Routes by the API key in the headers; a queue of responses per key."""
    def __init__(self, **queues):
        self.q = {k: list(v) for k, v in queues.items()}
        self.log = []

    def __call__(self, url, **kw):
        h = kw.get("headers", {})
        key = h.get("x-goog-api-key") or h.get("Authorization", "").replace("Bearer ", "")
        self.log.append(key)
        return self.q[key].pop(0)


class MultiKeyTests(unittest.TestCase):
    def test_keys_fall_back_in_order_then_groq(self):
        quota = Resp(429, text="quota")
        post = KeyPost(k1=[quota], k2=[quota], k3=[gemini_ok({"from": "k3"})])
        r = LLMRouter(["k1", "k2", "k3"], "gq", http_post=post)
        self.assertEqual(r.generate_json("s", "u"), {"from": "k3"})
        self.assertEqual(post.log, ["k1", "k2", "k3"])
        self.assertEqual((r.last_provider, r.last_key), ("gemini", "gemini#3"))

    def test_all_gemini_keys_exhausted_uses_groq_last(self):
        quota = Resp(429, text="quota")
        post = KeyPost(k1=[quota], k2=[quota], gq=[groq_ok({"from": "groq"})])
        r = LLMRouter(["k1", "k2"], "gq", http_post=post)
        self.assertEqual(r.generate_json("s", "u"), {"from": "groq"})
        self.assertEqual((r.last_provider, r.last_key), ("groq", "groq"))

    def test_blocked_keys_are_skipped_on_next_call(self):
        quota = Resp(429, text="quota")
        post = KeyPost(k1=[quota], k2=[gemini_ok({"n": 1}), gemini_ok({"n": 2})])
        r = LLMRouter(["k1", "k2"], "", http_post=post)
        r.generate_json("s", "u")
        r.generate_json("s", "u")
        self.assertEqual(post.log, ["k1", "k2", "k2"])  # k1 not retried during cooldown

    def test_invalid_key_is_blocked_for_a_long_time(self):
        clock = Clock()
        bad = Resp(400, text='{"error":{"message":"API key not valid"}}')
        post = KeyPost(k1=[bad], k2=[gemini_ok({"n": 1})])
        r = LLMRouter(["k1", "k2"], "", http_post=post, now=clock)
        r.generate_json("s", "u")
        clock.t += 4000  # well past a normal cooldown
        self.assertNotIn("gemini#1", [s.pid for s in r._available("json")])

    def test_works_for_media_too(self):
        quota = Resp(429, text="quota")
        post = KeyPost(k1=[quota], k2=[gemini_text("hello from k2")])
        r = LLMRouter(["k1", "k2"], "", http_post=post)
        self.assertEqual(r.transcribe("image", b"x", "image/png", "p"), "hello from k2")

    def test_event_log_explains_skipped_keys(self):
        post = KeyPost(k1=[Resp(429, text="PerDay quota")], k2=[Resp(400, text="API key not valid")],
                       k3=[gemini_ok({"ok": 1})])
        r = LLMRouter(["k1", "k2", "k3"], "", http_post=post)
        r.generate_json("s", "u")
        self.assertEqual([e[0] for e in r.events], ["gemini#1:m", "gemini#2:m", "gemini#3:m"])
        self.assertIn("daily", r.events[0][1])
        self.assertIn("rejected", r.events[1][1])
        self.assertEqual(r.events[2][1], "ok")

    def test_reason_whitespace_is_cleaned(self):
        llm = FakeLLM({"label": "False", "reason": "No proof.\r\n  At all.\n", "source_ids": [1]})
        self.assertEqual(judge(llm, "claim", EV)["reason"], "No proof. At all.")

    def test_config_parses_comma_separated_keys(self):
        from forward_checker import config
        self.assertEqual(config._split_keys("a, b;c\nd", "a", "e"), ["a", "b", "c", "d", "e"])


class FakeLLM:
    def __init__(self, *answers):
        self.answers, self.last_provider = list(answers), "fake"

    def generate_json(self, system, user):
        return self.answers.pop(0)


EV = [Evidence("WHO myths", "https://www.who.int/x", "No evidence lemon water cures cancer.")]


class VerdictTests(unittest.TestCase):
    def test_no_evidence_forces_cannot_confirm(self):
        v = judge(FakeLLM(), "claim", [])
        self.assertEqual(v["label"], "Cannot be confirmed")

    def test_valid_false_verdict(self):
        llm = FakeLLM({"label": "False", "reason": "No proof.", "source_ids": [1]})
        v = judge(llm, "claim", EV)
        self.assertEqual(v["label"], "False")
        self.assertEqual(v["sources"][0]["url"], "https://www.who.int/x")

    def test_label_without_valid_source_is_downgraded(self):
        llm = FakeLLM({"label": "Verified", "reason": "ok", "source_ids": [7]})
        self.assertEqual(judge(llm, "claim", EV)["label"], "Cannot be confirmed")

    def test_unknown_label_is_downgraded(self):
        llm = FakeLLM({"label": "Totally true", "reason": "ok", "source_ids": [1]})
        self.assertEqual(judge(llm, "claim", EV)["label"], "Cannot be confirmed")


class EvidenceTests(unittest.TestCase):
    def test_trust_filter(self):
        self.assertTrue(is_trusted("https://www.who.int/news"))
        self.assertTrue(is_trusted("https://pib.gov.in/x"))
        self.assertTrue(is_trusted("https://something.nic.in/x"))
        self.assertFalse(is_trusted("https://random-blog.com/x"))
        self.assertFalse(is_trusted("https://who.int.evil.com/x"))
        self.assertFalse(is_trusted("https://notwho.int/x"))

    def test_untrusted_results_are_dropped_and_deduped(self):
        def fake(q):
            return [Evidence("a", "https://blog.com/1", "s"), Evidence("b", "https://www.who.int/x", "s"),
                    Evidence("b2", "https://www.who.int/x", "s")]
        out = EvidenceRetriever(search_fn=fake).search("claim")
        self.assertEqual([e.url for e in out], ["https://www.who.int/x"])

    def test_search_failure_returns_empty(self):
        def boom(q):
            raise RuntimeError("down")
        self.assertEqual(EvidenceRetriever(search_fn=boom).search("claim"), [])


class TimeTests(unittest.TestCase):
    def test_time_sensitive_detection(self):
        self.assertTrue(is_time_sensitive("Last date to link Aadhaar with PAN is 31 March 2023"))
        self.assertTrue(is_time_sensitive("Scheme starts from tomorrow"))
        self.assertFalse(is_time_sensitive("Hot lemon water cures cancer"))

    def test_extra_latest_update_query_for_time_claims(self):
        seen = []
        def fake(q):
            seen.append(q)
            return []
        EvidenceRetriever(search_fn=fake).search("Last date to link Aadhaar with PAN is 31 March 2023")
        self.assertEqual(len(seen), 4)
        self.assertIn("latest update", seen[2])
        self.assertIn("site:pib.gov.in", seen[3])

    def test_prompt_contains_today_and_time_rules(self):
        captured = {}
        class Spy:
            def generate_json(self, system, user):
                captured["s"], captured["u"] = system, user
                return {"label": "Outdated", "reason": "Date passed.", "source_ids": [1]}
        judge(Spy(), "deadline is 31 March 2023", EV)
        self.assertIn("TODAY:", captured["u"])
        self.assertIn("TIME RULES", captured["s"])


class PipelineTests(unittest.TestCase):
    def test_end_to_end(self):
        llm = FakeLLM(
            {"language": "en",
             "claims": [{"text": "Hot lemon water cures cancer", "topic": "health"}],
             "manipulation_flags": ["chain_forward_request", "not_a_flag"]},
            {"label": "False", "reason": "No medical proof.", "source_ids": [1]},
        )
        res = check_forward("Drink lemon water cures cancer, share with everyone!", llm=llm,
                            retriever=EvidenceRetriever(search_fn=lambda q: EV))
        self.assertEqual(res["overall"], "False")
        self.assertEqual(res["manipulation_flags"], ["chain_forward_request"])
        card = format_card(res)
        self.assertIn("❌ *False*", card)
        self.assertIn("who.int", card)
        self.assertIn("forward it to many people", card)

    def test_no_claims(self):
        llm = FakeLLM({"language": "en", "claims": [], "manipulation_flags": []})
        res = check_forward("Good morning!", llm=llm, retriever=EvidenceRetriever(search_fn=lambda q: []))
        self.assertIn("could not find a factual claim", format_card(res))


if __name__ == "__main__":
    unittest.main()
