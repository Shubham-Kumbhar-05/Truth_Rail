import base64
import hashlib
import hmac
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

os.environ["FC_NO_AUTOSTART"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from forward_checker import config as _cfg
_cfg.GEMINI_MODELS = ["m"]        # one model per key: scripted test responses map one-to-one
_cfg.GROQ_MODELS = ["m"]
_cfg.GROQ_VISION_MODELS = ["m"]
_cfg.GROQ_WHISPER_MODELS = ["m"]
_cfg.LEGACY_GEMINI_MODELS = []

from fastapi.testclient import TestClient

from forward_checker import config, i18n
from forward_checker.cache import Cache
from forward_checker.evidence import Evidence, EvidenceRetriever
from forward_checker.extractors import ExtractionError, kind_from_mime, pdf_to_text
from forward_checker.llm_router import LLMError
from forward_checker.messaging import Inbound, Media, MetaAdapter, MessagingError, TwilioAdapter, chunk
from forward_checker.pipeline import cache_ttl, check_forward, format_card
from forward_checker.service import BotService, RateLimiter
from forward_checker.voice import VoiceStore
from forward_checker import webfetch
from server import create_app

EV = [Evidence("WHO myths", "https://www.who.int/x", "No evidence lemon water cures cancer.")]
SPLIT = {"language": "en", "claims": [{"text": "Hot lemon water cures cancer", "topic": "health"}],
         "manipulation_flags": ["chain_forward_request"]}
VERDICT = {"label": "False", "reason": "No medical proof.", "source_ids": [1]}


class StubLLM:
    def __init__(self, *answers, transcript="Hot lemon water cures cancer. Share it!"):
        self.answers, self.transcript, self.calls = list(answers), transcript, 0
        self.last_provider, self.last_key = "gemini", "gemini#1"

    def generate_json(self, system, user):
        self.calls += 1
        a = self.answers.pop(0)
        if isinstance(a, Exception):
            raise a
        return a

    def transcribe(self, kind, data, mime, prompt, filename="f"):
        return self.transcript


class FakeResp:
    def __init__(self, code=200, body=b"", headers=None, text="", js=None):
        self.status_code, self._body, self.headers, self.text, self._js = code, body, headers or {}, text, js

    def iter_content(self, n):
        for i in range(0, len(self._body), n):
            yield self._body[i:i + n]

    def json(self):
        return self._js


class FakeHttp:
    def __init__(self, get_resp=None):
        self.posts, self.gets, self.get_resp = [], [], get_resp

    def post(self, url, **kw):
        self.posts.append((url, kw))
        return FakeResp(201)

    def get(self, url, **kw):
        self.gets.append((url, kw))
        return self.get_resp


class FakeMessenger:
    def __init__(self, media_bytes=b"bytes"):
        self.texts, self.audios, self.media_bytes = [], [], media_bytes

    def send_text(self, to, body):
        self.texts.append(body)

    def send_audio(self, to, url):
        self.audios.append(url)

    def download(self, media):
        return self.media_bytes


def make_pdf(text):
    content = f"BT /F1 12 Tf 50 700 Td ({text}) Tj ET"
    objs = ["<< /Type /Catalog /Pages 2 0 R >>", "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
            f"<< /Length {len(content)} >>\nstream\n{content}\nendstream",
            "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    out, offs = b"%PDF-1.4\n", []
    for i, o in enumerate(objs, 1):
        offs.append(len(out))
        out += f"{i} 0 obj\n{o}\nendobj\n".encode()
    x = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    for o in offs:
        out += f"{o:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{x}\n%%EOF".encode()
    return out


def write_file(path, data=b"mp3"):
    with open(path, "wb") as f:
        f.write(data)


def retriever():
    return EvidenceRetriever(search_fn=lambda q: EV)


class CacheTests(unittest.TestCase):
    def test_put_get_and_normalisation(self):
        c = Cache(":memory:")
        c.put("Hello   World", {"a": 1}, 100)
        self.assertEqual(c.get("hello world"), {"a": 1})
        self.assertIsNone(c.get("something else"))

    def test_expiry(self):
        t = [1000.0]
        c = Cache(":memory:", now=lambda: t[0])
        c.put("x", {"a": 1}, 60)
        t[0] += 61
        self.assertIsNone(c.get("x"))

    def test_zero_ttl_not_stored_and_stats(self):
        c = Cache(":memory:")
        c.put("x", {"a": 1}, 0)
        self.assertIsNone(c.get("x"))
        c.log("whatsapp:+911234567890", "text", "False", False, "gemini#1")
        c.log("whatsapp:+911234567890", "text", "False", True, "cache")
        self.assertEqual(c.stats(), {"checks": 2, "served_from_cache": 1, "by_label": {"False": 2},
                                     "feedback": {"up": 0, "down": 0}})

    def test_second_check_uses_cache_not_llm(self):
        c = Cache(":memory:")
        llm = StubLLM(SPLIT, VERDICT)
        r1 = check_forward("Lemon cures cancer", llm=llm, retriever=retriever(), cache=c)
        r2 = check_forward("  lemon   cures cancer ", llm=llm, retriever=retriever(), cache=c)
        self.assertFalse(r1["cached"])
        self.assertTrue(r2["cached"])
        self.assertEqual(llm.calls, 2)

    def test_ttl_rules(self):
        base = {"claims": [{"claim": "Lemon water cures cancer"}], "overall": "False"}
        self.assertEqual(cache_ttl(base), config.CACHE_TTL_HOURS * 3600)
        timed = {"claims": [{"claim": "Last date is 31 March 2023"}], "overall": "Outdated"}
        self.assertEqual(cache_ttl(timed), config.CACHE_TTL_TIME_SENSITIVE_HOURS * 3600)
        unk = {"claims": [{"claim": "Some rumour"}], "overall": "Cannot be confirmed"}
        self.assertEqual(cache_ttl(unk), config.CACHE_TTL_UNCONFIRMED_HOURS * 3600)


class WebFetchTests(unittest.TestCase):
    pub = staticmethod(lambda host: ["93.184.216.34"])

    def test_single_url_detection(self):
        self.assertEqual(webfetch.find_single_url("https://a.com/x"), "https://a.com/x")
        self.assertEqual(webfetch.find_single_url("see https://a.com/x"), "https://a.com/x")
        self.assertIsNone(webfetch.find_single_url("Lemon cures cancer, read https://a.com/x and share it widely"))
        self.assertIsNone(webfetch.find_single_url("no link here"))

    def test_private_addresses_blocked(self):
        for ip in ("127.0.0.1", "10.0.0.5", "192.168.1.1", "169.254.169.254", "::1"):
            self.assertFalse(webfetch.is_public_host("h", lambda h, ip=ip: [ip]), ip)
        self.assertTrue(webfetch.is_public_host("h", self.pub))
        self.assertFalse(webfetch.is_public_host("h", lambda h: (_ for _ in ()).throw(OSError())))

    def test_bad_scheme_and_port(self):
        with self.assertRaises(webfetch.WebFetchError):
            webfetch.fetch("file:///etc/passwd", resolver=self.pub)
        with self.assertRaises(webfetch.WebFetchError):
            webfetch.fetch("http://example.com:8080/", resolver=self.pub)

    def test_redirect_to_private_is_blocked(self):
        def resolver(host):
            return ["127.0.0.1"] if host == "internal.local" else ["93.184.216.34"]
        getter = lambda url, **kw: FakeResp(302, headers={"Location": "http://internal.local/secret"})
        with self.assertRaises(webfetch.WebFetchError):
            webfetch.fetch("https://example.com/", getter=getter, resolver=resolver)

    def test_html_page_text(self):
        html = b"<html><body><article><p>" + b"The government has announced a new scheme for farmers today. " * 3 + b"</p></article></body></html>"
        getter = lambda url, **kw: FakeResp(200, html, {"Content-Type": "text/html; charset=utf-8"})
        text = webfetch.fetch_url_text("https://example.com/a", getter=getter, resolver=self.pub)
        self.assertIn("new scheme for farmers", text)

    def test_too_large_and_wrong_type(self):
        big = FakeResp(200, b"x" * (config.MAX_URL_BYTES + 10), {"Content-Type": "text/html"})
        with self.assertRaises(webfetch.WebFetchError):
            webfetch.fetch("https://example.com/", getter=lambda u, **kw: big, resolver=self.pub)
        exe = FakeResp(200, b"MZ", {"Content-Type": "application/x-msdownload"})
        with self.assertRaises(webfetch.WebFetchError):
            webfetch.fetch_url_text("https://example.com/", getter=lambda u, **kw: exe, resolver=self.pub)


class PdfAndMimeTests(unittest.TestCase):
    def test_pdf_text(self):
        text = pdf_to_text(make_pdf("Government gives free laptops to all students this year"))
        self.assertIn("free laptops", text)

    def test_blank_or_broken_pdf(self):
        with self.assertRaises(ExtractionError):
            pdf_to_text(make_pdf("hi"))
        with self.assertRaises(ExtractionError):
            pdf_to_text(b"not a pdf at all")

    def test_kind_from_mime(self):
        self.assertEqual(kind_from_mime("audio/ogg; codecs=opus"), ("audio", "audio/ogg"))
        self.assertEqual(kind_from_mime("image/jpeg")[0], "image")
        self.assertEqual(kind_from_mime("application/pdf")[0], "pdf")
        with self.assertRaises(ExtractionError):
            kind_from_mime("video/mp4")


class I18nVoiceTests(unittest.TestCase):
    def test_labels_and_script(self):
        res = {"language": "hi", "claims": [{"label": "False", "reason": "कोई सबूत नहीं।"}],
               "manipulation_flags": ["fear_appeal"]}
        script = i18n.spoken_script(res)
        self.assertTrue(script.startswith("झूठ."))
        self.assertIn("सावधान", script)
        res["language"] = "xx"   # unknown language falls back to English
        self.assertTrue(i18n.spoken_script(res).startswith("False."))

    def test_card_has_local_label(self):
        res = {"language": "mr", "overall": "False", "manipulation_flags": [],
               "claims": [{"claim": "c", "label": "False", "reason": "r", "sources": []}]}
        self.assertIn("(खोटे)", format_card(res))

    def test_voice_store(self):
        d = tempfile.mkdtemp()
        written = {}
        def tts(text, lang, path):
            write_file(path)
            written["lang"] = lang
        v = VoiceStore(d, tts=tts)
        name = v.synthesize("hello", "hi")
        self.assertTrue(v.path_for(name))
        self.assertEqual(written["lang"], "hi")
        self.assertIsNone(v.synthesize("hello", "fr"))          # unsupported language
        self.assertIsNone(v.path_for("../../etc/passwd"))        # path traversal
        self.assertIsNone(VoiceStore(d, tts=lambda *a: 1 / 0).synthesize("hi", "en"))  # tts failure

    def test_voice_cleanup(self):
        d = tempfile.mkdtemp()
        v = VoiceStore(d, tts=lambda t, l, p: write_file(p, b"x"), now=lambda: 10 ** 12)
        name = v.synthesize("a", "en")      # file is "old" compared with the fake clock
        v.cleanup()
        self.assertIsNone(v.path_for(name))


class MessagingTests(unittest.TestCase):
    def test_chunk(self):
        parts = chunk("line\n" * 100, 50)
        self.assertTrue(all(len(p) <= 50 for p in parts))
        self.assertEqual("".join(parts).replace("\n", ""), "line" * 100)
        self.assertEqual(len(chunk("x" * 120, 50)), 3)

    def test_twilio_parse_and_signature(self):
        t = TwilioAdapter("sid", "tok", "whatsapp:+1415", http=FakeHttp())
        form = {"From": "whatsapp:+9199", "Body": "hi", "NumMedia": "1", "MediaUrl0": "https://m/1",
                "MediaContentType0": "audio/ogg", "MessageSid": "SM1"}
        m = t.parse(form)
        self.assertEqual((m.sender, m.text, m.media.mime, m.msg_id), ("whatsapp:+9199", "hi", "audio/ogg", "SM1"))
        self.assertIsNone(t.parse({}))
        url = "https://x.test/whatsapp/twilio"
        base = url + "".join(k + form[k] for k in sorted(form))
        sig = base64.b64encode(hmac.new(b"tok", base.encode(), hashlib.sha1).digest()).decode()
        self.assertTrue(t.validate(url, form, sig))
        self.assertFalse(t.validate(url, form, "bad"))
        self.assertFalse(t.validate(url + "/other", form, sig))

    def test_twilio_send_and_download(self):
        http = FakeHttp(FakeResp(200, b"abc"))
        t = TwilioAdapter("sid", "tok", "whatsapp:+1415", http=http)
        t.send_text("whatsapp:+9199", "hello")
        t.send_audio("whatsapp:+9199", "https://x.test/media/a.mp3")
        self.assertIn("Accounts/sid/Messages.json", http.posts[0][0])
        self.assertEqual(http.posts[0][1]["data"]["Body"], "hello")
        self.assertEqual(http.posts[1][1]["data"]["MediaUrl"], "https://x.test/media/a.mp3")
        self.assertEqual(t.download(Media(url="https://m/1", mime="audio/ogg")), b"abc")
        self.assertEqual(http.gets[0][1]["auth"], ("sid", "tok"))

    def test_twilio_send_error(self):
        class Bad(FakeHttp):
            def post(self, url, **kw):
                return FakeResp(400, text="nope")
        with self.assertRaises(MessagingError):
            TwilioAdapter("s", "t", "f", http=Bad()).send_text("to", "x")

    def test_meta_verify_signature_parse(self):
        m = MetaAdapter("tok", "123", "verify-me", "secret", http=FakeHttp())
        self.assertEqual(m.verify_challenge({"hub.mode": "subscribe", "hub.verify_token": "verify-me", "hub.challenge": "42"}), "42")
        self.assertIsNone(m.verify_challenge({"hub.mode": "subscribe", "hub.verify_token": "wrong", "hub.challenge": "42"}))
        raw = b'{"a":1}'
        good = "sha256=" + hmac.new(b"secret", raw, hashlib.sha256).hexdigest()
        self.assertTrue(m.validate(raw, good))
        self.assertFalse(m.validate(raw, "sha256=bad"))
        self.assertFalse(MetaAdapter("t", "1", "v", "", http=FakeHttp()).validate(raw, good))  # no secret = reject
        payload = {"entry": [{"changes": [{"value": {"messages": [
            {"from": "9199", "id": "w1", "type": "text", "text": {"body": "hello"}},
            {"from": "9199", "id": "w2", "type": "audio", "audio": {"id": "M1", "mime_type": "audio/ogg; codecs=opus"}},
            {"from": "9199", "id": "w3", "type": "image", "image": {"id": "M2", "mime_type": "image/jpeg", "caption": "see"}},
            {"from": "9199", "id": "w4", "type": "sticker"}]}}]}]}
        msgs = m.parse(payload)
        self.assertEqual([x.msg_id for x in msgs], ["w1", "w2", "w3", "w4"])
        self.assertEqual(msgs[1].media.media_id, "M1")
        self.assertEqual(msgs[2].text, "see")

    def test_meta_download_two_steps_and_send(self):
        class H(FakeHttp):
            def get(self, url, **kw):
                self.gets.append(url)
                return FakeResp(200, b"file", js={"url": "https://cdn/x"}) if "cdn" not in url else FakeResp(200, b"file")
        h = H()
        m = MetaAdapter("tok", "123", "v", "s", http=h)
        self.assertEqual(m.download(Media(media_id="M1")), b"file")
        self.assertEqual(h.gets, ["https://graph.facebook.com/v21.0/M1", "https://cdn/x"])
        m.send_text("9199", "hi")
        m.send_audio("9199", "https://x/a.mp3")
        self.assertEqual(h.posts[0][1]["json"]["text"]["body"], "hi")
        self.assertEqual(h.posts[1][1]["json"]["audio"]["link"], "https://x/a.mp3")


class ServiceTests(unittest.TestCase):
    def make(self, *answers, voice=True, limit=20, **kw):
        d = tempfile.mkdtemp()
        v = VoiceStore(d, tts=lambda t, l, p: write_file(p)) if voice else None
        return BotService(router=StubLLM(*answers, **kw), retriever=retriever(), cache=Cache(":memory:"),
                          voice=v, limiter=RateLimiter(limit), public_base_url="https://x.test")

    def test_greeting_and_empty_get_help_without_llm(self):
        svc, m = self.make(), FakeMessenger()
        svc.handle(Inbound("u", "Hi!"), m)
        svc.handle(Inbound("u", ""), m)
        self.assertEqual(m.texts, [i18n.HELP, i18n.HELP])
        self.assertEqual(svc.router.calls, 0)

    def test_text_flow(self):
        svc, m = self.make(SPLIT, VERDICT), FakeMessenger()
        svc.handle(Inbound("u", "Lemon water cures cancer!"), m)
        self.assertEqual(m.texts[0], i18n.ACK)
        self.assertIn("❌ *False*", m.texts[1])
        self.assertEqual(m.audios, [])                       # English text: no voice

    def test_voice_note_gets_voice_reply(self):
        svc, m = self.make(SPLIT, VERDICT), FakeMessenger()
        svc.handle(Inbound("u", "", Media(mime="audio/ogg")), m)
        self.assertIn("❌ *False*", m.texts[1])
        self.assertEqual(len(m.audios), 1)
        self.assertRegex(m.audios[0], r"^https://x\.test/media/[a-f0-9]{32}\.mp3$")

    def test_hindi_text_gets_voice_reply(self):
        hi = dict(SPLIT, language="hi")
        svc, m = self.make(hi, VERDICT), FakeMessenger()
        svc.handle(Inbound("u", "नींबू पानी से कैंसर ठीक होता है"), m)
        self.assertEqual(len(m.audios), 1)

    def test_voice_off_when_never(self):
        svc, m = self.make(SPLIT, VERDICT), FakeMessenger()
        with mock.patch.object(config, "VOICE_REPLY_MODE", "never"):
            svc.handle(Inbound("u", "", Media(mime="audio/ogg")), m)
        self.assertEqual(m.audios, [])

    def test_pdf_flow(self):
        svc, m = self.make(SPLIT, VERDICT), FakeMessenger(make_pdf("Government gives free laptops to all students this year"))
        svc.handle(Inbound("u", "", Media(mime="application/pdf")), m)
        self.assertIn("False", m.texts[1])

    def test_link_flow(self):
        svc, m = self.make(SPLIT, VERDICT), FakeMessenger()
        with mock.patch("forward_checker.service.resolve_link", return_value="Page says lemon cures cancer"):
            svc.handle(Inbound("u", "https://example.com/a"), m)
        self.assertIn("❌ *False*", m.texts[1])

    def test_rate_limit(self):
        svc, m = self.make(SPLIT, VERDICT, limit=1), FakeMessenger()
        svc.handle(Inbound("u", "claim one"), m)
        svc.handle(Inbound("u", "claim two"), m)
        self.assertEqual(m.texts[-1], i18n.RATE_LIMITED)

    def test_errors_become_friendly_messages(self):
        svc, m = self.make(LLMError("quota")), FakeMessenger()
        svc.handle(Inbound("u", "some claim"), m)
        self.assertEqual(m.texts[-1], i18n.BUSY)
        svc2, m2 = self.make(), FakeMessenger()
        svc2.handle(Inbound("u", "", Media(mime="video/mp4")), m2)
        self.assertIn("text, screenshots", m2.texts[-1])
        svc3, m3 = self.make(RuntimeError("boom")), FakeMessenger()
        svc3.handle(Inbound("u", "some claim"), m3)           # unexpected error must not crash
        self.assertEqual(m3.texts[-1], i18n.BUSY)

    def test_unsupported_whatsapp_type(self):
        svc, m = self.make(), FakeMessenger()
        svc.handle(Inbound("u", "\u2063unsupported"), m)
        self.assertIn("text, screenshots", m.texts[0])

    def test_repeat_is_served_from_cache(self):
        svc, m = self.make(SPLIT, VERDICT), FakeMessenger()
        svc.handle(Inbound("u", "Lemon water cures cancer!"), m)
        svc.handle(Inbound("u", "lemon water cures cancer!"), m)
        self.assertEqual(svc.router.calls, 2)
        self.assertEqual(svc.cache.stats()["served_from_cache"], 1)


class ServerTests(unittest.TestCase):
    def setUp(self):
        d = tempfile.mkdtemp()
        self.voice = VoiceStore(d, tts=lambda t, l, p: write_file(p))
        self.svc = BotService(router=StubLLM(SPLIT, VERDICT, SPLIT, VERDICT, SPLIT, VERDICT),
                              retriever=retriever(), cache=Cache(":memory:"), voice=self.voice,
                              limiter=RateLimiter(50), public_base_url="https://x.test")
        self.http = FakeHttp(FakeResp(200, b"abc"))
        self.tw = TwilioAdapter("sid", "tok", "whatsapp:+1415", http=self.http)
        self.meta = MetaAdapter("tok", "123", "verify-me", "secret", http=FakeHttp())
        self.app = create_app(self.svc, twilio=self.tw, meta=self.meta, validate_twilio=False)
        self.c = TestClient(self.app)

    def test_pages_and_health(self):
        self.assertIn("Truth_Rail", self.c.get("/").text)
        h = self.c.get("/health").json()
        self.assertEqual(h["status"], "ok")
        self.assertTrue(h["whatsapp"]["twilio"] and h["whatsapp"]["meta"])

    def test_api_check_text_and_file(self):
        r = self.c.post("/api/check", data={"text": "Lemon cures cancer"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["result"]["overall"], "False")
        r = self.c.post("/api/check", files={"file": ("voice.ogg", b"audio", "audio/ogg")})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["kind"], "audio")

    def test_api_check_errors(self):
        self.assertEqual(self.c.post("/api/check", data={"text": ""}).status_code, 400)
        r = self.c.post("/api/check", files={"file": ("x.exe", b"MZ", "application/octet-stream")})
        self.assertEqual(r.status_code, 400)
        self.svc.limiter = RateLimiter(0)
        self.assertEqual(self.c.post("/api/check", data={"text": "claim"}).status_code, 429)

    def test_twilio_webhook_end_to_end(self):
        form = {"From": "whatsapp:+9199", "Body": "Lemon water cures cancer!", "MessageSid": "SM1", "NumMedia": "0"}
        r = self.c.post("/whatsapp/twilio", data=form)
        self.assertEqual(r.status_code, 200)
        self.assertIn("<Response>", r.text)
        bodies = [kw["data"].get("Body", "") for _, kw in self.http.posts]
        self.assertTrue(any("❌ *False*" in b for b in bodies))
        n = len(self.http.posts)
        self.c.post("/whatsapp/twilio", data=form)             # provider retry with same id
        self.assertEqual(len(self.http.posts), n)

    def test_twilio_signature_enforced(self):
        app = create_app(self.svc, twilio=self.tw, meta=None, validate_twilio=True)
        c = TestClient(app)
        form = {"From": "whatsapp:+9199", "Body": "hi", "MessageSid": "SM9"}
        self.assertEqual(c.post("/whatsapp/twilio", data=form).status_code, 403)
        url = "http://testserver/whatsapp/twilio"
        base = url + "".join(k + form[k] for k in sorted(form))
        sig = base64.b64encode(hmac.new(b"tok", base.encode(), hashlib.sha1).digest()).decode()
        self.assertEqual(c.post("/whatsapp/twilio", data=form, headers={"X-Twilio-Signature": sig}).status_code, 200)

    def test_meta_webhook(self):
        r = self.c.get("/whatsapp/meta", params={"hub.mode": "subscribe", "hub.verify_token": "verify-me", "hub.challenge": "777"})
        self.assertEqual(r.text, "777")
        self.assertEqual(self.c.get("/whatsapp/meta", params={"hub.mode": "subscribe", "hub.verify_token": "no"}).status_code, 403)
        body = json.dumps({"entry": [{"changes": [{"value": {"messages": [
            {"from": "9199", "id": "w1", "type": "text", "text": {"body": "Lemon water cures cancer!"}}]}}]}]}).encode()
        sig = "sha256=" + hmac.new(b"secret", body, hashlib.sha256).hexdigest()
        self.assertEqual(self.c.post("/whatsapp/meta", content=body, headers={"X-Hub-Signature-256": "sha256=bad"}).status_code, 403)
        r = self.c.post("/whatsapp/meta", content=body, headers={"X-Hub-Signature-256": sig})
        self.assertEqual(r.status_code, 200)
        sent = [kw["json"] for _, kw in self.meta.http.posts]
        self.assertTrue(any("False" in s.get("text", {}).get("body", "") for s in sent))

    def test_media_route(self):
        name = self.voice.synthesize("hello", "en")
        r = self.c.get(f"/media/{name}")
        self.assertEqual((r.status_code, r.headers["content-type"]), (200, "audio/mpeg"))
        self.assertEqual(self.c.get("/media/notafile.mp3").status_code, 404)

    def test_unconfigured_providers_return_404(self):
        c = TestClient(create_app(self.svc, twilio=None, meta=None))
        self.assertEqual(c.post("/whatsapp/twilio", data={"From": "x"}).status_code, 404)
        self.assertEqual(c.get("/whatsapp/meta").status_code, 404)


if __name__ == "__main__":
    unittest.main()
