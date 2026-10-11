"""Truth_Rail web server: WhatsApp webhooks (Twilio + Meta), a browser demo page, and voice-file hosting.

Run locally:   uvicorn server:app --port 8000
"""
import io
import json
import logging
import os
import re
from collections import OrderedDict
from typing import Literal, Optional
from urllib.parse import quote

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from forward_checker import config, i18n
from forward_checker.cache import Cache
from forward_checker.evidence import EvidenceRetriever
from forward_checker.extractors import ExtractionError, detect_kind
from forward_checker.llm_router import LLMError
from forward_checker.messaging import MessagingError, Media, MetaAdapter, TwilioAdapter
from forward_checker.pipeline import format_card
from forward_checker.service import BotService, RateLimiter
from forward_checker.voice import VoiceStore

log = logging.getLogger("truth_rail")
HERE = os.path.dirname(os.path.abspath(__file__))
LOCAL_HOSTS = ("127.0.0.1", "::1", "localhost")
EMPTY_TWIML = '<?xml version="1.0" encoding="UTF-8"?><Response></Response>'
CSP = ("default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; "
       "img-src 'self' data: blob:; media-src 'self' blob:; connect-src 'self'; "
       "frame-ancestors 'none'; base-uri 'none'")


class Feedback(BaseModel):
    rating: Literal["up", "down"]
    label: Optional[str] = None
    kind: Optional[str] = None


def whatsapp_url():
    number = re.sub(r"\D", "", config.WHATSAPP_NUMBER)
    if not number:
        return None
    return f"https://wa.me/{number}?text={quote(config.WHATSAPP_JOIN_TEXT or 'hi')}"


def create_app(service=None, twilio="auto", meta="auto", validate_twilio=None, wa_url="auto"):
    if service is None:
        service = BotService(cache=Cache(), voice=VoiceStore())
    if twilio == "auto":
        twilio = TwilioAdapter() if (config.TWILIO_ACCOUNT_SID and config.TWILIO_AUTH_TOKEN
                                     and config.TWILIO_WHATSAPP_FROM) else None
    if meta == "auto":
        meta = MetaAdapter() if (config.META_ACCESS_TOKEN and config.META_PHONE_NUMBER_ID) else None
    if validate_twilio is None:
        validate_twilio = config.TWILIO_VALIDATE_SIGNATURE
    if wa_url == "auto":
        wa_url = whatsapp_url()
    feedback_limiter = RateLimiter(60)

    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
        for noisy in ("httpx", "httpcore", "urllib3"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
    if not getattr(service.router, "keys", True):
        log.warning("No GEMINI_API_KEYS / GROQ_API_KEY set. Checks will fail until you add keys to .env")

    def allow_detail(request):
        """Technical error details are shown only on your own computer (or when DEBUG_ERRORS=true)."""
        return config.DEBUG_ERRORS or (request.client is not None and request.client.host in LOCAL_HOSTS)

    def error_body(message, exc, request):
        body = {"error": message}
        if exc is not None and allow_detail(request):
            body["detail"] = str(exc)[:700]
        return body
    app = FastAPI(title="Truth_Rail", docs_url=None, redoc_url=None, openapi_url=None)

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        resp = await call_next(request)
        resp.headers.setdefault("X-Content-Type-Options", "nosniff")
        resp.headers.setdefault("Referrer-Policy", "no-referrer")
        resp.headers.setdefault("Content-Security-Policy", CSP)
        return resp

    app.mount("/static", StaticFiles(directory=os.path.join(HERE, "web", "static")), name="static")
    seen = OrderedDict()  # message ids already handled (providers retry deliveries)

    def first_time(msg_id):
        if not msg_id:
            return True
        if msg_id in seen:
            return False
        seen[msg_id] = True
        while len(seen) > 2000:
            seen.popitem(last=False)
        return True

    # ---------- browser demo ----------
    @app.get("/", response_class=HTMLResponse)
    def home():
        with open(os.path.join(HERE, "web", "index.html"), encoding="utf-8") as f:
            return f.read()

    def page(name):
        with open(os.path.join(HERE, "web", name), encoding="utf-8") as f:
            return f.read()

    @app.get("/privacy", response_class=HTMLResponse)
    def privacy():
        return page("privacy.html")

    @app.get("/manifest.webmanifest")
    def manifest():
        return Response(page("manifest.webmanifest"), media_type="application/manifest+json")

    @app.get("/sw.js")
    def service_worker():
        return Response(page("sw.js"), media_type="application/javascript",
                        headers={"Cache-Control": "no-cache"})

    @app.get("/api/config")
    def api_config():
        return {"name": "Truth_Rail", "whatsapp_url": wa_url, "llm_ready": bool(getattr(service.router, "ready", True)),
                "voice_replies": bool(service.voice and service.base_url)}

    @app.get("/api/stats")
    def api_stats():
        s = service.cache.stats() if service.cache else {"checks": 0, "served_from_cache": 0}
        return {"checks": s["checks"], "served_from_cache": s["served_from_cache"]}

    @app.post("/api/feedback")
    def api_feedback(fb: Feedback, request: Request):
        who = "fb:" + (request.client.host if request.client else "unknown")
        if not feedback_limiter.allow(who):
            return JSONResponse({"error": i18n.RATE_LIMITED}, status_code=429)
        label = fb.label if fb.label in config.LABELS else None
        kind = fb.kind if fb.kind in ("text", "image", "audio", "pdf", "link") else None
        if service.cache:
            service.cache.add_feedback(fb.rating, label, kind)
        return {"status": "thanks"}

    @app.get("/qr.svg")
    def qr(request: Request, target: str = "whatsapp"):
        data = (config.PUBLIC_BASE_URL or str(request.base_url).rstrip("/")) if target == "web" else wa_url
        if not data:
            raise HTTPException(404)
        try:
            import segno
        except ImportError:
            raise HTTPException(404, "QR codes need the 'segno' package")
        buf = io.BytesIO()
        segno.make(data, error="m").save(buf, kind="svg", scale=6, border=2, dark="#2c2c2a")
        return Response(buf.getvalue(), media_type="image/svg+xml")

    @app.get("/health")
    def health():
        return {"status": "ok", "whatsapp": {"twilio": bool(twilio), "meta": bool(meta)},
                "stats": service.cache.stats() if service.cache else None}

    @app.post("/api/check")
    def api_check(request: Request, text: str = Form(""), file: UploadFile = File(None)):
        who = "web:" + (request.client.host if request.client else "unknown")
        if not service.limiter.allow(who):
            return JSONResponse({"error": i18n.RATE_LIMITED}, status_code=429)
        media, data = None, None
        try:
            if file is not None and file.filename:
                data = file.file.read(config.MAX_MEDIA_MB * 1024 * 1024 + 1)
                _, mime = detect_kind(file.filename)
                media = Media(mime=mime)
            kind, result = service.check(text, media, (lambda m: data), who=who)
        except ExtractionError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        except LLMError as e:
            log.warning("AI request failed: %s", e)
            ready = getattr(service.router, "ready", True)
            msg = i18n.BUSY if ready else "Truth_Rail is not fully set up yet: no AI key was found."
            return JSONResponse(error_body(msg, e, request), status_code=503)
        except Exception as e:      # never send an unreadable error page to the browser
            log.exception("Unexpected error in /api/check")
            return JSONResponse(error_body("Something went wrong on our side. Please try again.", e, request),
                                status_code=500)
        return {"kind": kind, "card": format_card(result), "result": result}

    @app.get("/api/diagnose")
    def api_diagnose(request: Request):
        """Tests every API key, model and the search. Only available on your own computer."""
        if not allow_detail(request):
            raise HTTPException(403, "Diagnostics are only available on localhost (or set DEBUG_ERRORS=true).")
        router = service.router
        if not hasattr(router, "diagnose"):
            return {"ready": True, "results": []}
        report = router.diagnose()
        n, trusted, err = (service.retriever or EvidenceRetriever()).probe()
        report["search"] = {"results": n, "trusted": trusted, "error": err}
        return report

    # ---------- voice files ----------
    @app.get("/media/{name}")
    def media_file(name: str):
        path = service.voice.path_for(name) if service.voice else None
        if not path:
            raise HTTPException(404)
        return FileResponse(path, media_type="audio/mpeg")

    # ---------- Twilio ----------
    @app.post("/whatsapp/twilio")
    async def twilio_hook(request: Request, bg: BackgroundTasks):
        if not twilio:
            raise HTTPException(404, "Twilio is not configured")
        form = {k: str(v) for k, v in (await request.form()).items()}
        if validate_twilio:
            base = config.PUBLIC_BASE_URL
            url = (base + request.url.path) if base else str(request.url)
            if not twilio.validate(url, form, request.headers.get("X-Twilio-Signature", "")):
                raise HTTPException(403, "Bad signature")
        inbound = twilio.parse(form)
        if inbound and first_time(inbound.msg_id):
            bg.add_task(service.handle, inbound, twilio)
        return Response(EMPTY_TWIML, media_type="application/xml")

    # ---------- Meta WhatsApp Cloud API ----------
    @app.get("/whatsapp/meta")
    def meta_verify(request: Request):
        if not meta:
            raise HTTPException(404)
        challenge = meta.verify_challenge(dict(request.query_params))
        if challenge is None:
            raise HTTPException(403)
        return PlainTextResponse(challenge)

    @app.post("/whatsapp/meta")
    async def meta_hook(request: Request, bg: BackgroundTasks):
        if not meta:
            raise HTTPException(404)
        raw = await request.body()
        if not meta.validate(raw, request.headers.get("X-Hub-Signature-256", "")):
            raise HTTPException(403, "Bad signature")
        try:
            payload = json.loads(raw)
        except ValueError:
            raise HTTPException(400)
        for inbound in meta.parse(payload):
            if first_time(inbound.msg_id):
                bg.add_task(service.handle, inbound, meta)
        return {"status": "ok"}

    app.state.service = service
    return app


app = create_app() if os.getenv("FC_NO_AUTOSTART") != "1" else None
