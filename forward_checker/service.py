"""The bot brain: one inbound WhatsApp (or web) message in, replies out."""
import logging
import threading
import time
from collections import defaultdict, deque

from . import config, i18n
from .cache import Cache
from .extractors import ExtractionError, extract_bytes, kind_from_mime, resolve_link
from .llm_router import LLMError, make_router
from .messaging import MessagingError
from .pipeline import check_forward, format_card


log = logging.getLogger("truth_rail")


class RateLimiter:
    def __init__(self, per_hour=None, now=None):
        self.limit = config.RATE_LIMIT_PER_HOUR if per_hour is None else per_hour
        self._now = now or time.time
        self._hits = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, who):
        now = self._now()
        with self._lock:
            q = self._hits[who]
            while q and now - q[0] > 3600:
                q.popleft()
            if len(q) >= self.limit:
                return False
            q.append(now)
            return True


class BotService:
    def __init__(self, router=None, retriever=None, cache=None, voice=None, limiter=None,
                 public_base_url=None):
        self.router = router or make_router()
        self.retriever = retriever
        self.cache = cache
        self.voice = voice
        self.limiter = limiter or RateLimiter()
        self.base_url = (public_base_url if public_base_url is not None else config.PUBLIC_BASE_URL)

    # ---- turn any input into text ----
    def to_text(self, text, media=None, downloader=None):
        """Returns (kind, text). kind is text|image|audio|pdf|link."""
        text = (text or "").strip()
        if media is not None:
            kind, mime = kind_from_mime(media.mime)
            data = downloader(media)
            body = extract_bytes(self.router, data, kind, mime, filename=f"upload.{mime.split('/')[-1]}")
            return kind, (f"{text}\n{body}".strip() if text else body)
        page = resolve_link(text)
        if page:
            return "link", page
        return "text", text

    def check(self, text, media=None, downloader=None, who="web"):
        """For the web page and CLI: returns (kind, result). Raises ExtractionError/LLMError."""
        kind, body = self.to_text(text, media, downloader)
        if not body:
            raise ExtractionError("Nothing to check. Send a message, screenshot, voice note, PDF or link.")
        result = check_forward(body, llm=self.router, retriever=self.retriever, cache=self.cache)
        if self.cache:
            self.cache.log(who, kind, result["overall"], result["cached"], result["llm_used"])
        return kind, result

    # ---- WhatsApp ----
    def handle(self, inbound, messenger):
        """Runs in the background after the webhook has already answered."""
        to = inbound.sender
        try:
            self._handle(inbound, messenger)
        except MessagingError:
            pass  # nothing more we can do if WhatsApp itself rejects our reply
        except Exception:
            log.exception("Unexpected error while handling a WhatsApp message")
            try:
                messenger.send_text(to, i18n.BUSY)
            except MessagingError:
                pass

    def _handle(self, inbound, messenger):
        to, text = inbound.sender, inbound.text
        if text == "\u2063unsupported":
            return messenger.send_text(to, "Sorry, I can read text, screenshots, voice notes, PDFs and links only.")
        if not inbound.media and (not text or text.lower().strip("!. ") in i18n.GREETINGS):
            return messenger.send_text(to, i18n.HELP)
        if not self.limiter.allow(to):
            return messenger.send_text(to, i18n.RATE_LIMITED)
        if config.SEND_ACK:
            messenger.send_text(to, i18n.ACK)
        try:
            kind, result = self.check(text, inbound.media, messenger.download, who=to)
        except ExtractionError as e:
            return messenger.send_text(to, str(e))
        except LLMError:
            return messenger.send_text(to, i18n.BUSY)
        messenger.send_text(to, format_card(result))
        self._maybe_voice(to, kind, result, messenger)

    def _maybe_voice(self, to, kind, result, messenger):
        mode = config.VOICE_REPLY_MODE
        if mode == "never" or not self.voice or not self.base_url or not result["claims"]:
            return
        wants = mode == "always" or kind == "audio" or result.get("language") in ("hi", "mr")
        if not wants:
            return
        name = self.voice.synthesize(i18n.spoken_script(result), result.get("language", "en"))
        if name:
            messenger.send_audio(to, f"{self.base_url}/media/{name}")
