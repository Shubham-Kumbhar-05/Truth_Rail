"""Module 10: WhatsApp adapters. Twilio (easy sandbox for demos) and Meta WhatsApp Cloud API."""
import base64
import hashlib
import hmac
from dataclasses import dataclass
from typing import Optional

import requests

from . import config


class MessagingError(Exception):
    pass


@dataclass
class Media:
    url: str = ""
    mime: str = ""
    media_id: str = ""     # Meta only


@dataclass
class Inbound:
    sender: str            # id used to reply (Twilio: "whatsapp:+91...", Meta: "91...")
    text: str = ""
    media: Optional[Media] = None
    msg_id: str = ""


def chunk(text, limit):
    """Split long text at line breaks so each WhatsApp message stays under the limit."""
    parts, cur = [], ""
    for line in text.split("\n"):
        while len(line) > limit:
            if cur:
                parts.append(cur.rstrip()); cur = ""
            parts.append(line[:limit]); line = line[limit:]
        if len(cur) + len(line) + 1 > limit:
            parts.append(cur.rstrip()); cur = ""
        cur += line + "\n"
    if cur.strip():
        parts.append(cur.rstrip())
    return parts or [""]


def _download(http, url, limit_mb=None, **kw):
    limit = (limit_mb or config.MAX_MEDIA_MB) * 1024 * 1024
    try:
        resp = http.get(url, stream=True, timeout=config.REQUEST_TIMEOUT, **kw)
    except requests.RequestException as e:
        raise MessagingError(f"Could not download the media: {e}")
    if resp.status_code != 200:
        raise MessagingError(f"Media download failed (HTTP {resp.status_code}).")
    data = b""
    for part in resp.iter_content(65536):
        data += part
        if len(data) > limit:
            raise MessagingError("The file is too large.")
    return data


class TwilioAdapter:
    name = "twilio"
    max_len = 1500

    def __init__(self, sid=None, token=None, from_number=None, http=None):
        self.sid = sid or config.TWILIO_ACCOUNT_SID
        self.token = token or config.TWILIO_AUTH_TOKEN
        self.from_number = from_number or config.TWILIO_WHATSAPP_FROM
        self.http = http or requests

    def validate(self, url, form, signature):
        """Twilio signs: url + sorted(key+value) with HMAC-SHA1 of the auth token."""
        base = url + "".join(k + str(form[k]) for k in sorted(form))
        expected = base64.b64encode(
            hmac.new(self.token.encode(), base.encode(), hashlib.sha1).digest()).decode()
        return hmac.compare_digest(expected, signature or "")

    def parse(self, form):
        sender = form.get("From", "")
        if not sender:
            return None
        media = None
        if int(form.get("NumMedia", "0") or 0) > 0 and form.get("MediaUrl0"):
            media = Media(url=form["MediaUrl0"], mime=form.get("MediaContentType0", ""))
        return Inbound(sender=sender, text=(form.get("Body") or "").strip(), media=media,
                       msg_id=form.get("MessageSid", ""))

    def download(self, media):
        return _download(self.http, media.url, auth=(self.sid, self.token))

    def _send(self, to, **fields):
        url = f"https://api.twilio.com/2010-04-01/Accounts/{self.sid}/Messages.json"
        try:
            resp = self.http.post(url, auth=(self.sid, self.token), timeout=config.REQUEST_TIMEOUT,
                                  data={"From": self.from_number, "To": to, **fields})
        except requests.RequestException as e:
            raise MessagingError(f"Could not send: {e}")
        if resp.status_code >= 300:
            raise MessagingError(f"Twilio error HTTP {resp.status_code}: {(resp.text or '')[:200]}")

    def send_text(self, to, body):
        for part in chunk(body, self.max_len):
            self._send(to, Body=part)

    def send_audio(self, to, url):
        self._send(to, MediaUrl=url)


class MetaAdapter:
    name = "meta"
    max_len = 3800
    GRAPH = "https://graph.facebook.com/v21.0"

    def __init__(self, token=None, phone_id=None, verify_token=None, app_secret=None, http=None):
        self.token = token or config.META_ACCESS_TOKEN
        self.phone_id = phone_id or config.META_PHONE_NUMBER_ID
        self.verify_token = verify_token or config.META_VERIFY_TOKEN
        self.app_secret = app_secret or config.META_APP_SECRET
        self.http = http or requests

    def verify_challenge(self, params):
        if (params.get("hub.mode") == "subscribe" and self.verify_token
                and hmac.compare_digest(params.get("hub.verify_token", ""), self.verify_token)):
            return params.get("hub.challenge", "")
        return None

    def validate(self, raw_body, signature_header):
        if not self.app_secret:
            return False
        expected = "sha256=" + hmac.new(self.app_secret.encode(), raw_body, hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, signature_header or "")

    def parse(self, payload):
        out = []
        for entry in payload.get("entry", []):
            for change in entry.get("changes", []):
                for m in change.get("value", {}).get("messages", []) or []:
                    kind = m.get("type")
                    text, media = "", None
                    if kind == "text":
                        text = (m.get("text") or {}).get("body", "")
                    elif kind in ("image", "audio", "document"):
                        d = m.get(kind) or {}
                        text = d.get("caption", "") or ""
                        media = Media(mime=d.get("mime_type", ""), media_id=d.get("id", ""))
                    else:
                        text = "\u2063unsupported"   # handled as unsupported by the service
                    out.append(Inbound(sender=m.get("from", ""), text=text.strip(), media=media,
                                       msg_id=m.get("id", "")))
        return out

    def _headers(self):
        return {"Authorization": f"Bearer {self.token}"}

    def download(self, media):
        try:
            info = self.http.get(f"{self.GRAPH}/{media.media_id}", headers=self._headers(),
                                 timeout=config.REQUEST_TIMEOUT)
            url = info.json()["url"]
        except (requests.RequestException, KeyError, ValueError) as e:
            raise MessagingError(f"Could not get the media link: {e}")
        return _download(self.http, url, headers=self._headers())

    def _send(self, payload):
        try:
            resp = self.http.post(f"{self.GRAPH}/{self.phone_id}/messages", headers=self._headers(),
                                  json={"messaging_product": "whatsapp", **payload},
                                  timeout=config.REQUEST_TIMEOUT)
        except requests.RequestException as e:
            raise MessagingError(f"Could not send: {e}")
        if resp.status_code >= 300:
            raise MessagingError(f"WhatsApp error HTTP {resp.status_code}: {(resp.text or '')[:200]}")

    def send_text(self, to, body):
        for part in chunk(body, self.max_len):
            self._send({"to": to, "type": "text", "text": {"body": part, "preview_url": False}})

    def send_audio(self, to, url):
        self._send({"to": to, "type": "audio", "audio": {"link": url}})
