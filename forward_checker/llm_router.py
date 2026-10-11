"""LLM router for Truth_Rail.

Order of use:  Gemini key 1 (newest model -> older models), Gemini key 2 (...), ... then Groq models.

What it does for you
- Tries the next model / key / provider when one is rate-limited, rejected, retired or down.
- Remembers what failed (skips a rate-limited model for a while, a retired model for good, a bad key for a day).
- Finds the newest Gemini Flash models available to your key at start-up (no code change when Google ships a new one).
- If a model refuses an optional setting (thinking level, JSON mode), retries without it.
- Keeps a plain-language event log, and can test every key and model (see diagnose()).
"""
import base64
import json
import logging
import re
import time
from dataclasses import dataclass

import requests

from . import config

log = logging.getLogger("truth_rail.llm")


class LLMError(Exception):
    def __init__(self, message="", status=None, body=""):
        super().__init__(message)
        self.status = status
        self.body = body


class QuotaError(LLMError):
    def __init__(self, message, retry_after=None, daily=False, status=429, body=""):
        super().__init__(message, status, body)
        self.retry_after = retry_after
        self.daily = daily


class AuthError(LLMError):
    """Key invalid / suspended: do not retry this key for a long time."""


class ModelGone(LLMError):
    """The model name no longer exists (retired or misspelled): skip it for good."""


def parse_json(text):
    t = (text or "").strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.I).strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", t, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    raise LLMError("Model did not return valid JSON")


@dataclass(frozen=True)
class Slot:
    pid: str      # "gemini#1", "gemini#2", "groq"
    kind: str     # "gemini" | "groq"
    key: str
    model: str

    @property
    def label(self):
        return f"{self.pid}:{self.model}"


def discover_flash_models(listing, top=3):
    """From a models.list response: newest stable 'gemini-N-flash' / 'flash-lite' models that can generate text."""
    found = []
    for m in listing.get("models", []):
        name = (m.get("name") or "").replace("models/", "")
        if "generateContent" not in (m.get("supportedGenerationMethods") or []):
            continue
        mt = re.fullmatch(r"gemini-(\d+(?:\.\d+)*)-flash(-lite)?", name)
        if mt:
            version = tuple(int(x) for x in mt.group(1).split("."))
            found.append((version, 1 if mt.group(2) else 0, name))
    found.sort(key=lambda t: (tuple(-v for v in t[0]), t[1]))
    return [n for _, _, n in found[:top]]


class LLMRouter:
    def __init__(self, gemini_key=None, groq_key=None, gemini_model=None, groq_model=None,
                 http_post=None, http_get=None, now=None, gemini_models=None, groq_models=None,
                 discover=False, max_attempts=None):
        """gemini_key may be one key or a list of keys (tried in order); Groq is always last."""
        if gemini_key is None:
            gkeys = list(config.GEMINI_KEYS)
        elif isinstance(gemini_key, (list, tuple)):
            gkeys = [k for k in gemini_key if k]
        else:
            gkeys = [gemini_key] if gemini_key else []
        groq = config.GROQ_API_KEY if groq_key is None else groq_key
        self.keys = {f"gemini#{i}": k for i, k in enumerate(gkeys, 1)}
        if groq:
            self.keys["groq"] = groq

        self._explicit_gemini = list(gemini_models or ([gemini_model] if gemini_model else []) or config.GEMINI_MODELS)
        self._groq_models = {
            "json": list(groq_models or ([groq_model] if groq_model else []) or config.GROQ_MODELS),
            "image": list(config.GROQ_VISION_MODELS),
            "audio": list(config.GROQ_WHISPER_MODELS),
        }
        self.discover = discover
        self.discovered = None
        self._chain = None
        self.max_attempts = max_attempts or config.MAX_ATTEMPTS

        self._post = http_post or requests.post
        self._get = http_get or requests.get
        self._now = now or time.time
        self.blocked_until = {}          # slot label or key id -> time it may be used again
        self.dead_models = set()         # (kind, model) that no longer exist
        self._no_extras = set()          # (kind, model) that rejected optional settings
        self.calls = {}
        self.events = []                 # (slot label, outcome), for debugging
        self.last_provider = None        # "gemini" or "groq"
        self.last_key = None             # e.g. "gemini#2" or "groq"
        self.last_model = None

    # ---------- which key / model to try ----------
    @property
    def ready(self):
        return bool(self.keys)

    def gemini_models(self):
        """Models tried for every Gemini key, newest first."""
        if self._chain is not None:
            return self._chain
        chain = list(self._explicit_gemini)
        if not chain:
            found = self._discover() if self.discover else []
            chain = found + [m for m in config.DEFAULT_GEMINI_MODELS if m not in found]
            chain += [m for m in config.LEGACY_GEMINI_MODELS if m not in chain]
        self._chain = chain
        return chain

    def _discover(self):
        for pid, key in self.keys.items():
            if not pid.startswith("gemini"):
                continue
            try:
                resp = self._get("https://generativelanguage.googleapis.com/v1beta/models?pageSize=200",
                                 headers={"x-goog-api-key": key}, timeout=config.REQUEST_TIMEOUT)
                if resp.status_code == 200:
                    self.discovered = discover_flash_models(resp.json())
                    log.info("Newest Gemini Flash models for your key: %s", ", ".join(self.discovered) or "none found")
                    return self.discovered
            except Exception as e:      # discovery is a bonus; never block a request because of it
                log.info("Model discovery skipped: %s", e)
        return []

    def _slots(self, purpose):
        slots = []
        for pid, key in self.keys.items():
            if pid.startswith("gemini"):
                slots += [Slot(pid, "gemini", key, m) for m in self.gemini_models()]
            else:
                slots += [Slot(pid, "groq", key, m) for m in self._groq_models[purpose]]
        return slots

    def _usable(self, slot, now=None):
        now = self._now() if now is None else now
        return (self.blocked_until.get(slot.pid, 0) <= now
                and self.blocked_until.get(slot.label, 0) <= now
                and (slot.kind, slot.model) not in self.dead_models)

    def _available(self, purpose="json"):
        now = self._now()
        return [s for s in self._slots(purpose) if self._usable(s, now)]

    # ---------- shared fallback loop ----------
    def _run(self, purpose, call):
        if not self.keys:
            raise LLMError("No API key set. Add GEMINI_API_KEYS and/or GROQ_API_KEY to your .env file.")
        order = self._available(purpose)
        if not order:
            raise LLMError("All keys and models are cooling down after hitting their limits. Try again shortly.")
        errors, attempts = [], 0
        for slot in order:
            if not self._usable(slot):
                continue            # something learned earlier in this loop (bad key, retired model)
            if attempts >= self.max_attempts:
                errors.append(f"stopped after {attempts} attempts")
                break
            attempts += 1
            try:
                result = call(slot)
                self.calls[slot.label] = self.calls.get(slot.label, 0) + 1
                self.last_provider, self.last_key, self.last_model = slot.kind, slot.pid, slot.model
                self.events.append((slot.label, "ok"))
                return result
            except AuthError as e:
                self.blocked_until[slot.pid] = self._now() + config.BAD_KEY_COOLDOWN
                self._note(slot, f"key rejected ({e})", errors)
            except ModelGone as e:
                self.dead_models.add((slot.kind, slot.model))
                self._note(slot, f"model not available ({e})", errors)
            except QuotaError as e:
                gem = slot.kind == "gemini"
                cooldown = e.retry_after or (
                    config.GEMINI_COOLDOWN_PER_DAY if (gem and e.daily)
                    else config.GEMINI_COOLDOWN_PER_MINUTE if gem else config.GROQ_COOLDOWN)
                self.blocked_until[slot.label] = self._now() + cooldown
                self._note(slot, "quota / rate limit" + (" (daily)" if e.daily else " (per minute)"), errors)
            except LLMError as e:
                self._note(slot, f"error: {str(e)[:160]}", errors)
        raise LLMError("All providers failed: " + "; ".join(errors))

    def _note(self, slot, outcome, errors):
        self.events.append((slot.label, outcome))
        errors.append(f"{slot.label}: {outcome}")
        log.warning("%s -> %s", slot.label, outcome)

    # ---------- HTTP helpers ----------
    @staticmethod
    def _raise_for_status(resp):
        code = resp.status_code
        if code == 200:
            return
        body = (getattr(resp, "text", "") or "")[:600]
        low = body.lower()
        if code == 429:
            daily = any(k in low for k in ("perday", "per day", "daily", "rpd", "limit: 0"))
            retry = None
            try:
                retry = float(resp.headers.get("retry-after", ""))
            except (TypeError, ValueError, AttributeError):
                pass
            raise QuotaError(f"HTTP 429: {body[:160]}", retry_after=retry, daily=daily, body=body)
        if code in (401, 403) or (code == 400 and any(
                k in low for k in ("api key not valid", "api_key_invalid", "api key expired"))):
            raise AuthError(f"HTTP {code}", status=code, body=body)
        if "model" in low and (code == 404 or (code == 400 and any(
                k in low for k in ("model_not_found", "decommissioned", "does not exist",
                                   "is not found", "no longer supported", "not supported for generatecontent")))):
            raise ModelGone(f"HTTP {code}: {body[:140]}", status=code, body=body)
        raise LLMError(f"HTTP {code}: {body[:200]}", status=code, body=body)

    def _send(self, url, **kw):
        try:
            resp = self._post(url, timeout=config.REQUEST_TIMEOUT, **kw)
        except requests.RequestException as e:
            host = url.split("/")[2] if "//" in url else url
            raise LLMError(f"network error reaching {host} ({type(e).__name__}): check internet, firewall or proxy")
        self._raise_for_status(resp)
        return resp

    @staticmethod
    def _is_param_error(body):
        low = (body or "").lower()
        return any(k in low for k in ("thinking", "unknown name", "invalid json payload", "temperature",
                                      "response_format", "json mode", "unsupported", "not supported"))

    def _with_extras(self, slot, build_and_send):
        """Calls build_and_send(use_extras). If the model rejects optional settings, retries without them."""
        marker = (slot.kind, slot.model)
        use = marker not in self._no_extras
        try:
            return build_and_send(use)
        except (QuotaError, AuthError, ModelGone):
            raise
        except LLMError as e:
            if use and e.status == 400 and self._is_param_error(e.body):
                self._no_extras.add(marker)
                return build_and_send(False)
            raise

    # ---------- Gemini ----------
    @staticmethod
    def _gemini_url(model):
        return f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    @staticmethod
    def _gemini_config(model, json_mode, extras):
        cfg = {}
        if json_mode:
            cfg["responseMimeType"] = "application/json"
        if model.startswith("gemini-2"):
            cfg["temperature"] = 0             # Gemini 3 works best at its default temperature
        if extras:                              # keep answers fast: little or no hidden "thinking"
            if model.startswith("gemini-3"):
                cfg["thinkingConfig"] = {"thinkingLevel": "low"}
            elif model.startswith("gemini-2.5-flash") and "lite" not in model:
                cfg["thinkingConfig"] = {"thinkingBudget": 0}
        return cfg

    def _gemini_text(self, slot, parts, system=None, json_mode=False):
        def go(extras):
            payload = {"contents": [{"role": "user", "parts": parts}],
                       "generationConfig": self._gemini_config(slot.model, json_mode, extras)}
            if system:
                payload["systemInstruction"] = {"parts": [{"text": system}]}
            return self._send(self._gemini_url(slot.model), json=payload, headers={"x-goog-api-key": slot.key})
        resp = self._with_extras(slot, go)
        try:
            parts_out = resp.json()["candidates"][0]["content"]["parts"]
            return "".join(p.get("text", "") for p in parts_out if not p.get("thought"))
        except (KeyError, IndexError, TypeError, ValueError):
            raise LLMError("Gemini returned no usable answer (possibly blocked)")

    # ---------- Groq ----------
    def _groq_chat(self, slot, messages, json_mode=False):
        def go(extras):
            payload = {"model": slot.model, "messages": messages, "temperature": 0}
            if json_mode and extras:
                payload["response_format"] = {"type": "json_object"}
            return self._send("https://api.groq.com/openai/v1/chat/completions", json=payload,
                              headers={"Authorization": f"Bearer {slot.key}"})
        resp = self._with_extras(slot, go)
        try:
            return resp.json()["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError, ValueError):
            raise LLMError("Groq returned no usable answer")

    # ---------- public: JSON text ----------
    def generate_json(self, system, user):
        return self._run("json", lambda s: self._json_call(s, system, user))

    def _json_call(self, slot, system, user):
        if slot.kind == "gemini":
            text = self._gemini_text(slot, [{"text": user}], system=system, json_mode=True)
        else:
            text = self._groq_chat(slot, [{"role": "system", "content": system},
                                          {"role": "user", "content": user}], json_mode=True)
        return parse_json(text)

    # ---------- public: image / audio -> text ----------
    def transcribe(self, kind, data, mime, prompt, filename="file"):
        if kind not in ("image", "audio"):
            raise LLMError(f"Unsupported media kind: {kind}")

        def call(slot):
            if slot.kind == "gemini":
                text = self._gemini_text(slot, [{"inline_data": {"mime_type": mime,
                                                                 "data": base64.b64encode(data).decode()}},
                                                {"text": prompt}])
            else:
                text = self._groq_media(slot, kind, data, mime, prompt, filename)
            text = (text or "").strip()
            if not text:
                raise LLMError("empty transcription")
            return text
        return self._run(kind, call)

    def _groq_media(self, slot, kind, data, mime, prompt, filename):
        headers = {"Authorization": f"Bearer {slot.key}"}
        if kind == "audio":
            resp = self._send("https://api.groq.com/openai/v1/audio/transcriptions", headers=headers,
                              files={"file": (filename, data, mime)},
                              data={"model": slot.model, "response_format": "json", "temperature": "0"})
            try:
                return resp.json()["text"]
            except (KeyError, TypeError, ValueError):
                raise LLMError("Groq Whisper returned no usable transcription")
        return self._groq_chat(slot, [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{base64.b64encode(data).decode()}"}}]}])

    # ---------- self-test ----------
    def diagnose(self):
        """Tests every key and model with a tiny request. Does not change the cooldown state."""
        report = {"keys": {pid: pid for pid in self.keys}, "gemini_models": [], "discovered": None, "results": []}
        if self.keys:
            if not self._explicit_gemini:
                self._chain = None            # re-read the newest models for the report
            previous, self.discover = self.discover, True
            try:
                report["gemini_models"] = self.gemini_models()
            finally:
                self.discover = previous
            report["discovered"] = self.discovered
        probe = ('Reply with exactly this JSON and nothing else: {"ok": true}', "ping")
        first_ok_model = None
        for slot in self._slots("json"):
            if slot.kind == "gemini" and slot.pid != "gemini#1" and first_ok_model and slot.model != first_ok_model:
                continue      # other keys: test only with the model that worked, to save quota
            t0 = time.time()
            try:
                self._json_call(slot, *probe)
                ok, err = True, ""
                if slot.kind == "gemini" and first_ok_model is None:
                    first_ok_model = slot.model
            except LLMError as e:
                ok, err = False, f"{type(e).__name__}: {str(e)[:200]}"
            report["results"].append({"slot": slot.label, "ok": ok, "ms": int((time.time() - t0) * 1000), "error": err})
        report["ready"] = any(r["ok"] for r in report["results"])
        return report


def make_router():
    """The router used by the app: reads keys and models from .env and discovers the newest Gemini models."""
    return LLMRouter(discover=config.DISCOVER_MODELS)
