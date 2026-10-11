"""Module 8 (voice): turn the verdict into a short spoken reply (mp3) in the user's language."""
import os
import re
import secrets
import time

from . import config
from .i18n import SUPPORTED_VOICE

NAME_RE = re.compile(r"^[a-f0-9]{32}\.mp3$")


def gtts_tts(text, lang, path):
    from gtts import gTTS
    gTTS(text=text, lang=lang).save(path)


class VoiceStore:
    def __init__(self, directory=None, tts=None, now=None, max_age=3600):
        self.dir = directory or os.getenv( "VOICE_CACHE_DIR", "/tmp/voice_cache" )
        self._tts = tts or gtts_tts
        self._now = now or time.time
        self.max_age = max_age
        os.makedirs(self.dir, exist_ok=True)

    def synthesize(self, text, lang):
        """Returns the file name of a new mp3, or None if voice is not possible."""
        lang = lang if lang in SUPPORTED_VOICE else None
        if not lang or not text.strip():
            return None
        self.cleanup()
        name = secrets.token_hex(16) + ".mp3"
        try:
            self._tts(text, lang, os.path.join(self.dir, name))
        except Exception:
            return None
        return name

    def path_for(self, name):
        if not NAME_RE.match(name or ""):
            return None
        path = os.path.join(self.dir, name)
        return path if os.path.isfile(path) else None

    def cleanup(self):
        now = self._now()
        for f in os.listdir(self.dir):
            p = os.path.join(self.dir, f)
            if NAME_RE.match(f) and now - os.path.getmtime(p) > self.max_age:
                try:
                    os.remove(p)
                except OSError:
                    pass
