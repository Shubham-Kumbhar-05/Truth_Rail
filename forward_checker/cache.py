"""Module 9: SQLite cache of past checks + a tiny usage log."""
import hashlib
import json
import re
import sqlite3
import threading
import time

from . import config


def normalize(text):
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def hash_id(value):
    """Short one-way id, so logs never store phone numbers."""
    return hashlib.sha256((value or "").encode()).hexdigest()[:12]


class Cache:
    def __init__(self, path=None, now=None):
        self._now = now or time.time
        self._lock = threading.Lock()
        self._db = sqlite3.connect(path or config.CACHE_DB, check_same_thread=False)
        with self._lock:
            self._db.executescript("""
                CREATE TABLE IF NOT EXISTS results(
                    key TEXT PRIMARY KEY, created REAL, expires REAL, result TEXT);
                CREATE TABLE IF NOT EXISTS logs(
                    ts REAL, sender TEXT, kind TEXT, overall TEXT, cached INTEGER, llm TEXT);
                CREATE TABLE IF NOT EXISTS feedback(
                    ts REAL, rating TEXT, label TEXT, kind TEXT);
            """)
            self._db.commit()

    @staticmethod
    def key(text):
        return hashlib.sha256(normalize(text).encode()).hexdigest()

    def get(self, text):
        with self._lock:
            row = self._db.execute("SELECT expires, result FROM results WHERE key=?",
                                   (self.key(text),)).fetchone()
        if not row or row[0] <= self._now():
            return None
        return json.loads(row[1])

    def put(self, text, result, ttl_seconds):
        if ttl_seconds <= 0:
            return
        now = self._now()
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO results VALUES (?,?,?,?)",
                             (self.key(text), now, now + ttl_seconds, json.dumps(result)))
            self._db.commit()

    def log(self, sender, kind, overall, cached, llm):
        with self._lock:
            self._db.execute("INSERT INTO logs VALUES (?,?,?,?,?,?)",
                             (self._now(), hash_id(sender), kind, overall, int(bool(cached)), llm or ""))
            self._db.commit()

    def add_feedback(self, rating, label=None, kind=None):
        with self._lock:
            self._db.execute("INSERT INTO feedback VALUES (?,?,?,?)",
                             (self._now(), rating, label or "", kind or ""))
            self._db.commit()

    def stats(self):
        with self._lock:
            total, cached = self._db.execute(
                "SELECT COUNT(*), COALESCE(SUM(cached),0) FROM logs").fetchone()
            by_label = dict(self._db.execute(
                "SELECT overall, COUNT(*) FROM logs GROUP BY overall").fetchall())
            fb = dict(self._db.execute("SELECT rating, COUNT(*) FROM feedback GROUP BY rating").fetchall())
        return {"checks": total, "served_from_cache": cached, "by_label": by_label,
                "feedback": {"up": fb.get("up", 0), "down": fb.get("down", 0)}}
