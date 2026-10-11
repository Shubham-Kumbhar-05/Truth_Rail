import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")


def _split_keys(*raw):
    seen, out = set(), []
    for chunk in raw:
        for k in (chunk or "").replace("\n", ",").replace(";", ",").split(","):
            k = k.strip()
            if k and k not in seen:
                seen.add(k)
                out.append(k)
    return out


# GEMINI_API_KEYS=key1,key2,key3 (tried in this order). GEMINI_API_KEY (single) still works.
GEMINI_KEYS = _split_keys(os.getenv("GEMINI_API_KEYS", ""), GEMINI_API_KEY)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")



def _csv(name):
    return [x.strip() for x in os.getenv(name, "").replace(";", ",").split(",") if x.strip()]



# --- Model configuration ---

# Verified working in the latest Truth_Rail self-test.
# Newest-first order based on successful tests.
DEFAULT_GEMINI_MODELS = [
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite",
]

# Explicit .env overrides; empty means use defaults/discovery.
GEMINI_MODELS = _csv("GEMINI_MODELS")
LEGACY_GEMINI_MODELS = _csv("GEMINI_MODEL")

# Keep working Groq models first.
GROQ_MODELS = _csv("GROQ_MODELS") or [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
]

GROQ_VISION_MODELS = _csv("GROQ_VISION_MODELS") or [
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
]

GROQ_WHISPER_MODELS = _csv("GROQ_WHISPER_MODELS") or [
    "whisper-large-v3-turbo",
    "whisper-large-v3",
]

# Explicit configuration prevents discovery from inserting untested models first.
DISCOVER_MODELS = os.getenv(
    "DISCOVER_MODELS", "false"
).lower() == "true"

MAX_ATTEMPTS = int(os.getenv("MAX_ATTEMPTS", "12"))
DEBUG_ERRORS = os.getenv("DEBUG_ERRORS", "false").lower() == "true"

# Network settings
REQUEST_TIMEOUT = 15

GEMINI_COOLDOWN_PER_MINUTE = 65
GEMINI_COOLDOWN_PER_DAY = 3600
GROQ_COOLDOWN = 65
BAD_KEY_COOLDOWN = 24 * 3600


MAX_CLAIMS = 4  # keeps LLM calls per forward low (1 split + up to 4 verdicts)

LABELS = ["Verified", "False", "Outdated", "Partly supported", "Cannot be confirmed"]
SEVERITY_ORDER = ["False", "Outdated", "Partly supported", "Cannot be confirmed", "Verified"]

MANIPULATION_FLAGS = [
    "fake_urgency",
    "chain_forward_request",
    "fake_authority",
    "too_good_to_be_true",
    "fear_appeal",
    "suspicious_link",
]

# Exact domains (and their subdomains) we accept as evidence
TRUSTED_DOMAINS = [
    "who.int", "cdc.gov", "nih.gov", "nhs.uk",
    "pib.gov.in", "mohfw.gov.in", "icmr.gov.in", "rbi.org.in", "uidai.gov.in",
    "incometax.gov.in", "india.gov.in",
    "reuters.com", "apnews.com", "bbc.com", "thehindu.com", "pti.in",
    "altnews.in", "boomlive.in", "factly.in", "newschecker.in", "vishvasnews.com",
    "factcheck.org", "snopes.com", "afp.com",
]
# Any host ending with one of these is also accepted (government sites)
TRUSTED_SUFFIXES = [".gov.in", ".nic.in", ".gov"]

# --- media (screenshots, voice notes) ---
MAX_MEDIA_MB = 15

# --- cache, limits ---
CACHE_DB = os.getenv("CACHE_DB", "forward_checker.db")
CACHE_TTL_HOURS = float(os.getenv("CACHE_TTL_HOURS", "24"))
CACHE_TTL_TIME_SENSITIVE_HOURS = 6      # deadlines, "from tomorrow", dates: facts change
CACHE_TTL_UNCONFIRMED_HOURS = 2         # new evidence may appear soon
MAX_TEXT_CHARS = 6000                   # text from PDFs / web pages sent to the LLM
URL_FETCH_TIMEOUT = 10
MAX_URL_BYTES = 2_000_000
RATE_LIMIT_PER_HOUR = int(os.getenv("RATE_LIMIT_PER_HOUR", "20"))

# --- public access ---
WHATSAPP_NUMBER = os.getenv("WHATSAPP_NUMBER", "")        # digits only, e.g. 14155238886 (country code first)
WHATSAPP_JOIN_TEXT = os.getenv("WHATSAPP_JOIN_TEXT", "hi")  # Twilio sandbox: the "join your-word" code

# --- WhatsApp / server ---
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")   # e.g. https://abc.ngrok.app
VOICE_REPLY_MODE = os.getenv("VOICE_REPLY_MODE", "auto")         # auto | always | never
VOICE_DIR = os.getenv("VOICE_DIR", "voice_cache")
SEND_ACK = os.getenv("SEND_ACK", "true").lower() == "true"       # "checking..." message

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_WHATSAPP_FROM = os.getenv("TWILIO_WHATSAPP_FROM", "")     # e.g. whatsapp:+14155238886
TWILIO_VALIDATE_SIGNATURE = os.getenv("TWILIO_VALIDATE_SIGNATURE", "true").lower() == "true"

META_VERIFY_TOKEN = os.getenv("META_VERIFY_TOKEN", "")
META_ACCESS_TOKEN = os.getenv("META_ACCESS_TOKEN", "")
META_PHONE_NUMBER_ID = os.getenv("META_PHONE_NUMBER_ID", "")
META_APP_SECRET = os.getenv("META_APP_SECRET", "")
