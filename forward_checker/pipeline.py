"""Ties modules 4-6 together and formats a WhatsApp-friendly verdict card."""
from . import config, i18n
from .claim_splitter import split_claims
from .evidence import EvidenceRetriever, is_time_sensitive
from .llm_router import make_router
from .verdict import judge

ICONS = {"Verified": "✅", "False": "❌", "Outdated": "⏳",
         "Partly supported": "⚠️", "Cannot be confirmed": "❓"}
FLAG_TEXT = {
    "fake_urgency": "pushes you to act in a hurry",
    "chain_forward_request": "asks you to forward it to many people",
    "fake_authority": "may be using fake official branding",
    "too_good_to_be_true": "offers something too good to be true",
    "fear_appeal": "tries to scare you",
    "suspicious_link": "contains a suspicious link",
}


def cache_ttl(result):
    """How long a result may be reused. Time-sensitive facts and 'unknown' answers go stale faster."""
    hours = config.CACHE_TTL_HOURS
    if any(is_time_sensitive(c["claim"]) for c in result["claims"]):
        hours = min(hours, config.CACHE_TTL_TIME_SENSITIVE_HOURS)
    if result["overall"] == "Cannot be confirmed":
        hours = min(hours, config.CACHE_TTL_UNCONFIRMED_HOURS)
    return hours * 3600


def check_forward(text, llm=None, retriever=None, cache=None):
    if cache:
        hit = cache.get(text)
        if hit:
            hit["cached"], hit["llm_used"] = True, "cache"
            return hit
    llm = llm or make_router()
    retriever = retriever or EvidenceRetriever()
    parsed = split_claims(llm, text)
    results = []
    for c in parsed["claims"]:
        ev = retriever.search(c["text"])
        v = judge(llm, c["text"], ev, parsed["language"])
        results.append({"claim": c["text"], "topic": c["topic"], **v})
    overall = None
    for lab in config.SEVERITY_ORDER:
        if any(r["label"] == lab for r in results):
            overall = lab
            break
    result = {"language": parsed["language"], "overall": overall or "No factual claim found",
              "claims": results, "manipulation_flags": parsed["manipulation_flags"],
              "llm_used": getattr(llm, "last_key", None) or getattr(llm, "last_provider", None),
              "cached": False}
    if cache:
        cache.put(text, result, cache_ttl(result))
    return result


def format_card(result):
    if not result["claims"]:
        return "I could not find a factual claim to check in this message."
    lines = []
    for r in result["claims"]:
        local = i18n.local_label(result.get("language"), r["label"])
        title = f"{ICONS[r['label']]} *{r['label']}*" + (f" ({local})" if local else "")
        lines.append(f"{title}\nClaim: {r['claim']}\nWhy: {r['reason']}")
        for s in r["sources"]:
            lines.append(f"Source: {s['url']}")
        lines.append("")
    if result["manipulation_flags"]:
        warn = "; ".join(FLAG_TEXT[f] for f in result["manipulation_flags"])
        lines.append(f"🚩 Be careful: this message {warn}.")
    return "\n".join(lines).strip()
