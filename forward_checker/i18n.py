"""Small translations for labels, help text and the spoken (voice) reply: English, Hindi, Marathi."""

LABELS_LOCAL = {
    "hi": {"Verified": "सही पाया गया", "False": "झूठ", "Outdated": "पुराना, अब मान्य नहीं",
           "Partly supported": "आंशिक रूप से सही", "Cannot be confirmed": "पुष्टि नहीं हो सकी"},
    "mr": {"Verified": "खरे आढळले", "False": "खोटे", "Outdated": "जुने, आता वैध नाही",
           "Partly supported": "अंशतः बरोबर", "Cannot be confirmed": "पुष्टी होऊ शकली नाही"},
}

FLAG_SPOKEN = {
    "en": "Be careful. This message tries to rush you or make you forward it.",
    "hi": "सावधान रहें। यह संदेश आपको जल्दबाजी में कदम उठाने या आगे भेजने के लिए उकसा रहा है।",
    "mr": "सावध राहा. हा मेसेज तुम्हाला घाईघाईत निर्णय घ्यायला किंवा पुढे पाठवायला भाग पाडत आहे.",
}

HELP = (
    "Hello! I am Truth_Rail. Send me any forwarded message, screenshot, voice note, PDF or link, and I will check "
    "it against trusted sources and tell you if it is true.\n\n"
    "नमस्ते! मैं Truth_Rail हूँ। कोई भी फॉरवर्ड संदेश, स्क्रीनशॉट, वॉइस नोट, PDF या लिंक भेजें, मैं जाँचकर बताऊँगा कि वह सच है या नहीं।\n\n"
    "नमस्कार! मी Truth_Rail आहे. कोणताही फॉरवर्ड मेसेज, स्क्रीनशॉट, व्हॉइस नोट, PDF किंवा लिंक पाठवा, मी तपासून सांगेन की तो खरा आहे का."
)
ACK = "Checking... this takes about 10-20 seconds. ⏳ / जाँच हो रही है... / तपासत आहे..."
BUSY = ("Sorry, I am very busy right now. Please try again in a minute. / "
        "अभी बहुत व्यस्त हूँ, एक मिनट बाद फिर भेजें। / आत्ता खूप व्यस्त आहे, एका मिनिटाने पुन्हा पाठवा.")
RATE_LIMITED = ("You have sent many messages this hour. Please try again a little later. / "
                "आपने इस घंटे बहुत संदेश भेजे हैं, थोड़ी देर बाद कोशिश करें। / "
                "तुम्ही या तासात खूप मेसेज पाठवले आहेत, थोड्या वेळाने प्रयत्न करा.")
GREETINGS = {"hi", "hello", "hey", "help", "start", "menu", "namaste", "hii", "नमस्ते", "नमस्कार", "हेल्प"}

SUPPORTED_VOICE = ("en", "hi", "mr")


def local_label(lang, label):
    return LABELS_LOCAL.get(lang, {}).get(label)


def spoken_script(result, max_chars=600):
    """Short text for the voice reply: verdict of the first claims, plus a warning if flagged."""
    lang = result.get("language", "en")
    if lang not in SUPPORTED_VOICE:
        lang = "en"
    parts = []
    for r in result.get("claims", [])[:2]:
        label = local_label(lang, r["label"]) or r["label"]
        parts.append(f"{label}. {r['reason']}")
    if result.get("manipulation_flags"):
        parts.append(FLAG_SPOKEN[lang])
    return " ".join(parts)[:max_chars]
