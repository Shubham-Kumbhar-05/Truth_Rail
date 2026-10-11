"""Truth_Rail: try it in the terminal:
  python cli.py "forwarded text"          (a link on its own is opened and read)
  python cli.py --file message.txt
  python cli.py --image screenshot.png
  python cli.py --audio voice_note.ogg
  python cli.py --pdf document.pdf
  python cli.py --selftest      tests every API key, model and the search, and says what is wrong
  add --verbose to see why any API key was skipped, --no-cache to skip the saved answers
"""
import argparse
import sys

from forward_checker.cache import Cache
from forward_checker.extractors import ExtractionError, extract_text
from forward_checker.evidence import EvidenceRetriever
from forward_checker.llm_router import LLMError, make_router
from forward_checker.pipeline import check_forward, format_card
from forward_checker.service import BotService


def selftest(router):
    print("Truth_Rail self-test\n")
    if not router.ready:
        sys.exit("No API keys found. Put GEMINI_API_KEYS=... and/or GROQ_API_KEY=... in the .env file "
                 "(same folder as cli.py), then run this again.")
    gem = [p for p in router.keys if p.startswith("gemini")]
    print(f"Keys found: {len(gem)} Gemini, Groq: {'yes' if 'groq' in router.keys else 'no'}")
    report = router.diagnose()
    if report["discovered"] is not None:
        print("Newest Gemini Flash models available to your key:", ", ".join(report["discovered"]) or "none found")
    print("Gemini models tried, in order:", ", ".join(report["gemini_models"]) or "-")
    print("\nResults")
    for r in report["results"]:
        print(f"  {'OK  ' if r['ok'] else 'FAIL'}  {r['slot']:<42} {r['ms']:>5} ms  {r['error']}")
    n, trusted, err = EvidenceRetriever().probe()
    print("\nSearch:", f"{n} results, {trusted} from trusted sites" if not err else f"FAILED ({err})")
    if "No module named" in err:
        print("  Fix: run  pip install -r requirements.txt  (the search package is missing)")
    print()
    if report["ready"]:
        print("READY: at least one key/model works.")
        if not n or err:
            print("WARNING: search returned nothing, so most answers will be 'Cannot be confirmed'. "
                  "Try again in a minute (DuckDuckGo limits fast repeats) or add a free TAVILY_API_KEY.")
    else:
        print("NOT READY: no key/model worked. Read the errors above:")
        print("  - 'key rejected' -> the API key is wrong or the project has no access")
        print("  - 'quota / rate limit' -> wait a minute, or add another key")
        print("  - 'model not available' -> set GEMINI_MODELS=... in .env to a model listed above")
        print("  - 'network error' -> check the internet connection or proxy")
    sys.exit(0 if report["ready"] else 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="?", help="forwarded message text or a link")
    ap.add_argument("--file", help="text file with the message")
    ap.add_argument("--image", help="screenshot (png/jpg/webp)")
    ap.add_argument("--audio", help="voice note (ogg/opus/mp3/wav/m4a)")
    ap.add_argument("--pdf", help="PDF with text")
    ap.add_argument("--verbose", action="store_true", help="show which API keys were used or skipped, and why")
    ap.add_argument("--selftest", action="store_true", help="test every API key and model, then exit")
    ap.add_argument("--no-cache", action="store_true", help="do not reuse earlier answers")
    a = ap.parse_args()

    router = make_router()
    if a.selftest:
        return selftest(router)
    cache = None if a.no_cache else Cache()
    svc = BotService(router=router, cache=cache)
    path = a.image or a.audio or a.pdf or a.file
    try:
        kind, text = extract_text(router, path) if path else svc.to_text(a.text)
        if kind not in ("text",):
            print(f"[Read {kind} via {router.last_key or 'direct'}]\n{text[:1200]}\n")
        if not text:
            sys.exit("Give the message as text or a link, or use --file / --image / --audio / --pdf")
        result = check_forward(text, llm=router, cache=cache)
        if cache:
            cache.log("cli", kind, result["overall"], result["cached"], result["llm_used"])
    except (ExtractionError, LLMError) as e:
        sys.exit(str(e))
    print(format_card(result))
    print(f"\n[LLM used: {result['llm_used']}{' / ' + router.last_model if router.last_model and result['llm_used'] != 'cache' else ''}]")
    if a.verbose:
        print("\nKey log:")
        for key, outcome in router.events:
            print(f"  {key}: {outcome}")


if __name__ == "__main__":
    main()
