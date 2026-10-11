# Truth_Rail  (PS 3 "Is This Forward True?", GenAI)

Checks forwarded messages, screenshots, voice notes, PDFs and links against trusted sources and
answers with one of five labels (Verified, False, Outdated, Partly supported, Cannot be confirmed),
a reason in simple words, and the sources. Works as a WhatsApp bot and as a browser page.

## 1. Setup
    pip install -r requirements.txt
    copy .env.example .env        (Windows)   |   cp .env.example .env
    # fill in GEMINI_API_KEYS and GROQ_API_KEY

## 1b. Check that your keys and models work (do this first)
    python cli.py --selftest
It tests every Gemini key, every model and the search, and prints OK or FAIL with the reason for each one.
If the website shows an error, this command (or opening http://localhost:8000/api/diagnose on your own computer)
tells you exactly why. The website also shows a "Technical details" box on errors when you run it on localhost.

Models: Truth_Rail asks Google which Gemini Flash models your key can use and tries the newest first
(today: gemini-3.5-flash, gemini-3.1-flash-lite), then older ones as backups. A model that is rate-limited, retired
or refuses a setting is skipped automatically. Groq is the last fallback (llama-3.3-70b-versatile, openai/gpt-oss-120b,
openai/gpt-oss-20b; images: llama-4 vision; voice: whisper-large-v3-turbo). To pin models yourself, set GEMINI_MODELS
or GROQ_MODELS in .env.

## 2. Try it in the terminal
    python cli.py "Drinking hot water with lemon cures cancer. Share with everyone!"
    python cli.py "https://example.com/some-article"
    python cli.py --image samples/lemon_forward.png
    python cli.py --audio voice_note.m4a
    python cli.py --pdf document.pdf
    python cli.py --verbose "..."        # shows which API key answered / why keys were skipped

## 3. Run the server (web page + WhatsApp webhooks)
    uvicorn server:app --port 8000
    open http://localhost:8000           # the browser demo page (works without WhatsApp)

## 4. Connect WhatsApp with Twilio (easiest for a demo)
1. Create a free Twilio account, open Messaging > Try it out > Send a WhatsApp message (the sandbox).
2. On your phone, send the join code shown there (for example "join something-word") to the sandbox number.
3. Make your server public. Easiest: install ngrok and run `ngrok http 8000`. Copy the https address.
4. Put in .env:  PUBLIC_BASE_URL=https://xxxx.ngrok-free.app, TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN,
   TWILIO_WHATSAPP_FROM=whatsapp:+14155238886. Restart the server.
5. In the Twilio sandbox settings, set "When a message comes in" to
   https://xxxx.ngrok-free.app/whatsapp/twilio  (method POST).
6. Send a forward to the sandbox number from your phone. Send "hi" to see the help message.
If replies never come: check http://localhost:8000/health, make sure PUBLIC_BASE_URL is only the
address (no path, no trailing slash), and temporarily set TWILIO_VALIDATE_SIGNATURE=false to find out
whether the signature check is the problem.

## 4b. Meta WhatsApp Cloud API (optional alternative)
Set META_* values in .env. Webhook URL: https://your-host/whatsapp/meta , verify token = META_VERIFY_TOKEN.
META_APP_SECRET is required: requests without a valid X-Hub-Signature-256 are rejected.

## 5. Deploy for free (so it stays online)
Render / Railway: connect the repo, build command `pip install -r requirements.txt`,
start command `uvicorn server:app --host 0.0.0.0 --port $PORT`, add the .env values as environment
variables, and set PUBLIC_BASE_URL to the address the host gives you.
Free hosts sleep when idle (the first message can be slow) and may lose files on restart:
the cache and voice files are only a convenience, so that is fine.

## 6. How people get access
- **Website / app:** share your public address. People open it, no sign-up. On a phone they can tap
  "Add to Home screen" (or the "Install app" button) and it behaves like an app. Languages: English, Hindi, Marathi.
- **WhatsApp:** set WHATSAPP_NUMBER (and WHATSAPP_JOIN_TEXT for the Twilio sandbox). The website then shows an
  "Open WhatsApp" button and a QR code. Print or show the QR codes: /qr.svg (WhatsApp) and /qr.svg?target=web (website).
- **Share-back:** every verdict has a "Share on WhatsApp" button, so one check reaches the whole family group.
- The Twilio sandbox is for testing and demos: each person joins once with the join code. A permanent public
  WhatsApp number needs a Meta business account (see section 4b).
- Privacy page: /privacy. Health check: /health.

## Troubleshooting /api/check errors
| What you see | What to do |
|---|---|
| "Sorry, I am very busy..." | Run `python cli.py --selftest`. It names the real cause (rate limit, bad key, no internet) |
| "not fully set up: no AI key" | Put GEMINI_API_KEYS and/or GROQ_API_KEY in `.env` (same folder as server.py) and restart |
| Every answer is "We could not confirm this" | Search is failing. Self-test shows it. Run `pip install -r requirements.txt`, wait a minute, or add TAVILY_API_KEY |
| "model not available" in the self-test | The name was retired. Remove GEMINI_MODEL / GEMINI_MODELS from `.env` so the newest are found automatically |
| Microphone does not work | Open the site through `localhost` or https, and allow the microphone |
| Changed `.env` but nothing changed | Stop the server (Ctrl + C) and start it again |

## How the pieces fit
- llm_router.py       Gemini key 1 -> 2 -> ... -> N -> Groq, for text, images and audio
- claim_splitter.py   claims + manipulation flags      - evidence.py  trusted-source search
- verdict.py          labels with guard rails           - cache.py     SQLite cache + usage log
- extractors.py       image / voice / PDF / link -> text   - webfetch.py  safe page download (blocks private addresses)
- voice.py + i18n.py  voice reply in English, Hindi, Marathi
- messaging.py        Twilio + Meta adapters            - service.py   the bot logic (limits, replies)
- server.py + web/    webhooks, web app (installable), QR codes, voice files, privacy page
- docs/wireframes/    laptop + phone wireframe;  docs/screenshots/  real screenshots of the site

## Tests (no internet or keys needed)
    python -m unittest discover -s tests
    python tests/e2e_ui.py        # optional: clicks through the web page in a real browser (pip install -r requirements-dev.txt)

## Good to know
- Gemini quota is per Google Cloud project, not per key. Keep the key pool small.
- Voice replies use gTTS (needs internet, no key). They are sent only if PUBLIC_BASE_URL is set.
- Never commit .env. It is in .gitignore.
