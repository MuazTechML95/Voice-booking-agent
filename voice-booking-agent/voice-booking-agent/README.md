# Voice Appointment Booking Agent

A generic, **config-driven** AI voice appointment booking agent built with
Streamlit. The same codebase works for medical clinics, law firms, salons,
consultants, service providers, educational institutes, or customer support
desks — just by editing `config/config.json`, with **no changes to the
core code**.

## Features

- 🎤 Voice input (record from your microphone in the browser, transcribed with Whisper)
- ⌨️ Text input as a fallback/alternative to voice
- 🔊 Optional spoken responses (pyttsx3 Text-to-Speech)
- 🧠 OpenAI-assisted intent detection, with an automatic rule-based fallback
- 📋 Step-by-step collection of Name, Phone, Date, Time, Purpose, and Business Type
- ✅ Confirmation card before saving, with explicit Confirm / Edit buttons
- 💾 SQLite storage (`database/appointments.db`), created automatically
- 📊 Appointment history table + dashboard metrics (total / confirmed / cancelled)
- 🔁 Reschedule and ❌ Cancel existing appointments
- 📚 **RAG-powered FAQ / services / pricing answers** — ask "what are your hours?" or "how much is a haircut?" anytime during the conversation; the agent retrieves the answer from `config/knowledge/` and resumes booking exactly where it left off
- ⚠️ Robust error handling: empty input, invalid phone/date/time, cancellation,
  speech-recognition failure, and API errors are all handled gracefully

## Project Structure

```
voice-booking-agent/
├── streamlit_app.py            # Main Streamlit UI / entry point
├── config/
│   └── config.json             # All business types, categories, questions, wording
├── modules/
│   ├── speech_to_text.py       # Whisper-based STT (local model or OpenAI API)
│   ├── text_to_speech.py       # pyttsx3-based TTS, rendered to playable audio
│   ├── conversation_manager.py # Generic dialog state machine + intent detection
│   ├── appointment_manager.py  # Booking / reschedule / cancel business logic
│   ├── database.py             # SQLite persistence layer
│   ├── validator.py            # Field validators (name, phone, date, time...)
│   └── utils.py                # Config loading, logging, helpers
├── database/
│   └── appointments.db         # Created automatically on first run
├── requirements.txt
├── README.md
└── .env                        # Your OPENAI_API_KEY goes here
```

## Setup Instructions

1. **Clone / unzip the project**, then move into the folder:
   ```bash
   cd voice-booking-agent
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   Notes:
   - `openai-whisper` needs `ffmpeg` installed on your system
     (e.g. `sudo apt install ffmpeg` on Ubuntu, `brew install ffmpeg` on macOS).
   - `pyttsx3` uses your OS's native TTS engine. On Linux you may need:
     `sudo apt install espeak`.
   - If you don't want to install the heavy local Whisper model, skip
     `openai-whisper` — the app will automatically use the OpenAI Whisper
     API instead, as long as `OPENAI_API_KEY` is set.

4. **Add your OpenAI API key:**
   Open `.env` and replace the placeholder:
   ```
   OPENAI_API_KEY=sk-...your-key...
   ```
   The app still runs without a key (it falls back to rule-based intent
   detection and the local Whisper model), but an API key enables smarter
   NLP and the hosted Whisper fallback.

5. **Run the app:**
   ```bash
   streamlit run streamlit_app.py
   ```
   Open the URL Streamlit prints (usually `http://localhost:8501`).

## Configuring for a New Industry

Open `config/config.json` and edit:

- `business_types`: list of industries shown in the dropdown
- `appointment_categories`: a list of categories per business type
  (falls back to `"default"` if a business type isn't listed)
- `questions`: the ordered list of fields to collect (each has a `key`,
  `type`, `prompt`, and `retry_prompt`)
- `messages`: all conversational wording, with `{placeholders}` filled
  in automatically from the business name/type/answers
- `working_hours`: valid appointment time range

No Python code needs to change to support a new business type.

## RAG (FAQ / Services / Pricing) Knowledge Base

Each business type has a plain-text knowledge file under `config/knowledge/`
(mapped in `config.json` under `"knowledge_files"`). At runtime:

1. The file is split into paragraph-sized chunks.
2. Each chunk is embedded once via OpenAI's `text-embedding-3-small` and
   cached locally in `database/embeddings_cache.json` (re-embedded
   automatically only if the source file changes).
3. When a user asks a question (anywhere in the conversation — e.g.
   "what are your hours?" or "how much is a haircut?"), the most relevant
   chunks are retrieved by cosine similarity and passed to `gpt-4o-mini`,
   which answers using **only** that retrieved context.
4. The agent then re-shows whatever booking prompt it was waiting on, so
   the appointment flow is never lost.

To update a business's FAQs, services, or pricing, just edit its `.txt`
file in `config/knowledge/` — no code changes needed. This requires an
`OPENAI_API_KEY`; without one, the agent politely says it can't look
that up right now and continues the booking flow normally.

## Database Schema

```sql
CREATE TABLE appointments (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    business_type     TEXT NOT NULL,
    full_name         TEXT NOT NULL,
    phone             TEXT NOT NULL,
    appointment_date  TEXT NOT NULL,
    appointment_time  TEXT NOT NULL,
    purpose           TEXT,
    category          TEXT,     -- extra column, supports the optional category feature
    status            TEXT NOT NULL DEFAULT 'CONFIRMED',  -- extra column, supports cancel/reschedule
    created_at        TEXT NOT NULL
);
```

## Troubleshooting

| Problem | Likely cause / fix |
|---|---|
| "No OPENAI_API_KEY found" warning | Add your key to `.env`, or ignore it — rule-based fallback still works |
| Voice input does nothing | Browser microphone permission may be blocked; check the browser's site settings |
| "Speech recognition failed" | No local Whisper model and no API key — install `openai-whisper` or add an API key |
| No sound on "speak responses" | `pyttsx3` couldn't find a system TTS engine — install `espeak` (Linux) or use text-only mode |
| `appointments.db` not found | It's created automatically on first run; make sure the app has write access to the `database/` folder |
