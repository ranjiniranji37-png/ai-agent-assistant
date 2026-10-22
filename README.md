# 🛠️ AI Agent Assistant (Tools & Automation)

A working AI agent built with LLM **function / tool calling**.

**Agent loop:** Understand Task → Decide Action → Use Tool → Process Result → Respond

## Chatbot vs Agent
A chatbot only generates text. This agent decides when to call tools (calculator, weather, Wikipedia, currency, date/time, notes database), reads the results, repeats if needed (multi-step) and then answers.

## Tools
| Tool | What it does | Source |
|---|---|---|
| calculator | safe math (AST, no eval) | local |
| get_weather | current weather | Open-Meteo API |
| wikipedia_summary | facts | Wikipedia API |
| convert_currency | live exchange rates | Frankfurter API |
| get_current_datetime | date/time in any timezone | local |
| add_note / list_notes / delete_note | long-term memory | SQLite database |

## Memory and state
- Short-term: recent chat messages are sent back to the model (adjustable window).
- Long-term: notes stored in a SQLite database via tools.
- Each task shows its step trace (decide, tool call, result).

## Safety and limitations
- Tool allow-list: only enabled tools can be called; unknown tools are rejected.
- Destructive action (delete note) is blocked unless the user enables it.
- Max steps per task stops endless loops; tool results are truncated; HTTP calls have timeouts.
- Calculator uses a safe parser, never `eval`.
- Errors in tools are returned to the model as text, not crashes.
- Limitations: models can still choose the wrong tool or misread results; free APIs can be slow or rate-limited; notes on free hosting may reset when the app restarts.

## Run locally
```bash
pip install -r requirements.txt
cp .env.example .env   # add your key
streamlit run app.py
```

## Deploy
share.streamlit.io -> New app -> main file `app.py` -> Secrets: `GEMINI_API_KEY = "..."`
