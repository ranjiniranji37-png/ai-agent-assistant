"""Core of the AI Agent: tools, safety checks and the agent loop.
Loop:  Understand Task -> Decide Action -> Use Tool -> Process Result -> Respond
No Streamlit code here, so it is easy to test and explain."""
import ast
import json
import math
import operator
import os
import sqlite3
import tempfile
import time
import urllib.parse
import urllib.request
from datetime import datetime
from zoneinfo import ZoneInfo

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    RateLimitError,
)

DB_PATH = os.path.join(tempfile.gettempdir(), "agent_notes.db")
MAX_RESULT_CHARS = 1500


# ============================================================ TOOLS
# ---- 1. calculator (safe: no eval) ----
_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv, ast.USub: operator.neg, ast.UAdd: operator.pos,
}
_FUNCS = {"sqrt": math.sqrt, "abs": abs, "round": round, "sin": math.sin,
          "cos": math.cos, "tan": math.tan, "log": math.log, "log10": math.log10}
_CONSTS = {"pi": math.pi, "e": math.e}


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.Name) and node.id in _CONSTS:
        return _CONSTS[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("exponent too large")
        return _OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS:
        return _FUNCS[node.func.id](*[_eval(a) for a in node.args])
    raise ValueError("unsupported expression")


def calculator(expression: str) -> str:
    if len(expression) > 200:
        return "Error: expression too long"
    try:
        value = _eval(ast.parse(expression.replace("^", "**"), mode="eval").body)
        return f"{expression} = {round(value, 10)}"
    except ZeroDivisionError:
        return "Error: division by zero"
    except Exception as e:
        return f"Error: could not calculate ({e})"


# ---- helpers for external APIs ----
def _get_json(url: str, timeout: int = 8):
    req = urllib.request.Request(url, headers={"User-Agent": "ai-agent-assistant/1.0 (student project)"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read(300_000).decode("utf-8"))


WEATHER_CODES = {0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast", 45: "fog",
                 51: "light drizzle", 61: "light rain", 63: "rain", 65: "heavy rain",
                 71: "light snow", 80: "rain showers", 95: "thunderstorm"}


# ---- 2. weather (Open-Meteo, no API key) ----
def get_weather(city: str) -> str:
    try:
        geo = _get_json("https://geocoding-api.open-meteo.com/v1/search?count=1&name=" + urllib.parse.quote(city))
        if not geo.get("results"):
            return f"Could not find a place called '{city}'."
        p = geo["results"][0]
        w = _get_json(
            "https://api.open-meteo.com/v1/forecast?current=temperature_2m,relative_humidity_2m,"
            f"wind_speed_10m,weather_code&timezone=auto&latitude={p['latitude']}&longitude={p['longitude']}"
        )["current"]
        desc = WEATHER_CODES.get(w.get("weather_code"), "unknown conditions")
        return (f"Weather in {p['name']}, {p.get('country', '')}: {w['temperature_2m']}°C, {desc}, "
                f"humidity {w['relative_humidity_2m']}%, wind {w['wind_speed_10m']} km/h")
    except Exception as e:
        return f"Error: weather service unavailable ({type(e).__name__})"


# ---- 3. Wikipedia summary ----
def wikipedia_summary(topic: str) -> str:
    try:
        found = _get_json("https://en.wikipedia.org/w/api.php?action=opensearch&limit=1&format=json&search="
                          + urllib.parse.quote(topic))
        if not found[1]:
            return f"No Wikipedia article found for '{topic}'."
        title = found[1][0]
        data = _get_json("https://en.wikipedia.org/api/rest_v1/page/summary/"
                         + urllib.parse.quote(title.replace(" ", "_")))
        return f"{data.get('title', title)}: {data.get('extract', 'No summary')[:800]}"
    except Exception as e:
        return f"Error: Wikipedia unavailable ({type(e).__name__})"


# ---- 4. currency conversion (Frankfurter, no API key) ----
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    try:
        f, t = from_currency.upper()[:3], to_currency.upper()[:3]
        d = _get_json(f"https://api.frankfurter.dev/v1/latest?amount={float(amount)}&base={f}&symbols={t}")
        return f"{amount} {f} = {d['rates'][t]} {t} (rate date {d['date']})"
    except Exception as e:
        return f"Error: currency service unavailable or bad currency code ({type(e).__name__})"


# ---- 5. date and time ----
def get_current_datetime(timezone: str = "Asia/Kolkata") -> str:
    try:
        now = datetime.now(ZoneInfo(timezone))
        return now.strftime(f"%A, %d %B %Y, %I:%M %p ({timezone})")
    except Exception:
        return "Error: unknown timezone. Use names like Asia/Kolkata or America/New_York."


# ---- 6. notes database (SQLite) = long-term agent memory ----
def _db():
    con = sqlite3.connect(DB_PATH)
    con.execute("CREATE TABLE IF NOT EXISTS notes (id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT, created TEXT)")
    return con


def add_note(text: str) -> str:
    text = text.strip()[:300]
    if not text:
        return "Error: empty note"
    with _db() as con:
        cur = con.execute("INSERT INTO notes (text, created) VALUES (?, ?)",
                          (text, datetime.now().strftime("%Y-%m-%d %H:%M")))
    return f"Saved note #{cur.lastrowid}: {text}"


def list_notes() -> str:
    with _db() as con:
        rows = con.execute("SELECT id, text, created FROM notes ORDER BY id DESC LIMIT 20").fetchall()
    if not rows:
        return "No notes saved yet."
    return "\n".join(f"#{i} [{c}] {t}" for i, t, c in rows)


def delete_note(note_id: int) -> str:
    with _db() as con:
        cur = con.execute("DELETE FROM notes WHERE id = ?", (int(note_id),))
    return f"Deleted note #{note_id}" if cur.rowcount else f"No note with id {note_id}"


def fetch_notes():
    """For the UI sidebar."""
    with _db() as con:
        return con.execute("SELECT id, text, created FROM notes ORDER BY id DESC LIMIT 20").fetchall()


def _obj(props, required):
    return {"type": "object", "properties": props, "required": required}


TOOLS = {
    "calculator": dict(fn=calculator, group="Calculator", destructive=False,
        description="Evaluate a math expression exactly. Use for ANY arithmetic.",
        parameters=_obj({"expression": {"type": "string", "description": "e.g. (45*12)+sqrt(81)"}}, ["expression"])),
    "get_weather": dict(fn=get_weather, group="Weather", destructive=False,
        description="Get the current weather for a city.",
        parameters=_obj({"city": {"type": "string"}}, ["city"])),
    "wikipedia_summary": dict(fn=wikipedia_summary, group="Wikipedia", destructive=False,
        description="Get a short factual summary of a topic, person or place from Wikipedia.",
        parameters=_obj({"topic": {"type": "string"}}, ["topic"])),
    "convert_currency": dict(fn=convert_currency, group="Currency", destructive=False,
        description="Convert an amount between currencies using live exchange rates.",
        parameters=_obj({"amount": {"type": "number"},
                         "from_currency": {"type": "string", "description": "3-letter code, e.g. USD"},
                         "to_currency": {"type": "string", "description": "3-letter code, e.g. INR"}},
                        ["amount", "from_currency", "to_currency"])),
    "get_current_datetime": dict(fn=get_current_datetime, group="Date & time", destructive=False,
        description="Get the current date and time in a timezone (default Asia/Kolkata).",
        parameters=_obj({"timezone": {"type": "string", "description": "IANA name, e.g. Asia/Kolkata"}}, [])),
    "add_note": dict(fn=add_note, group="Notes (database)", destructive=False,
        description="Save a note or reminder for the user in the database.",
        parameters=_obj({"text": {"type": "string"}}, ["text"])),
    "list_notes": dict(fn=list_notes, group="Notes (database)", destructive=False,
        description="List the user's saved notes.", parameters=_obj({}, [])),
    "delete_note": dict(fn=delete_note, group="Notes (database)", destructive=True,
        description="Delete a saved note by its id. Destructive action.",
        parameters=_obj({"note_id": {"type": "integer"}}, ["note_id"])),
}


def tool_groups():
    groups = {}
    for name, t in TOOLS.items():
        groups.setdefault(t["group"], []).append(name)
    return groups


def schemas(enabled):
    return [{"type": "function", "function": {"name": n, "description": TOOLS[n]["description"],
                                              "parameters": TOOLS[n]["parameters"]}}
            for n in TOOLS if n in enabled]


# ============================================================ SAFETY + EXECUTION
def execute_tool(name, raw_args, enabled, allow_destructive):
    """Validate and run one tool call. Never raises - errors become text for the LLM."""
    if name not in TOOLS or name not in enabled:
        return f"Error: tool '{name}' is not available."
    if TOOLS[name]["destructive"] and not allow_destructive:
        return ("Blocked by safety rule: this action deletes data. "
                "Ask the user to enable 'Allow destructive actions' in the sidebar.")
    try:
        args = json.loads(raw_args) if isinstance(raw_args, str) and raw_args.strip() else (raw_args or {})
        if not isinstance(args, dict):
            return "Error: tool arguments must be a JSON object."
        result = str(TOOLS[name]["fn"](**args))
    except TypeError as e:
        return f"Error: wrong arguments for {name} ({e})"
    except Exception as e:
        return f"Error while running {name}: {e}"
    return result[:MAX_RESULT_CHARS]


# ============================================================ LLM CALL + AGENT LOOP
SYSTEM_PROMPT = (
    "You are a helpful AI agent that can use tools. First understand the user's task, then decide: "
    "answer directly if no tool is needed, otherwise call the right tool(s). "
    "Use tools for arithmetic, current date/time, weather, facts, currency and notes - never guess these. "
    "For multi-step tasks, call tools one after another and use earlier results. "
    "Never invent tool results. If a tool fails or is blocked, say so honestly. "
    "Finish with a short, clear answer for the user."
)


def llm_call(client, model, messages, temperature, tools=None, retries=3):
    kwargs = dict(model=model, messages=messages, temperature=temperature, max_tokens=800, timeout=60)
    if tools:
        kwargs["tools"] = tools
    for attempt in range(1, retries + 1):
        try:
            return client.chat.completions.create(**kwargs)
        except AuthenticationError:
            raise RuntimeError("Invalid API key. Check your key / secrets.")
        except RateLimitError:
            if attempt == retries:
                raise RuntimeError("Rate limit reached. Wait a minute and try again.")
            time.sleep(2 ** attempt)
        except (APIConnectionError, APITimeoutError):
            if attempt == retries:
                raise RuntimeError("Network problem / timeout. Please retry.")
            time.sleep(2 ** attempt)
        except APIStatusError as e:
            if e.status_code >= 500 and attempt < retries:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"API error {e.status_code}: {getattr(e, 'message', str(e))[:300]}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error: {e}")


def run_agent(client, model, history, user_message, enabled, allow_destructive=False,
              max_steps=5, temperature=0.2):
    """Generator yielding events: decide, tool_call, tool_result, final.
    `history` = earlier plain user/assistant messages (short-term memory)."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + list(history) + \
               [{"role": "user", "content": user_message}]
    tools = schemas(enabled) or None
    usage = {"prompt": 0, "completion": 0}

    def add_usage(resp):
        u = getattr(resp, "usage", None)
        usage["prompt"] += getattr(u, "prompt_tokens", 0) or 0
        usage["completion"] += getattr(u, "completion_tokens", 0) or 0

    for step in range(1, max_steps + 1):
        resp = llm_call(client, model, messages, temperature, tools)
        add_usage(resp)
        msg = resp.choices[0].message
        calls = getattr(msg, "tool_calls", None) or []
        if not calls:  # Respond
            yield {"type": "final", "content": msg.content or "(no answer)", "usage": usage}
            return
        if msg.content:
            yield {"type": "decide", "step": step, "content": msg.content}
        # keep provider-specific fields (model_dump) so the next request is accepted
        messages.append({"role": "assistant", "content": msg.content or None,
                         "tool_calls": [c.model_dump(exclude_none=True) for c in calls]})
        for c in calls:  # Use tool -> Process result
            name, raw = c.function.name, c.function.arguments
            yield {"type": "tool_call", "step": step, "name": name, "args": raw}
            result = execute_tool(name, raw, enabled, allow_destructive)
            yield {"type": "tool_result", "step": step, "name": name, "result": result}
            messages.append({"role": "tool", "tool_call_id": c.id, "content": result})

    # step limit reached -> ask for best answer without tools
    messages.append({"role": "user", "content":
                     "Step limit reached. Give your best final answer from the tool results so far."})
    resp = llm_call(client, model, messages, temperature, None)
    add_usage(resp)
    text = (resp.choices[0].message.content or "") + f"\n\n_(Stopped after {max_steps} steps - safety limit.)_"
    yield {"type": "final", "content": text, "usage": usage}
