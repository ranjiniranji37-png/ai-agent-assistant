"""AI Agent Assistant (Module 5): Understand Task -> Decide Action -> Use Tool -> Process Result -> Respond"""
import json
import os

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

import agent_core as ac

load_dotenv()
st.set_page_config(page_title="AI Agent Assistant", page_icon="🛠️", layout="centered")

PROVIDERS = {
    "Google Gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "key_name": "GEMINI_API_KEY",
        "models": ["gemini-3.8-flash"],
        "key_help": "Free key: aistudio.google.com/apikey",
    },
    "Groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "key_name": "GROQ_API_KEY",
        "models": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
        "key_help": "Free key: console.groq.com/keys",
    },
}
EXAMPLES = [
    "What is 15% of 2480, plus the square root of 144?",
    "Weather in Chennai now, and convert 100 USD to INR",
    "Save a note: submit Module 5 tomorrow. Then show my notes",
    "Who was Alan Turing? Summarise in 2 lines",
]


def get_secret(name):
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.getenv(name, "")


st.session_state.setdefault("messages", [])
st.session_state.setdefault("tokens", {"prompt": 0, "completion": 0, "runs": 0})

# ------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    provider_name = st.selectbox("LLM provider", list(PROVIDERS))
    provider = PROVIDERS[provider_name]
    model = st.selectbox("Model", provider["models"])
    api_key = get_secret(provider["key_name"])
    if not api_key:
        api_key = st.text_input("API key", type="password")
        st.caption(provider["key_help"])
    else:
        st.success("API key loaded ✅")

    st.subheader("🧰 Tools")
    enabled = []
    for group, names in ac.tool_groups().items():
        if st.checkbox(group, value=True, key=f"tool_{group}"):
            enabled += names
    allow_destructive = st.toggle("Allow destructive actions (delete note)", value=False,
                                  help="Safety: deleting is blocked unless you enable this.")

    st.subheader("🛡️ Agent limits")
    max_steps = st.slider("Max steps per task", 1, 8, 5, help="Prevents endless tool loops.")
    memory_window = st.slider("Chat memory (messages)", 0, 20, 8, 2)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)

    st.subheader("🗒️ Saved notes (database)")
    rows = ac.fetch_notes()
    if rows:
        for i, t, c in rows:
            st.caption(f"#{i} · {t}")
    else:
        st.caption("No notes yet. Ask the agent to save one.")

    tk = st.session_state.tokens
    st.caption(f"🔢 {tk['prompt']} in / {tk['completion']} out tokens · {tk['runs']} tasks")
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ------------------------------------------------------------ main
st.title("🛠️ AI Agent Assistant")
st.caption(f"Tool-calling agent · {provider_name} · {model}")


def show_steps(steps):
    for s in steps:
        if s["type"] == "decide":
            st.markdown(f"🤔 **Decide:** {s['content']}")
        elif s["type"] == "tool_call":
            st.markdown(f"🔧 **Use tool:** `{s['name']}` `{s['args']}`")
        elif s["type"] == "tool_result":
            st.markdown(f"📥 **Result:** {s['result']}")


def render(m):
    st.markdown(m["content"])
    if m.get("steps"):
        with st.expander(f"🧠 Agent steps ({sum(1 for s in m['steps'] if s['type'] == 'tool_call')} tool calls)"):
            show_steps(m["steps"])


if not st.session_state.messages:
    st.info("Give the agent a task. It decides which tool to use, runs it, and answers. Try an example:")
    for ex in EXAMPLES:
        if st.button(ex, use_container_width=True):
            st.session_state.pending = ex
            st.rerun()

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        render(m)

typed = st.chat_input("Give the agent a task...")
task = typed or st.session_state.pop("pending", None)

if task:
    if not api_key:
        st.error("Please add your API key in the sidebar first.")
        st.stop()
    if len(task) > 1500:
        st.warning("Task too long (max 1500 characters).")
        st.stop()

    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    history = history[-memory_window:] if memory_window else []
    st.session_state.messages.append({"role": "user", "content": task})
    with st.chat_message("user"):
        st.markdown(task)

    with st.chat_message("assistant"):
        steps, final = [], None
        try:
            client = OpenAI(api_key=api_key, base_url=provider["base_url"])
            with st.status("Agent is working...", expanded=True) as status:
                for ev in ac.run_agent(client, model, history, task, enabled,
                                       allow_destructive, max_steps, temperature):
                    if ev["type"] == "final":
                        final = ev
                    else:
                        steps.append(ev)
                        show_steps([ev])
                status.update(label="Done", state="complete", expanded=False)
            tk = st.session_state.tokens
            tk["prompt"] += final["usage"]["prompt"]
            tk["completion"] += final["usage"]["completion"]
            tk["runs"] += 1
            st.session_state.messages.append(
                {"role": "assistant", "content": final["content"], "steps": steps})
            st.rerun()
        except RuntimeError as err:
            st.error(f"⚠️ {err}")
            st.session_state.messages.pop()
