"""
pages/chatbot.py
================
AI Health Assistant chatbot page.

Features:
  - Multi-session chat (each conversation saved separately)
  - Auto-titles sessions from first user message
  - Context-aware: injects latest prediction result + SHAP values into system prompt
  - Persistent: all sessions saved to SQLite, survive app restarts
  - Session types: 'normal' (regular chat) and 'report' (from OCR upload)

LLM: Groq Llama 3.1 8b Instant (free tier)
     ~14,400 requests/day, 30/minute — plenty for personal use

Context injection:
  Every message to Groq includes the patient's disease, risk score, all feature
  values, and top 5 SHAP drivers in the system prompt. This is why the AI can
  answer questions like "why is my glucose affecting my score so much?"
  with specific, personalised answers rather than generic advice.
"""

import streamlit as st

from utils.database import (create_chat_session, load_chat_sessions,
                             save_chat_messages, delete_chat_session)
from utils.llm      import groq_chat, build_prediction_context

import os
GROQ_API_KEY = ""   # set in app.py

SYSTEM_PROMPT = """You are MediXAI's Health Assistant — a knowledgeable, compassionate medical AI.

You have access to the patient's latest prediction results (context injected below).
Use this context to give specific, personalised answers.

RULES:
1. Always clarify you are an AI and not a doctor. Recommend consulting a professional.
2. Never diagnose. Explain what the ML model found and what it means statistically.
3. Explain SHAP values in plain, simple language when asked.
4. Be warm, clear, and concise.
5. For lifestyle advice, give concrete actionable suggestions based on the patient's values.
6. For medication questions, always defer to a doctor.
7. If no prediction has been run, encourage the user to make a prediction first."""


def _send_and_save(user_input: str, uid: int, api_key: str):
    """
    Core function: append user message → get Groq reply → save to DB.
    Auto-titles the session if it's still called 'New Chat'.
    """
    history = st.session_state.get("chat_messages", [])
    history.append({"role": "user", "content": user_input})

    system = SYSTEM_PROMPT + build_prediction_context(st.session_state.get("last_result"))
    reply  = groq_chat(history[-20:], system, api_key)  # last 20 messages to stay within token limit
    history.append({"role": "assistant", "content": reply})
    st.session_state["chat_messages"] = history

    # Persist to DB
    chat_id = st.session_state.get("chat_session_id")
    if chat_id:
        sessions = load_chat_sessions(uid)
        current  = next((s for s in sessions if s["id"] == chat_id), None)
        if current and current["title"] in ("New Chat", "", None):
            auto_title = user_input[:28] + ("..." if len(user_input) > 28 else "")
            save_chat_messages(chat_id, history, title=auto_title)
        else:
            save_chat_messages(chat_id, history)


def show():
    st.title("💬 AI Health Assistant")
    st.caption("Context-aware · Groq Llama 3 · Multi-session · Knows your prediction results")

    uid     = st.session_state.get("user_id", 0)
    api_key = st.session_state.get("api_key", GROQ_API_KEY)

    # ── Sidebar ────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 🔑 Groq API Key")
        api_key = st.text_input(
            "API Key", type="password",
            value=st.session_state.get("api_key", GROQ_API_KEY),
            placeholder="gsk_...",
            help="Get your free key at console.groq.com",
        )
        if api_key:
            st.session_state["api_key"] = api_key

        # Show current prediction context
        st.markdown("### 🧠 Current Context")
        result = st.session_state.get("last_result")
        if result:
            st.success(
                f"✅ {result['disease']}\n"
                f"{result['risk_level']} ({result['risk_percent']:.1f}%)"
            )
        else:
            st.info("Run a prediction first to unlock context-aware answers.")

        # Session management
        st.markdown("### 💬 Chat Sessions")
        sessions = load_chat_sessions(uid)

        if st.button("➕ New Chat", use_container_width=True):
            sid = create_chat_session(uid, title="New Chat")
            st.session_state["chat_session_id"] = sid
            st.session_state["chat_messages"]   = []
            st.rerun()

        for s in sessions:
            tag      = "📝" if s["chat_type"] == "report" else "💬"
            short    = s["title"][:22] + ("..." if len(s["title"]) > 22 else "")
            is_active = (s["id"] == st.session_state.get("chat_session_id"))
            label    = f"{'> ' if is_active else ''}{tag} {short}"
            if st.button(label, key=f"sess_{s['id']}", use_container_width=True):
                st.session_state["chat_session_id"] = s["id"]
                st.session_state["chat_messages"]   = s["messages"]
                st.rerun()

        st.markdown("---")
        col_clr, col_del = st.columns(2)
        with col_clr:
            if st.button("Clear", use_container_width=True):
                st.session_state["chat_messages"] = []
                if "chat_session_id" in st.session_state:
                    save_chat_messages(st.session_state["chat_session_id"], [])
                st.rerun()
        with col_del:
            if st.button("Delete", use_container_width=True):
                if "chat_session_id" in st.session_state:
                    delete_chat_session(st.session_state["chat_session_id"])
                    remaining = load_chat_sessions(uid)
                    if remaining:
                        st.session_state["chat_session_id"] = remaining[0]["id"]
                        st.session_state["chat_messages"]   = remaining[0]["messages"]
                    else:
                        sid = create_chat_session(uid)
                        st.session_state["chat_session_id"] = sid
                        st.session_state["chat_messages"]   = []
                st.rerun()

    # ── Ensure a session exists ────────────────────────────────────
    if "chat_session_id" not in st.session_state:
        sessions = load_chat_sessions(uid)
        if sessions:
            st.session_state["chat_session_id"] = sessions[0]["id"]
            st.session_state["chat_messages"]   = sessions[0]["messages"]
        else:
            sid = create_chat_session(uid)
            st.session_state["chat_session_id"] = sid
            st.session_state["chat_messages"]   = []

    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = []

    # ── Welcome message ────────────────────────────────────────────
    if not st.session_state["chat_messages"]:
        username = st.session_state.get("username", "there")
        result   = st.session_state.get("last_result")
        if result:
            welcome = (
                f"👋 Hello **{username}**! I can see your latest **{result['disease']}** "
                f"prediction shows **{result['risk_level']}** ({result['risk_percent']:.1f}%). "
                "Ask me anything about your results, SHAP values, or what you can do."
            )
        else:
            welcome = (
                f"👋 Hello **{username}**! I'm your MediXAI Health Assistant. "
                "Run a prediction first, then I'll give you personalised insights about your results."
            )
        with st.chat_message("assistant"):
            st.markdown(welcome)

    # ── Quick question buttons ─────────────────────────────────────
    if not st.session_state["chat_messages"] and st.session_state.get("last_result"):
        st.markdown("**Quick questions:**")
        questions = [
            "What does my risk score mean?",
            "Which factors affected my result the most?",
            "What lifestyle changes could lower my risk?",
            "Explain my SHAP values in simple terms.",
            "What should I discuss with my doctor?",
            "Is my glucose / cholesterol level concerning?",
        ]
        cols = st.columns(3)
        for i, q in enumerate(questions):
            if cols[i % 3].button(q, key=f"sq_{i}"):
                _send_and_save(q, uid, api_key)
                st.rerun()

    # ── Conversation history ───────────────────────────────────────
    for msg in st.session_state["chat_messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ── Input ──────────────────────────────────────────────────────
    user_input = st.chat_input("Ask about your results, risk factors, or health advice...")
    if user_input:
        _send_and_save(user_input, uid, api_key)
        st.rerun()
