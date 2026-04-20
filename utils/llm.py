"""
utils/llm.py
============
All Groq / LLM interaction for MediXAI.

Model used: llama-3.1-8b-instant  (free tier, fast)
API:        Groq  →  https://console.groq.com  (free account, no billing needed)

Two entry points:
  groq_call()  → single prompt/response  (used by Recommendations page)
  groq_chat()  → full conversation with history  (used by AI Chatbot page)
"""

GROQ_MODEL = "llama-3.1-8b-instant"


def groq_call(prompt: str, system: str, api_key: str) -> str:
    """
    Send a single prompt to Groq and return the reply as a string.
    Used for one-shot tasks like generating a recommendations plan.

    Args:
        prompt   : the user's question / instruction
        system   : system-level instruction that shapes the AI's behaviour
        api_key  : Groq API key (from session_state or GROQ_API_KEY constant)
    """
    if not api_key:
        return "⚠️ No Groq API key set. Paste your key into GROQ_API_KEY in app.py."
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        resp = client.chat.completions.create(
            model    = GROQ_MODEL,
            messages = [
                {"role": "system", "content": system},
                {"role": "user",   "content": prompt},
            ],
            max_tokens  = 1024,
            temperature = 0.7,
        )
        return resp.choices[0].message.content
    except ImportError:
        return "⚠️ `groq` package not installed. Run: pip install groq"
    except Exception as e:
        return f"⚠️ Groq error: {e}"


def groq_chat(messages: list, system: str, api_key: str) -> str:
    """
    Continue a multi-turn conversation with Groq.
    Passes full conversation history so the AI remembers context.

    Args:
        messages : list of {"role": "user"/"assistant", "content": "..."}
        system   : system prompt with prediction context injected
        api_key  : Groq API key

    To keep requests within token limits, callers pass messages[-20:].
    """
    if not api_key:
        return (
            "⚠️ No Groq API key set. Enter it in the sidebar or paste it into "
            "the GROQ_API_KEY variable at the top of app.py\n\n"
            "Get your free key at https://console.groq.com"
        )
    try:
        from groq import Groq
        client = Groq(api_key=api_key)

        groq_messages = [{"role": "system", "content": system}]
        for msg in messages:
            groq_messages.append({"role": msg["role"], "content": msg["content"]})

        resp = client.chat.completions.create(
            model       = GROQ_MODEL,
            messages    = groq_messages,
            max_tokens  = 1024,
            temperature = 0.7,
        )
        return resp.choices[0].message.content
    except ImportError:
        return "⚠️ `groq` package not installed. Run: pip install groq"
    except Exception as e:
        return f"⚠️ Groq error: {e}"


def build_prediction_context(result: dict) -> str:
    """
    Convert a prediction result dict into a text block that gets
    injected into the system prompt so the AI knows exactly what
    this patient's numbers and SHAP drivers are.

    Called by both the Chatbot page and the Recommendations page.
    """
    if not result:
        return "\n\nNo prediction has been run yet in this session."

    labels = result.get("labels", {})
    lines  = [
        "\n\n--- PATIENT PREDICTION CONTEXT ---",
        f"Disease     : {result.get('disease', '')}",
        f"Result      : {'Positive' if result.get('prediction') == 1 else 'Negative'}",
        f"Risk Level  : {result.get('risk_level', '')}",
        f"Risk Score  : {result.get('risk_percent', 0):.1f}%",
        f"Confidence  : {result.get('confidence', 0):.1f}%",
        "Input Values:",
    ]
    for k, v in result.get("features", {}).items():
        lines.append(f"  {labels.get(k, k)}: {round(float(v), 3)}")

    shap = result.get("shap_values", {})
    if shap:
        top5 = sorted(shap.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        lines.append("Top SHAP Drivers:")
        for k, v in top5:
            d = "increases risk" if v > 0 else "decreases risk"
            lines.append(f"  {labels.get(k, k)}: {v:+.5f}  ({d})")

    lines.append("--- END CONTEXT ---")
    return "\n".join(lines)
