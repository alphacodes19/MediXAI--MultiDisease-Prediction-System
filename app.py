"""
app.py  — MediXAI Entry Point
==============================
This file is intentionally SHORT (~80 lines).
All real logic lives in utils/ and pages/.

To run:
    streamlit run app.py

File map:
    app.py                      ← YOU ARE HERE  (just routing + auth guard)
    utils/
        database.py             ← SQLite: users, predictions, chat sessions
        models.py               ← Load diabetes/heart/parkinsons models (cached)
        xai.py                  ← SHAP + LIME chart helpers
        llm.py                  ← Groq API calls
        pdf_export.py           ← PDF report generation
    pages/
        diabetes.py             ← Diabetes prediction page
        heart.py                ← Heart disease prediction page
        parkinsons.py           ← Parkinson's prediction page
        bulk_csv.py             ← Batch CSV prediction page
        ocr.py                  ← OCR report upload + AI analysis
        history.py              ← History & tracker charts
        recommendations.py      ← SHAP-driven + LLM recommendations
        chatbot.py              ← Multi-session AI chatbot
"""

import streamlit as st
from utils.database import init_db, create_user, login_user, load_chat_sessions, create_chat_session

# ─────────────────────────────────────────────────────────────────
# CONFIGURATION  — edit these two lines
# ──────────────────────────────────────import os
import streamlit as st
from groq import Groq

# Get API key from Streamlit Secrets or env
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")

client = Groq(api_key=GROQ_API_KEY)
# Get a free key at https://console.groq.com (30 req/min, 14,400/day)
# ─────────────────────────────────────────────────────────────────
# STARTUP
# ─────────────────────────────────────────────────────────────────
init_db()   # creates medixai_history.db and all tables if they don't exist

st.set_page_config(
    page_title="MediXAI — Unified Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stSidebar"] { background: #0f172a; }
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
div[data-testid="stMetricValue"] { font-size: 1.8rem !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# LOGIN WALL  — show login/register screen if not authenticated
# ─────────────────────────────────────────────────────────────────
if "user_id" not in st.session_state:
    st.title("🏥 MediXAI — Multi-Disease Prediction")
    st.markdown("Please **login** or **register** to continue.")

    tab_login, tab_reg = st.tabs(["🔑 Login", "📝 Register"])

    with tab_login:
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login", type="primary", use_container_width=True):
            uid = login_user(username, password)
            if uid:
                st.session_state["user_id"]  = uid
                st.session_state["username"] = username.strip()
                # Load or create first chat session
                sessions = load_chat_sessions(uid)
                if sessions:
                    st.session_state["chat_session_id"] = sessions[0]["id"]
                    st.session_state["chat_messages"]   = sessions[0]["messages"]
                else:
                    sid = create_chat_session(uid)
                    st.session_state["chat_session_id"] = sid
                    st.session_state["chat_messages"]   = []
                st.rerun()
            else:
                st.error("Invalid username or password.")

    with tab_reg:
        new_user = st.text_input("Choose a username", key="reg_user")
        new_pass = st.text_input("Choose a password", type="password", key="reg_pass")
        if st.button("Register", type="primary", use_container_width=True):
            if len(new_user.strip()) < 3:
                st.error("Username must be at least 3 characters.")
            elif len(new_pass) < 4:
                st.error("Password must be at least 4 characters.")
            elif create_user(new_user, new_pass):
                st.success("Account created! Please login now.")
            else:
                st.error("Username already taken. Try another.")
    st.stop()

# ─────────────────────────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 MediXAI")
    st.markdown("*Unified Disease Prediction*")
    st.divider()
    st.markdown(f"👤 **{st.session_state.get('username', 'User')}**")
    if st.button("🚪 Logout", use_container_width=True):
        st.session_state.clear()
        st.rerun()
    st.divider()
    page = st.radio("Navigate", [
        "🏠 Home",
        "🩸 Diabetes",
        "❤️ Heart Disease",
        "🧠 Parkinson's",
        "📂 Bulk CSV Upload",
        "📄 OCR Report Upload",
        "📊 History & Tracker",
        "🥗 Recommendations",
        "💬 AI Health Assistant",
    ], label_visibility="collapsed")
    st.divider()
    st.caption("⚠️ Educational use only.\nNot a substitute for medical advice.")

# ─────────────────────────────────────────────────────────────────
# PAGE ROUTING  — import and call each page's show() function
# ─────────────────────────────────────────────────────────────────
if page == "🏠 Home":
    st.title("🏥 MediXAI — Unified Disease Prediction")
    st.markdown(
        "A single portal for all three disease models with **SHAP + LIME** explainability, "
        "bulk CSV prediction, OCR report autofill, prediction history, "
        "SHAP-driven recommendations, PDF export, and AI chatbot."
    )
    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.markdown("### 🩸 Diabetes\nNHANES · RandomForest · 7 features\n\nGlucose, HbA1c, BMI, BP, Age")
    c2.markdown("### ❤️ Heart Disease\nUCI · ExtraTrees · 14 features\n\nCholesterol, MaxHR, ST Depression")
    c3.markdown("### 🧠 Parkinson's\nOxford voice · Best model\n\nJitter, Shimmer, HNR, RPDE, PPE")
    st.divider()
    c4, c5, c6 = st.columns(3)
    c4.markdown("### 📊 History & Tracker\nAuto-saved per user\nTrend charts · Export CSV")
    c5.markdown("### 🥗 Recommendations\nSHAP-driven + Groq LLM\nDiet / Exercise / Lifestyle")
    c6.markdown("### 📄 PDF Export\nFull report download\nafter every prediction")
    st.divider()
    c7, c8, c9 = st.columns(3)
    c7.markdown("### 📂 Bulk CSV\nBatch predict hundreds\nof patients at once")
    c8.markdown("### 📄 OCR Report\nUpload lab image\nautofill + AI analysis")
    c9.markdown("### 💬 AI Chatbot\nGroq Llama 3 · Free\nMulti-session · Context-aware")
    st.caption("> For educational and research purposes only. Always consult a qualified healthcare professional.")

elif page == "🩸 Diabetes":
    from src.diabetes import show; show()

elif page == "❤️ Heart Disease":
    from src.heart import show; show()

elif page == "🧠 Parkinson's":
    from src.parkinsons import show; show()

elif page == "📂 Bulk CSV Upload":
    from src.bulk_csv import show; show()

elif page == "📄 OCR Report Upload":
    import src.ocr as ocr_page
    ocr_page.GROQ_API_KEY = GROQ_API_KEY
    ocr_page.show()

elif page == "📊 History & Tracker":
    from src.history import show; show()

elif page == "🥗 Recommendations":
    import src.recommendations as rec_page
    rec_page.GROQ_API_KEY = GROQ_API_KEY
    rec_page.show()

elif page == "💬 AI Health Assistant":
    import src.chatbot as chat_page
    chat_page.GROQ_API_KEY = GROQ_API_KEY
    chat_page.show()
