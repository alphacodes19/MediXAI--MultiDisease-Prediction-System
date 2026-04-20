"""
pages/ocr.py
============
OCR Report Upload page.

What it does:
  1. User uploads a lab report image (PNG/JPG/PDF)
  2. Tesseract OCR extracts all text
  3. Regex patterns scan the text for health values
     (Glucose, HbA1c, BMI, Age, BP, Cholesterol, Heart Rate, ST Depression)
  4. AI (Groq) automatically analyses the full report text
  5. Analysis is saved as a 'report' chat session
  6. Extracted numeric values can be used to autofill Diabetes/Heart forms

Dependencies:
  pip install pytesseract pillow pdf2image
  + Tesseract binary from https://github.com/UB-Mannheim/tesseract/wiki
"""

import io
import re
import streamlit as st

from utils.database import create_chat_session, save_chat_messages
from utils.llm      import groq_call

import os
BASE         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GROQ_API_KEY = ""   # set in app.py

# ── Regex patterns for extracting values from OCR text ────────────
_OCR_PATTERNS = [
    ("LBXGLU",    [r"(?:fasting\s+(?:blood\s+)?(?:glucose|sugar)|fbs|rbs|blood\s+glucose)[^\d]*(\d+\.?\d*)",
                   r"glucose\s*[:\-]\s*(\d+\.?\d*)"]),
    ("LBXGH",     [r"hba1c\s*[:\-]\s*(\d+\.?\d*)", r"a1c\s*[:\-]\s*(\d+\.?\d*)",
                   r"glycated[^\d]*[:\-]\s*(\d+\.?\d*)"]),
    ("BMXBMI",    [r"bmi\s*[:\-]\s*(\d+\.?\d*)", r"body\s+mass\s+index\s*[:\-]\s*(\d+\.?\d*)"]),
    ("RIDAGEYR",  [r"\bage\b\s*[:\-]\s*(\d+)"]),
    ("SBP_mean",  [r"systolic\s*[:\-]\s*(\d+)", r"sbp\s*[:\-]\s*(\d+)",
                   r"(\d{2,3})\s*/\s*\d{2,3}\s*mm\s*hg"]),
    ("DBP_mean",  [r"diastolic\s*[:\-]\s*(\d+)", r"dbp\s*[:\-]\s*(\d+)",
                   r"\d{2,3}\s*/\s*(\d{2,3})\s*mm\s*hg"]),
    ("Cholesterol",[r"(?:total\s+)?cholesterol\s*[:\-]\s*(\d+\.?\d*)", r"chol\s*[:\-]\s*(\d+\.?\d*)"]),
    ("RestingBP", [r"resting\s+(?:blood\s+pressure|bp)\s*[:\-]\s*(\d+)"]),
    ("MaxHR",     [r"(?:max(?:imum)?\s+)?heart\s+rate\s*[:\-]\s*(\d+)", r"\bhr\b\s*[:\-]\s*(\d+)"]),
    ("Oldpeak",   [r"(?:st\s+depression|oldpeak)\s*[:\-]\s*(\d+\.?\d*)"]),
    ("Age",       [r"\bage\b\s*[:\-]\s*(\d+)"]),
]
_OCR_FRIENDLY = {
    "LBXGLU":"Glucose (mg/dL)", "LBXGH":"HbA1c (%)", "BMXBMI":"BMI",
    "RIDAGEYR":"Age (years)", "SBP_mean":"Systolic BP (mmHg)", "DBP_mean":"Diastolic BP (mmHg)",
    "Cholesterol":"Cholesterol (mg/dL)", "RestingBP":"Resting BP (mmHg)",
    "MaxHR":"Max Heart Rate", "Oldpeak":"ST Depression", "Age":"Age",
}


def _run_ocr(file_bytes: bytes, filename: str) -> str:
    try:
        from PIL import Image
        import pytesseract
        if filename.lower().endswith(".pdf"):
            try:
                from pdf2image import convert_from_bytes
                pages = convert_from_bytes(file_bytes, first_page=1, last_page=1)
                if not pages: return "[No pages found in PDF]"
                img = pages[0]
            except ImportError:
                return "[pdf2image not installed: pip install pdf2image]"
        else:
            img = Image.open(io.BytesIO(file_bytes))
        return pytesseract.image_to_string(img, config="--psm 6")
    except ImportError:
        return "[pytesseract not installed: pip install pytesseract]"
    except Exception as e:
        return f"[OCR error: {e}]"


def _parse_ocr(text: str) -> dict:
    tl  = text.lower()
    out = {}
    for key, patterns in _OCR_PATTERNS:
        if key in out:
            continue
        for pat in patterns:
            m = re.search(pat, tl, re.IGNORECASE)
            if m:
                try:
                    out[key] = float(m.group(1))
                    break
                except (ValueError, IndexError):
                    continue
    return out


def show():
    api_key = st.session_state.get("api_key", GROQ_API_KEY)

    st.title("📄 Upload Medical Report — OCR Autofill")
    st.caption("Upload a lab report → AI analyses it → Autofill prediction forms.")
    st.info("Works best with clear printed text at 300+ DPI. Supports PNG, JPG, PDF.")

    uploaded = st.file_uploader(
        "Upload lab report",
        type=["png", "jpg", "jpeg", "pdf", "tiff", "bmp"],
    )
    if uploaded is None:
        st.markdown("**Detectable values:** Glucose · HbA1c · BMI · Age · BP · Cholesterol · Heart Rate · ST Depression")
        return

    file_bytes = uploaded.read()
    if uploaded.name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
        st.image(file_bytes, caption="Uploaded report", use_container_width=True)

    with st.spinner("Running OCR…"):
        raw_text = _run_ocr(file_bytes, uploaded.name)

    if not raw_text or raw_text.startswith("["):
        st.error(raw_text or "OCR returned no text.")
        st.info("Install Tesseract: https://github.com/UB-Mannheim/tesseract/wiki\n"
                "Then: `pip install pytesseract pillow`")
        return

    # ── AI analysis of full report ─────────────────────────────────
    report_prompt = (
        f"I have uploaded a health report. Here is the extracted text:\n\n{raw_text}\n\n"
        "Please analyse this report: identify the key health metrics, explain what they mean, "
        "highlight any values that are outside normal ranges, and give actionable health advice."
    )
    with st.spinner("AI is analysing your report…"):
        reply = groq_call(
            report_prompt,
            "You are a medical report analyst. Analyse the report, explain each metric clearly, "
            "flag abnormal values, and give practical advice. "
            "Always remind the user to consult a healthcare professional.",
            api_key,
        )

    # Save as a report-type chat session
    uid = st.session_state.get("user_id", 0)
    report_msgs = [
        {"role": "user",      "content": f"[Uploaded health report]\n\n{raw_text}"},
        {"role": "assistant", "content": reply},
    ]
    sid = create_chat_session(uid, title=f"Report: {uploaded.name[:20]}", chat_type="report")
    save_chat_messages(sid, report_msgs)
    st.session_state["chat_session_id"] = sid
    st.session_state["chat_messages"]   = report_msgs

    st.subheader("🤖 AI Report Analysis")
    st.markdown(reply)
    st.info("💬 Go to **AI Health Assistant** in the sidebar to continue this conversation.")

    # ── Autofill extracted values ──────────────────────────────────
    extracted = _parse_ocr(raw_text)
    if extracted:
        st.divider()
        st.subheader("Extracted Values — Autofill")
        st.success(f"✅ {len(extracted)} value(s) extracted — review before autofilling:")
        edited = {}
        cols = st.columns(3)
        for i, (key, val) in enumerate(extracted.items()):
            edited[key] = cols[i % 3].number_input(
                _OCR_FRIENDLY.get(key, key), value=float(val), step=0.1, key=f"ocr_{key}"
            )

        diab_keys  = {"RIDAGEYR", "BMXBMI", "DBP_mean", "SBP_mean", "LBXGLU", "LBXGH"}
        heart_keys = {"Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"}
        c1, c2 = st.columns(2)
        with c1:
            if bool(diab_keys & set(edited)):
                if st.button("🩸 Autofill Diabetes Form", type="primary"):
                    st.session_state["ocr_diab"] = {k: v for k, v in edited.items() if k in diab_keys}
                    st.success("Done — go to 🩸 Diabetes in the sidebar.")
            else:
                st.info("No diabetes-relevant fields detected.")
        with c2:
            if bool(heart_keys & set(edited)):
                if st.button("❤️ Autofill Heart Form", type="primary"):
                    st.session_state["ocr_heart"] = {k: v for k, v in edited.items() if k in heart_keys}
                    st.success("Done — go to ❤️ Heart Disease in the sidebar.")
            else:
                st.info("No heart-relevant fields detected.")

    with st.expander("📜 Raw OCR text (debug)"):
        st.text(raw_text)
