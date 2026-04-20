"""
utils/pdf_export.py
===================
Generates a downloadable PDF report after every prediction.

Library: fpdf2  (pip install fpdf2)

IMPORTANT — Character encoding:
  fpdf2's built-in fonts (Helvetica, Times, Courier) only support
  Latin-1 characters. Avoid em-dashes (—), curly quotes, and emoji
  in any string passed to pdf.cell() or pdf.multi_cell().
  Use plain hyphens (-) and straight quotes instead.

Report layout:
  1. Header bar  (dark background, white text)
  2. Risk banner (colour-coded by risk level)
  3. Metrics row (Risk %, Confidence, Outcome, Risk Level)
  4. Feature values table
  5. Top SHAP drivers table
  6. Footer disclaimer
"""

import datetime


def generate_pdf(result: dict) -> bytes:
    """
    Build and return a PDF report for one prediction result.

    Args:
        result : the same dict saved to session_state["last_result"]
                 (disease, prediction, risk_percent, risk_level,
                  confidence, features, shap_values, labels)

    Returns:
        PDF file as bytes — pass directly to st.download_button(data=...)
    """
    try:
        from fpdf import FPDF
    except ImportError:
        return b"fpdf2 not installed. Run: pip install fpdf2"

    disease    = result.get("disease",    "")
    risk_pct   = result.get("risk_percent", 0)
    risk_label = result.get("risk_level",  "")
    confidence = result.get("confidence",  0)
    prediction = result.get("prediction",  0)
    features   = result.get("features",   {})
    labels     = result.get("labels",     {})
    shap_vals  = result.get("shap_values",{})

    # Colour mapping per risk level  (RGB tuples)
    bg_colors = {
        "Low Risk":       (220, 252, 231),
        "Moderate Risk":  (254, 249, 195),
        "High Risk":      (254, 215, 170),
        "Very High Risk": (254, 226, 226),
    }
    text_colors = {
        "Low Risk":       (20,  83,  45),
        "Moderate Risk":  (113, 63,  18),
        "High Risk":      (124, 45,  18),
        "Very High Risk": (127, 29,  29),
    }

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # ── 1. Header ─────────────────────────────────────────────────
    pdf.set_fill_color(15, 23, 42)          # dark navy
    pdf.rect(0, 0, 210, 26, "F")
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(255, 255, 255)
    pdf.set_xy(10, 7)
    # NOTE: use plain hyphen, NOT em-dash (—) — fpdf built-in fonts don't support it
    pdf.cell(0, 12, "MediXAI - Prediction Report", ln=True)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(180, 180, 180)
    pdf.set_xy(10, 19)
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    pdf.cell(0, 6, f"Generated: {ts}  |  Educational use only", ln=True)
    pdf.ln(5)

    # ── 2. Risk banner ─────────────────────────────────────────────
    r, g, b   = bg_colors.get(risk_label,   (240, 240, 240))
    tr, tg, tb = text_colors.get(risk_label, (30,  30,  30))
    pdf.set_fill_color(r, g, b)
    pdf.set_text_color(tr, tg, tb)
    pdf.set_font("Helvetica", "B", 13)
    outcome = "POSITIVE" if prediction == 1 else "NEGATIVE"
    pdf.cell(
        0, 13,
        f"  {disease}  |  {outcome}  |  {risk_label}  |  Risk: {risk_pct:.1f}%",
        fill=True, ln=True,
    )
    pdf.ln(3)

    # ── 3. Metrics row ─────────────────────────────────────────────
    pdf.set_text_color(0, 0, 0)
    pdf.set_fill_color(248, 250, 252)
    for lbl, val in [
        ("Risk Score",  f"{risk_pct:.1f}%"),
        ("Confidence",  f"{confidence:.1f}%"),
        ("Outcome",     outcome),
        ("Risk Level",  risk_label),
    ]:
        pdf.cell(47, 14, "", border=1, fill=True)
        x, y = pdf.get_x() - 47, pdf.get_y()
        pdf.set_xy(x, y + 2)
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(47, 5, lbl, align="C")
        pdf.set_xy(x, y + 7)
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(15, 23, 42)
        pdf.cell(47, 6, val, align="C")
        pdf.set_xy(x + 47, y)
    pdf.ln(18)

    # ── 4. Feature values table ────────────────────────────────────
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 8, "Input Feature Values", ln=True)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(241, 245, 249)
    pdf.cell(100, 6, "Feature", border=1, fill=True)
    pdf.cell(90,  6, "Value",   border=1, fill=True, ln=True)
    pdf.set_font("Helvetica", "", 9)
    for i, (k, v) in enumerate(features.items()):
        lbl  = labels.get(k, k)
        fill = (i % 2 == 0)
        if fill:
            pdf.set_fill_color(248, 250, 252)
        pdf.cell(100, 6, f"  {lbl}",           border="B", fill=fill)
        pdf.cell(90,  6, f"  {round(float(v), 4)}", border="B", fill=fill, ln=True)

    # ── 5. SHAP drivers table ──────────────────────────────────────
    if shap_vals:
        pdf.ln(5)
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 8, "Top SHAP Risk Drivers", ln=True)
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_fill_color(241, 245, 249)
        pdf.cell(90, 6, "Feature",    border=1, fill=True)
        pdf.cell(45, 6, "SHAP Value", border=1, fill=True)
        pdf.cell(55, 6, "Impact",     border=1, fill=True, ln=True)
        pdf.set_font("Helvetica", "", 9)
        top_shap = sorted(shap_vals.items(), key=lambda x: abs(x[1]), reverse=True)[:8]
        for i, (k, v) in enumerate(top_shap):
            lbl    = labels.get(k, k)
            impact = "Increases risk" if v > 0 else "Decreases risk"
            fill   = (i % 2 == 0)
            if fill:
                pdf.set_fill_color(248, 250, 252)
            pdf.cell(90, 6, f"  {lbl}",     border="B", fill=fill)
            pdf.cell(45, 6, f"  {v:+.5f}",  border="B", fill=fill)
            pdf.cell(55, 6, f"  {impact}",  border="B", fill=fill, ln=True)

    # ── 6. Footer disclaimer ───────────────────────────────────────
    pdf.ln(6)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(150, 150, 150)
    pdf.multi_cell(
        0, 5,
        "This report is for educational purposes only and does not constitute "
        "a medical diagnosis. Always consult a qualified healthcare professional.",
    )

    return bytes(pdf.output())
