"""
pages/diabetes.py
=================
Diabetes Risk Prediction page.

Model  : RandomForestClassifier
Dataset: NHANES (National Health and Nutrition Examination Survey)
Features (7): Age, Gender, BMI, Diastolic BP, Systolic BP, Glucose, HbA1c
Scaler : StandardScaler

Clinical thresholds used:
  Glucose  < 100       → Normal
  Glucose  100–125     → Pre-diabetic
  Glucose  ≥ 126       → Diabetic
  HbA1c    < 5.7%      → Normal
  HbA1c    5.7–6.4%    → Pre-diabetic
  HbA1c    ≥ 6.5%      → Diabetic

OCR autofill: if the user came from the OCR Report page and extracted
Glucose / HbA1c / BMI / BP values, they are pre-filled here via
st.session_state["ocr_diab"].
"""

import os
import numpy as np
import pandas as pd
import streamlit as st

from utils.models  import (load_diabetes, DIAB_FEATURES, DIAB_LABELS,
                            extract_shap_values, classify_risk)
from utils.xai     import show_xai_tabs, show_result_banner
from utils.database import save_prediction
from utils.pdf_export import generate_pdf

import datetime

BASE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIAB_DIR  = os.path.join(BASE, "models", "diabetes")


def show():
    """Entry point called by app.py when user selects Diabetes page."""
    st.title("🩸 Diabetes Risk Prediction")
    st.caption("NHANES dataset · RandomForest · SHAP + LIME")

    # Sidebar controls
    with st.sidebar:
        show_lime = st.checkbox("Show LIME", value=False)
        n_samples = st.slider("LIME samples", 100, 500, 200, 50)

    # Guard: check model file exists
    if not os.path.exists(os.path.join(DIAB_DIR, "diabetes_model.pkl")):
        st.error("diabetes_model.pkl not found in models/diabetes/")
        st.info("Copy the file from your diabetes_pred_xai project.")
        return

    model, scaler, shap_exp, lime_exp = load_diabetes()

    # Pre-fill from OCR if available
    ocr = st.session_state.get("ocr_diab", {})

    # ── Input form ────────────────────────────────────────────────
    st.subheader("Enter Patient Values")
    c1, c2 = st.columns(2)
    with c1:
        age    = st.number_input("Age (years)",         18, 100,  int(ocr.get("RIDAGEYR", 50)))
        gender = st.selectbox("Gender", ["Male (1)", "Female (2)"])
        bmi    = st.number_input("BMI",                 10.0, 70.0, float(ocr.get("BMXBMI",  26.0)), 0.1)
        dbp    = st.number_input("Diastolic BP (mmHg)", 40.0, 140.0, float(ocr.get("DBP_mean", 75.0)), 1.0)
    with c2:
        sbp     = st.number_input("Systolic BP (mmHg)", 70.0, 220.0, float(ocr.get("SBP_mean", 120.0)), 1.0)
        glucose = st.number_input("Fasting Glucose (mg/dL)", 40.0, 500.0, float(ocr.get("LBXGLU", 90.0)), 1.0)
        hba1c   = st.number_input("HbA1c (%)",          3.0,  20.0,  float(ocr.get("LBXGH",   5.2)), 0.1)

    with st.expander("Clinical reference ranges"):
        st.markdown("""| Metric | Normal | Pre-diabetic | Diabetic |
|---|---|---|---|
| Glucose | < 100 mg/dL | 100–125 | ≥ 126 |
| HbA1c | < 5.7% | 5.7–6.4% | ≥ 6.5% |
| BMI | 18.5–24.9 | 25–29.9 | ≥ 30 (obese) |""")

    # ── Predict ───────────────────────────────────────────────────
    if st.button("🔍 Predict", type="primary", use_container_width=True):
        gender_val = 1 if "Male" in gender else 2
        features   = [float(age), float(gender_val), float(bmi), float(dbp),
                      float(sbp), float(glucose), float(hba1c)]
        feat_dict  = dict(zip(DIAB_FEATURES, features))

        arr    = np.array(features).reshape(1, -1)
        scaled = scaler.transform(arr)
        proba  = model.predict_proba(scaled)[0]
        pred   = int(model.predict(scaled)[0])
        prob   = float(proba[1])
        conf   = float(max(proba)) * 100
        risk_pct = prob * 100
        risk_label, color, icon = classify_risk(risk_pct)

        show_result_banner(risk_pct, risk_label, color, icon, "Diabetes", pred, conf)

        # SHAP
        shap_vals = {}
        if shap_exp:
            try:
                shap_vals = dict(zip(DIAB_FEATURES, extract_shap_values(shap_exp, scaled)))
            except Exception as e:
                st.warning(f"SHAP: {e}")

        # LIME
        lime_vals = {}
        if show_lime and lime_exp:
            with st.spinner("Computing LIME…"):
                try:
                    exp = lime_exp.explain_instance(
                        scaled[0], model.predict_proba,
                        num_features=len(DIAB_FEATURES),
                        num_samples=n_samples, top_labels=1
                    )
                    lime_vals = dict(exp.as_list(label=1))
                except Exception as e:
                    st.warning(f"LIME: {e}")

        show_xai_tabs(shap_vals, lime_vals, DIAB_LABELS, "Diabetes", show_lime)

        # Build result dict and save
        result = {
            "disease":      "Diabetes",
            "prediction":   pred,
            "risk_percent": risk_pct,
            "risk_level":   risk_label,
            "confidence":   conf,
            "features":     feat_dict,
            "shap_values":  shap_vals,
            "labels":       DIAB_LABELS,
        }
        st.session_state["last_result"] = result
        uid = st.session_state.get("user_id", 0)
        save_prediction(result, uid)
        st.success("✅ Result saved to history.")

        # PDF download
        pdf = generate_pdf(result)
        fname = f"MediXAI_Diabetes_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        st.download_button("📄 Download PDF Report", data=pdf,
                           file_name=fname, mime="application/pdf")

        with st.expander("Input values"):
            st.dataframe(
                pd.DataFrame([{"Feature": DIAB_LABELS[k], "Value": v}
                               for k, v in feat_dict.items()]),
                hide_index=True, use_container_width=True,
            )
