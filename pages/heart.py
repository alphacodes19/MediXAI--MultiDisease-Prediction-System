"""
pages/heart.py
==============
Heart Disease Risk Prediction page.

Model  : ExtraTreesClassifier  (best of 17 models trained in the notebook)
Dataset: UCI Heart Disease dataset (918 patients)
Features (14): 11 raw + 3 engineered
  Raw:        Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS,
              RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope
  Engineered: Age_MaxHR_ratio = Age / (MaxHR + 1)
              BP_Chol_ratio   = RestingBP / (Cholesterol + 1)
              OldpeakAbs      = abs(Oldpeak)

Why ExtraTrees?
  - Best accuracy in 17-model comparison (~91.3%)
  - Tree-based → no feature scaling required at inference
  - Native support for SHAP TreeExplainer (fast, exact)

LIME note:
  LIME requires a background distribution for perturbations.
  Save X_train.values as heart_lime_bg.npy from the notebook to enable LIME.
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import datetime

from utils.models   import (load_heart, HEART_ALL, HEART_RAW, HEART_LABELS,
                             extract_shap_values, classify_risk)
from utils.xai      import show_xai_tabs, show_result_banner
from utils.database  import save_prediction
from utils.pdf_export import generate_pdf

BASE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HEART_DIR = os.path.join(BASE, "models", "heart", "saved_models")


def show():
    st.title("❤️ Heart Disease Risk Prediction")
    st.caption("UCI Heart dataset · ExtraTrees · SHAP + LIME")

    with st.sidebar:
        show_lime = st.checkbox("Show LIME", value=False)
        n_samples = st.slider("LIME samples", 100, 500, 200, 50)

    if not os.path.exists(os.path.join(HEART_DIR, "extra_trees.pkl")):
        st.error("extra_trees.pkl not found in models/heart/saved_models/")
        st.info("Copy from your Part Two project → saved_models/extra_trees.pkl")
        return

    model, _, shap_exp, lime_exp = load_heart()
    ocr = st.session_state.get("ocr_heart", {})

    st.subheader("Enter Patient Values")
    c1, c2, c3 = st.columns(3)
    with c1:
        age    = st.number_input("Age",              1,   120, int(ocr.get("Age",      50)))
        sex    = st.selectbox("Sex", ["Male", "Female"])
        cp     = st.selectbox("Chest Pain", ["Typical Angina", "Atypical Angina",
                                              "Non-Anginal", "Asymptomatic"])
        rbp    = st.number_input("Resting BP (mmHg)", 50, 300, int(ocr.get("RestingBP", 120)))
    with c2:
        chol   = st.number_input("Cholesterol (mg/dL)", 0, 700, int(ocr.get("Cholesterol", 200)))
        fbs    = st.selectbox("Fasting BS > 120 mg/dL", ["No (0)", "Yes (1)"])
        ecg    = st.selectbox("Resting ECG", ["Normal", "ST-T Abnormal", "LV Hypertrophy"])
        maxhr  = st.number_input("Max Heart Rate",   60,  220, int(ocr.get("MaxHR",    150)))
    with c3:
        angina  = st.selectbox("Exercise Angina", ["No", "Yes"])
        oldpeak = st.number_input("ST Depression (Oldpeak)", 0.0, 10.0,
                                   float(ocr.get("Oldpeak", 1.0)), 0.1)
        slope   = st.selectbox("ST Slope", ["Upsloping", "Flat", "Downsloping"])

    # Encode categorical inputs
    sex_enc    = 0 if sex == "Male" else 1
    cp_enc     = ["Typical Angina", "Atypical Angina", "Non-Anginal", "Asymptomatic"].index(cp)
    fbs_enc    = 1 if "Yes" in fbs else 0
    ecg_enc    = ["Normal", "ST-T Abnormal", "LV Hypertrophy"].index(ecg)
    angina_enc = 1 if angina == "Yes" else 0
    slope_enc  = ["Upsloping", "Flat", "Downsloping"].index(slope)

    with st.expander("Clinical reference ranges"):
        st.markdown("""| Metric | Normal | Concern |
|---|---|---|
| Resting BP | < 120 mmHg | ≥ 130 |
| Cholesterol | < 200 mg/dL | ≥ 240 |
| ST Depression | 0 | > 2.0 (significant) |
| Max Heart Rate | 220 - Age (target) | Very low = deconditioning |""")

    if st.button("🔍 Predict", type="primary", use_container_width=True):
        raw = {
            "Age": float(age), "Sex": float(sex_enc), "ChestPainType": float(cp_enc),
            "RestingBP": float(rbp), "Cholesterol": float(chol), "FastingBS": float(fbs_enc),
            "RestingECG": float(ecg_enc), "MaxHR": float(maxhr),
            "ExerciseAngina": float(angina_enc), "Oldpeak": float(oldpeak),
            "ST_Slope": float(slope_enc),
        }
        # Compute the 3 engineered features
        raw["Age_MaxHR_ratio"] = raw["Age"] / (raw["MaxHR"] + 1)
        raw["BP_Chol_ratio"]   = raw["RestingBP"] / (raw["Cholesterol"] + 1)
        raw["OldpeakAbs"]      = abs(raw["Oldpeak"])

        arr   = np.array([raw[f] for f in HEART_ALL]).reshape(1, -1)
        proba = model.predict_proba(arr)[0]
        pred  = int(model.predict(arr)[0])
        prob  = float(proba[1])
        conf  = float(max(proba)) * 100
        risk_pct = prob * 100
        risk_label, color, icon = classify_risk(risk_pct)

        show_result_banner(risk_pct, risk_label, color, icon, "Heart Disease", pred, conf)

        shap_vals = {}
        if shap_exp:
            try:
                shap_vals = dict(zip(HEART_ALL, extract_shap_values(shap_exp, arr)))
            except Exception as e:
                st.warning(f"SHAP: {e}")

        lime_vals = {}
        if show_lime and lime_exp:
            with st.spinner("Computing LIME…"):
                try:
                    exp = lime_exp.explain_instance(
                        arr[0], model.predict_proba,
                        num_features=10, num_samples=n_samples, top_labels=1
                    )
                    lime_vals = dict(exp.as_list(label=1))
                except Exception as e:
                    st.warning(f"LIME: {e}")

        show_xai_tabs(shap_vals, lime_vals, HEART_LABELS, "Heart Disease", show_lime)

        result = {
            "disease":      "Heart Disease",
            "prediction":   pred,
            "risk_percent": risk_pct,
            "risk_level":   risk_label,
            "confidence":   conf,
            "features":     raw,
            "shap_values":  shap_vals,
            "labels":       HEART_LABELS,
        }
        st.session_state["last_result"] = result
        uid = st.session_state.get("user_id", 0)
        save_prediction(result, uid)
        st.success("✅ Result saved to history.")

        pdf   = generate_pdf(result)
        fname = f"MediXAI_Heart_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        st.download_button("📄 Download PDF Report", data=pdf,
                           file_name=fname, mime="application/pdf")

        with st.expander("Input values"):
            st.dataframe(
                pd.DataFrame([{"Feature": HEART_LABELS.get(k, k), "Value": round(v, 4)}
                               for k, v in raw.items()]),
                hide_index=True, use_container_width=True,
            )
