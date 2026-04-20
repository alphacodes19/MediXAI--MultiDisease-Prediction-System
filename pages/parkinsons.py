"""
pages/parkinsons.py
===================
Parkinson's Disease Risk Prediction page.

Model  : Best model selected from XGBoost / RandomForest / SVM + ensembles
         (saved as best_model.pkl — the winning model from notebook comparison)
Dataset: Oxford Parkinson's Voice dataset (195 patients, 22 voice biomarkers)
Scaler : RobustScaler  (chosen because voice data has significant outliers)

Feature selection:
  Mutual Information (MI) selects the top 75% most informative features.
  The exact list is saved in feature_names.json at training time.
  This page reads that file and only shows inputs for selected features.

Engineered features added before MI selection:
  Freq_Range       = MDVP:Fhi(Hz) - MDVP:Flo(Hz)   (frequency spread)
  Freq_Variability = MDVP:Fhi(Hz) / MDVP:Fo(Hz)    (relative frequency change)
  NHR_HNR_ratio    = NHR / HNR                       (noise-to-harmonic ratio)

Feature groups explained:
  Frequency → fundamental voice frequency (Hz)
  Jitter    → variation in fundamental frequency (cycle-to-cycle)
  Shimmer   → variation in amplitude (loudness)
  Noise     → NHR (noise-to-harmonic), HNR (harmonic-to-noise)
  Nonlinear → RPDE, DFA, spread1/2, D2, PPE (complexity measures)
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import datetime

from utils.models    import load_parkinsons, extract_shap_values, classify_risk
from utils.xai       import show_xai_tabs, show_result_banner
from utils.database  import save_prediction
from utils.pdf_export import generate_pdf

BASE     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARK_DIR = os.path.join(BASE, "models", "parkinsons", "saved_models")

# Sample values from a real Parkinson's patient in the Oxford dataset
# Used as default inputs so users can test immediately
SAMPLE_VALUES = {
    "MDVP:Fo(Hz)":    119.99, "MDVP:Fhi(Hz)": 157.30, "MDVP:Flo(Hz)": 74.99,
    "MDVP:Jitter(%)":  0.00784, "MDVP:Jitter(Abs)": 0.00007,
    "MDVP:RAP":        0.00370,  "MDVP:PPQ":   0.00554, "Jitter:DDP": 0.01109,
    "MDVP:Shimmer":    0.04374,  "MDVP:Shimmer(dB)": 0.42600,
    "Shimmer:APQ3":    0.02182,  "Shimmer:APQ5": 0.03130,
    "MDVP:APQ":        0.02971,  "Shimmer:DDA": 0.06545,
    "NHR":             0.02211,  "HNR":   21.033, "RPDE": 0.41400,
    "DFA":             0.81560,  "spread1": -4.81322, "spread2": 0.31656,
    "D2":              2.30113,  "PPE":   0.28477,
    # Engineered features
    "Freq_Range":      82.31, "Freq_Variability": 1.31, "NHR_HNR_ratio": 0.00105,
}


def show():
    st.title("🧠 Parkinson's Disease Risk Prediction")
    st.caption("Oxford voice dataset · Best model · SHAP + LIME")

    with st.sidebar:
        show_lime = st.checkbox("Show LIME", value=False)
        n_samples = st.slider("LIME samples", 100, 500, 200, 50)

    if not os.path.exists(os.path.join(PARK_DIR, "best_model.pkl")):
        st.error("best_model.pkl not found in models/parkinsons/saved_models/")
        st.info("Run Cell 62 in parkinson_prediction_enhanced_XAI.ipynb to generate saved_models/")
        return

    model, scaler, shap_exp, lime_exp, PARK_FEATURES = load_parkinsons()

    st.info(
        f"Using **{len(PARK_FEATURES)} features** selected by mutual information "
        f"(out of 25 total = 22 voice biomarkers + 3 engineered)."
    )

    use_sample = st.checkbox("Load sample values (Parkinson's patient)", value=True)

    # Group features for cleaner UI layout
    GROUPS = {
        "Frequency (Hz)":   [f for f in PARK_FEATURES if any(x in f for x in ["Fo", "Fhi", "Flo", "Freq"])],
        "Jitter":           [f for f in PARK_FEATURES if any(x in f for x in ["Jitter", "jitter", "RAP", "PPQ", "DDP"])],
        "Shimmer":          [f for f in PARK_FEATURES if any(x in f for x in ["Shimmer", "shimmer", "APQ", "DDA"])],
        "Noise / Harmonic": [f for f in PARK_FEATURES if f in ("NHR", "HNR", "NHR_HNR_ratio")],
        "Nonlinear":        [f for f in PARK_FEATURES if f in ("RPDE", "DFA", "spread1", "spread2", "D2", "PPE")],
    }

    values = {}
    for group, feats in GROUPS.items():
        if not feats:
            continue
        st.subheader(group)
        cols = st.columns(min(len(feats), 3))
        for i, feat in enumerate(feats):
            default = SAMPLE_VALUES.get(feat, 0.0) if use_sample else 0.0
            step    = 0.001 if abs(default) < 1 else 0.01
            values[feat] = cols[i % 3].number_input(
                feat, value=float(default), format="%.5f", step=step, key=f"pk_{feat}"
            )

    # Any features not caught by the groups above
    ungrouped = [f for f in PARK_FEATURES if f not in values]
    if ungrouped:
        st.subheader("Other features")
        cols = st.columns(3)
        for i, feat in enumerate(ungrouped):
            default = SAMPLE_VALUES.get(feat, 0.0) if use_sample else 0.0
            values[feat] = cols[i % 3].number_input(
                feat, value=float(default), format="%.5f", key=f"pk_{feat}"
            )

    if st.button("🔍 Predict", type="primary", use_container_width=True):
        arr_raw = np.array([values[f] for f in PARK_FEATURES]).reshape(1, -1)
        df_raw  = pd.DataFrame(arr_raw, columns=PARK_FEATURES)
        arr_sc  = scaler.transform(df_raw)
        df_sc   = pd.DataFrame(arr_sc, columns=PARK_FEATURES)

        proba = model.predict_proba(df_sc)[0]
        pred  = int(model.predict(df_sc)[0])
        prob  = float(proba[1])
        conf  = float(max(proba)) * 100
        risk_pct = prob * 100
        risk_label, color, icon = classify_risk(risk_pct)

        show_result_banner(risk_pct, risk_label, color, icon, "Parkinson's Disease", pred, conf)

        shap_vals = {}
        if shap_exp:
            try:
                shap_vals = dict(zip(PARK_FEATURES, extract_shap_values(shap_exp, df_sc)))
            except Exception as e:
                st.warning(f"SHAP: {e}")

        lime_vals = {}
        if show_lime and lime_exp:
            with st.spinner("Computing LIME…"):
                try:
                    exp = lime_exp.explain_instance(
                        arr_sc[0], model.predict_proba,
                        num_features=min(10, len(PARK_FEATURES)),
                        num_samples=n_samples, top_labels=1
                    )
                    lime_vals = dict(exp.as_list(label=1))
                except Exception as e:
                    st.warning(f"LIME: {e}")

        park_labels = {f: f for f in PARK_FEATURES}
        show_xai_tabs(shap_vals, lime_vals, park_labels, "Parkinson's Disease", show_lime)

        result = {
            "disease":      "Parkinson's Disease",
            "prediction":   pred,
            "risk_percent": risk_pct,
            "risk_level":   risk_label,
            "confidence":   conf,
            "features":     values,
            "shap_values":  shap_vals,
            "labels":       park_labels,
        }
        st.session_state["last_result"] = result
        uid = st.session_state.get("user_id", 0)
        save_prediction(result, uid)
        st.success("✅ Result saved to history.")

        pdf   = generate_pdf(result)
        fname = f"MediXAI_Parkinsons_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        st.download_button("📄 Download PDF Report", data=pdf,
                           file_name=fname, mime="application/pdf")

        with st.expander("Input values"):
            st.dataframe(
                pd.DataFrame([{"Feature": k, "Value": round(v, 5)} for k, v in values.items()]),
                hide_index=True, use_container_width=True,
            )
