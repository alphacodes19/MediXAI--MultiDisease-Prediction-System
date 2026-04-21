"""
pages/bulk_csv.py
=================
Bulk CSV Prediction page.

Allows uploading a CSV of multiple patients and running predictions on all rows at once.

Workflow:
  1. User selects a disease model
  2. Downloads the CSV template (correct column names pre-filled)
  3. Fills in their patient data and uploads it
  4. App predicts every row and shows a summary + downloadable results CSV

Template columns match exactly what each model expects:
  Diabetes    → 7 NHANES feature columns
  Heart       → 11 raw UCI columns (3 engineered columns added automatically)
  Parkinson's → MI-selected features (loaded from feature_names.json)
"""

import os
import json
import numpy as np
import pandas as pd
import streamlit as st

from utils.models  import (load_diabetes, load_heart, load_parkinsons,
                            DIAB_FEATURES, HEART_RAW, HEART_ALL, classify_risk)
from utils.database import save_prediction

BASE     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARK_DIR = os.path.join(BASE, "models", "parkinsons", "saved_models")


def show():
    st.title("📂 Bulk CSV Prediction")
    st.caption("Upload a CSV with multiple patients — get batch risk scores instantly.")

    disease = st.selectbox("Select disease model",
                           ["Diabetes", "Heart Disease", "Parkinson's Disease"])

    # Build template for selected disease
    if disease == "Diabetes":
        template_cols = DIAB_FEATURES
        template_vals = {"RIDAGEYR": 50, "RIAGENDR": 1, "BMXBMI": 28.0,
                         "DBP_mean": 78.0, "SBP_mean": 125.0, "LBXGLU": 105.0, "LBXGH": 5.8}
    elif disease == "Heart Disease":
        template_cols = HEART_RAW
        template_vals = {"Age": 50, "Sex": 0, "ChestPainType": 1, "RestingBP": 130,
                         "Cholesterol": 220, "FastingBS": 0, "RestingECG": 0,
                         "MaxHR": 155, "ExerciseAngina": 0, "Oldpeak": 1.0, "ST_Slope": 1}
    else:
        fn_path = os.path.join(PARK_DIR, "feature_names.json")
        if not os.path.exists(fn_path):
            st.error("feature_names.json not found in models/parkinsons/saved_models/")
            return
        with open(fn_path) as f:
            pk_feats = json.load(f)
        template_cols = pk_feats
        template_vals = {f: 0.0 for f in pk_feats}

    # Step 1: template download
    st.subheader("Step 1 — Download template")
    st.download_button(
        "📥 Download CSV Template",
        data=pd.DataFrame([template_vals]).to_csv(index=False).encode(),
        file_name=f"template_{disease.lower().replace(' ', '_')}.csv",
        mime="text/csv",
    )
    st.caption(f"Required columns: `{', '.join(template_cols)}`")

    # Step 2: upload
    st.subheader("Step 2 — Upload your CSV")
    uploaded = st.file_uploader("Upload filled CSV", type=["csv"])
    if uploaded is None:
        return

    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        return

    st.write(f"Loaded **{len(df)} rows** × {len(df.columns)} columns")
    st.dataframe(df.head(3), use_container_width=True)

    missing = [c for c in template_cols if c not in df.columns]
    if missing:
        st.error(f"Missing columns in your CSV: {missing}")
        return

    save_to_history = st.checkbox("Save all predictions to history", value=False)

    if not st.button("🚀 Run Batch Predictions", type="primary"):
        return

    results = []
    prog    = st.progress(0)
    status  = st.empty()
    uid     = st.session_state.get("user_id", 0)

    # ── Diabetes batch ────────────────────────────────────────────
    if disease == "Diabetes":
        model, scaler, _, __ = load_diabetes()
        for i, row in df.iterrows():
            status.text(f"Row {i+1}/{len(df)}…")
            try:
                feats = [float(row[f]) for f in DIAB_FEATURES]
                sc    = scaler.transform(np.array(feats).reshape(1, -1))
                prob  = float(model.predict_proba(sc)[0][1])
                pred  = int(model.predict(sc)[0])
                rl, _, __ = classify_risk(prob * 100)
                results.append({**{f: row[f] for f in DIAB_FEATURES},
                                 "Prediction": "Positive" if pred == 1 else "Negative",
                                 "Risk %": round(prob * 100, 1), "Risk Level": rl})
                if save_to_history:
                    save_prediction({"disease": "Diabetes", "prediction": pred,
                                     "risk_percent": prob*100, "risk_level": rl,
                                     "confidence": float(max(model.predict_proba(sc)[0]))*100,
                                     "features": {f: row[f] for f in DIAB_FEATURES},
                                     "shap_values": {}}, uid)
            except Exception as e:
                results.append({**{f: row.get(f, "") for f in DIAB_FEATURES},
                                 "Prediction": "Error", "Risk %": 0, "Risk Level": str(e)})
            prog.progress((i + 1) / len(df))

    # ── Heart batch ───────────────────────────────────────────────
    elif disease == "Heart Disease":
        model, _, shap_exp, __ = load_heart()
        for i, row in df.iterrows():
            status.text(f"Row {i+1}/{len(df)}…")
            try:
                raw = {c: float(row[c]) for c in HEART_RAW}
                raw["Age_MaxHR_ratio"] = raw["Age"] / (raw["MaxHR"] + 1)
                raw["BP_Chol_ratio"]   = raw["RestingBP"] / (raw["Cholesterol"] + 1)
                raw["OldpeakAbs"]      = abs(raw["Oldpeak"])
                arr  = np.array([raw[f] for f in HEART_ALL]).reshape(1, -1)
                prob = float(model.predict_proba(arr)[0][1])
                pred = int(model.predict(arr)[0])
                rl, _, __ = classify_risk(prob * 100)
                results.append({**{f: row[f] for f in HEART_RAW},
                                 "Prediction": "Positive" if pred == 1 else "Negative",
                                 "Risk %": round(prob * 100, 1), "Risk Level": rl})
                if save_to_history:
                    save_prediction({"disease": "Heart Disease", "prediction": pred,
                                     "risk_percent": prob*100, "risk_level": rl,
                                     "confidence": float(max(model.predict_proba(arr)[0]))*100,
                                     "features": raw, "shap_values": {}}, uid)
            except Exception as e:
                results.append({**{f: row.get(f, "") for f in HEART_RAW},
                                 "Prediction": "Error", "Risk %": 0, "Risk Level": str(e)})
            prog.progress((i + 1) / len(df))

    # ── Parkinson's batch ─────────────────────────────────────────
    else:
        model, scaler, _, __, pk_feats = load_parkinsons()
        for i, row in df.iterrows():
            status.text(f"Row {i+1}/{len(df)}…")
            try:
                vals   = {f: float(row[f]) for f in pk_feats}
                df_raw = pd.DataFrame(np.array([vals[f] for f in pk_feats]).reshape(1,-1), columns=pk_feats)
                df_sc  = pd.DataFrame(scaler.transform(df_raw), columns=pk_feats)
                prob   = float(model.predict_proba(df_sc)[0][1])
                pred   = int(model.predict(df_sc)[0])
                rl, _, __ = classify_risk(prob * 100)
                results.append({**{f: row[f] for f in pk_feats},
                                 "Prediction": "Positive" if pred == 1 else "Negative",
                                 "Risk %": round(prob * 100, 1), "Risk Level": rl})
            except Exception as e:
                results.append({**{f: row.get(f, "") for f in pk_feats},
                                 "Prediction": "Error", "Risk %": 0, "Risk Level": str(e)})
            prog.progress((i + 1) / len(df))

    status.text("✅ Done!")
    res_df = pd.DataFrame(results)

    # Summary
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Patients", len(res_df))
    c2.metric("Positive",       int((res_df["Prediction"] == "Positive").sum()))
    c3.metric("Avg Risk %",     f"{res_df['Risk %'].mean():.1f}%")

    st.bar_chart(res_df["Risk Level"].value_counts())
    st.dataframe(res_df[["Prediction", "Risk %", "Risk Level"]], use_container_width=True)

    st.download_button(
        "📥 Download Results CSV",
        data=res_df.to_csv(index=False).encode(),
        file_name=f"results_{disease.lower().replace(' ', '_')}.csv",
        mime="text/csv",
    )
