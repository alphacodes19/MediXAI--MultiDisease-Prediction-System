"""
utils/models.py
===============
Loads and caches all three ML models.
Also contains shared prediction utilities used by every page.

Why @st.cache_resource?
  Models are large files. Without caching they reload from disk on
  every page interaction. cache_resource loads each model ONCE when
  the app starts, then reuses the same object in memory — fast.

Model details:
  Diabetes    → RandomForestClassifier  (NHANES clinical data, 7 features)
  Heart       → ExtraTreesClassifier   (UCI Heart, 14 features incl. 3 engineered)
  Parkinson's → Best of XGB/RF/SVM     (Oxford voice biomarkers, MI-selected features)
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
import streamlit as st

# ─────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────
BASE      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIAB_DIR  = os.path.join(BASE, "models", "diabetes")
HEART_DIR = os.path.join(BASE, "models", "heart",      "saved_models")
PARK_DIR  = os.path.join(BASE, "models", "parkinsons", "saved_models")


# ─────────────────────────────────────────────────────────────────
# FEATURE DEFINITIONS
# (single source of truth — imported by pages that need them)
# ─────────────────────────────────────────────────────────────────
DIAB_FEATURES = ["RIDAGEYR","RIAGENDR","BMXBMI","DBP_mean","SBP_mean","LBXGLU","LBXGH"]
DIAB_LABELS   = {
    "RIDAGEYR":"Age (years)", "RIAGENDR":"Gender",      "BMXBMI":"BMI",
    "DBP_mean":"Diastolic BP","SBP_mean":"Systolic BP",
    "LBXGLU":  "Glucose (mg/dL)", "LBXGH":"HbA1c (%)",
}

# Heart raw = original UCI columns; Heart ENG = 3 features we engineer at predict time
HEART_RAW = ["Age","Sex","ChestPainType","RestingBP","Cholesterol",
             "FastingBS","RestingECG","MaxHR","ExerciseAngina","Oldpeak","ST_Slope"]
HEART_ENG = ["Age_MaxHR_ratio","BP_Chol_ratio","OldpeakAbs"]
HEART_ALL = HEART_RAW + HEART_ENG
HEART_LABELS = {
    "Age":"Age","Sex":"Sex","ChestPainType":"Chest Pain Type",
    "RestingBP":"Resting BP (mmHg)","Cholesterol":"Cholesterol (mg/dL)",
    "FastingBS":"Fasting BS > 120","RestingECG":"Resting ECG",
    "MaxHR":"Max Heart Rate","ExerciseAngina":"Exercise Angina",
    "Oldpeak":"ST Depression","ST_Slope":"ST Slope",
    "Age_MaxHR_ratio":"Age/MaxHR ratio","BP_Chol_ratio":"BP/Chol ratio",
    "OldpeakAbs":"|Oldpeak|",
}
# Parkinson feature names are loaded from feature_names.json at runtime
# because they depend on mutual-information selection done during training


# ─────────────────────────────────────────────────────────────────
# MODEL LOADERS
# ─────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading Diabetes model…")
def load_diabetes():
    """
    Returns: (model, scaler, shap_explainer, lime_explainer)

    SHAP  → TreeExplainer (exact, fast for Random Forests)
    LIME  → trained on real NHANES data for realistic perturbations
    """
    model  = joblib.load(os.path.join(DIAB_DIR, "diabetes_model.pkl"))
    scaler = joblib.load(os.path.join(DIAB_DIR, "scaler.pkl"))

    shap_exp = None
    try:
        import shap
        shap_exp = shap.TreeExplainer(model)
    except Exception:
        pass

    lime_exp = None
    try:
        from lime.lime_tabular import LimeTabularExplainer
        bg = _load_nhanes_bg(scaler)
        lime_exp = LimeTabularExplainer(
            training_data         = bg,
            feature_names         = [DIAB_LABELS[f] for f in DIAB_FEATURES],
            class_names           = ["No Diabetes", "Diabetes"],
            mode                  = "classification",
            discretize_continuous = True,
            random_state          = 42,
        )
    except Exception:
        pass

    return model, scaler, shap_exp, lime_exp


def _load_nhanes_bg(scaler) -> np.ndarray:
    """
    Load NHANES training data as LIME background distribution.
    The .xls file is actually CSV stored in an old Excel wrapper —
    we read it as plain CSV and parse manually.
    Falls back to zeros if the file is missing (LIME still works, just less accurate).
    """
    for fname in ("nhanes_diabetes_clean.xls", "nhanes_diabetes.csv"):
        fpath = os.path.join(DIAB_DIR, fname)
        if not os.path.exists(fpath):
            continue
        try:
            if fname.endswith(".xls"):
                df = pd.read_csv(fpath, header=None)
                df = df[0].str.split(",", expand=True)
                df.columns = df.iloc[0]
                df = df[1:].reset_index(drop=True)
                for col in DIAB_FEATURES:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
            else:
                df = pd.read_csv(fpath)

            avail = [c for c in DIAB_FEATURES if c in df.columns]
            if not avail:
                continue
            bg = pd.DataFrame({f: (df[f] if f in df.columns else 0.0)
                                for f in DIAB_FEATURES})
            bg = bg.dropna().head(2000)
            return scaler.transform(bg.values.astype(float))
        except Exception:
            continue

    return np.zeros((50, len(DIAB_FEATURES)))


@st.cache_resource(show_spinner="Loading Heart Disease model…")
def load_heart():
    """
    Returns: (model, None, shap_explainer, lime_explainer)

    ExtraTrees is tree-based → no scaling needed at inference time.
    Second return value is None (kept for API consistency with other loaders).
    LIME background: save heart_lime_bg.npy from training notebook for best results.
    """
    model = joblib.load(os.path.join(HEART_DIR, "extra_trees.pkl"))

    shap_exp = None
    try:
        import shap
        shap_exp = shap.TreeExplainer(model)
    except Exception:
        pass

    lime_exp = None
    try:
        from lime.lime_tabular import LimeTabularExplainer
        bg_path = os.path.join(HEART_DIR, "heart_lime_bg.npy")
        bg = np.load(bg_path) if os.path.exists(bg_path) else np.zeros((100, len(HEART_ALL)))
        lime_exp = LimeTabularExplainer(
            training_data         = bg,
            feature_names         = [HEART_LABELS.get(f, f) for f in HEART_ALL],
            class_names           = ["No Disease", "Heart Disease"],
            mode                  = "classification",
            discretize_continuous = True,
            random_state          = 42,
        )
    except Exception:
        pass

    return model, None, shap_exp, lime_exp


@st.cache_resource(show_spinner="Loading Parkinson's model…")
def load_parkinsons():
    """
    Returns: (model, scaler, shap_explainer, lime_explainer, feature_names)

    Feature names come from feature_names.json because mutual-information
    selection during training picks a subset of the 22 voice biomarkers.
    RobustScaler is used because voice data has significant outliers.
    """
    model  = joblib.load(os.path.join(PARK_DIR, "best_model.pkl"))
    scaler = joblib.load(os.path.join(PARK_DIR, "scaler.pkl"))
    with open(os.path.join(PARK_DIR, "feature_names.json")) as f:
        feat_names = json.load(f)

    shap_exp = None
    try:
        import shap
        if hasattr(model, "estimators_") or hasattr(model, "get_booster"):
            shap_exp = shap.TreeExplainer(model)
        else:
            bg       = np.zeros((50, len(feat_names)))
            shap_exp = shap.KernelExplainer(model.predict_proba, shap.kmeans(bg, 20))
    except Exception:
        pass

    lime_exp = None
    try:
        from lime.lime_tabular import LimeTabularExplainer
        bg_path = None
        for name in ("parkinsons_lime_bg.npy", "park_lime_bg.npy"):
            p = os.path.join(PARK_DIR, name)
            if os.path.exists(p):
                bg_path = p
                break
        bg = np.load(bg_path) if bg_path else np.zeros((100, len(feat_names)))
        lime_exp = LimeTabularExplainer(
            training_data         = bg,
            feature_names         = feat_names,
            class_names           = ["Healthy", "Parkinson's"],
            mode                  = "classification",
            discretize_continuous = True,
            random_state          = 42,
        )
    except Exception:
        pass

    return model, scaler, shap_exp, lime_exp, feat_names


# ─────────────────────────────────────────────────────────────────
# SHARED PREDICTION UTILITIES
# ─────────────────────────────────────────────────────────────────

def classify_risk(pct: float) -> tuple:
    """
    Map a risk percentage to a (label, hex_colour, emoji) tuple.
    Thresholds: <30 Low, 30-50 Moderate, 50-70 High, >70 Very High.
    """
    if pct < 30:  return "Low Risk",       "#22c55e", "🟢"
    if pct < 50:  return "Moderate Risk",  "#eab308", "🟡"
    if pct < 70:  return "High Risk",      "#f97316", "🟠"
    return              "Very High Risk",  "#ef4444", "🔴"


def extract_shap_values(explainer, X) -> np.ndarray:
    """
    Robustly extract class-1 SHAP values regardless of explainer return format.

    SHAP can return:
      - a list of arrays  (multi-class format, older versions)
      - a 3D array        (n_samples, n_features, n_classes)
      - a 2D array        (n_samples, n_features)
    We always want the values for class 1 (disease positive).
    """
    sv = explainer.shap_values(X)
    if isinstance(sv, list):
        return np.array(sv[1])[0]
    sv = np.array(sv)
    if sv.ndim == 3:
        return sv[0, :, 1]
    return sv[0]
