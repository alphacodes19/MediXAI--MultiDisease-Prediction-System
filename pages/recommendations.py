"""
pages/recommendations.py
========================
Personalised Recommendations page.

Two-tab approach:

Tab 1 — Evidence-Based Tips (instant, no API needed)
  Rule database: each rule is (feature_key, threshold_function, category, message_template).
  The patient's actual feature values are checked against thresholds.
  Matching rules produce personalised tips with the real numbers filled in.

Tab 2 — AI-Generated Plan (Groq Llama 3)
  The patient's values + top SHAP drivers are formatted into a structured prompt.
  Groq generates a full Diet / Exercise / Lifestyle plan that references the
  patient's specific numbers and explains what the SHAP values mean clinically.

Why combine both?
  Rule-based = instant, deterministic, always works offline.
  LLM = natural language, truly personalised, explains SHAP in plain English.
"""

import streamlit as st
from utils.llm import groq_call

import os
GROQ_API_KEY = ""   # set in app.py


# ─────────────────────────────────────────────────────────────────
# RULE DATABASE
# Format: (feature_key, condition_lambda, category, message_template)
# Use {v:.0f} or {v:.1f} as placeholder for the actual value.
# ─────────────────────────────────────────────────────────────────

_DIAB_RULES = [
    ("LBXGLU", lambda v: v >= 126, "diet",
     "🍽️ **Your fasting glucose ({v:.0f} mg/dL) is in the diabetic range.** "
     "Strictly limit simple carbohydrates — white rice, bread, sugar, fruit juices. "
     "Prioritise low-GI foods: oats, lentils, leafy greens, and whole grains."),
    ("LBXGLU", lambda v: 100 <= v < 126, "diet",
     "🥗 **Glucose ({v:.0f} mg/dL) is pre-diabetic.** "
     "Reduce refined carbs and sugary drinks. Include fibre-rich vegetables and legumes "
     "with every meal to slow glucose absorption."),
    ("LBXGH", lambda v: v >= 6.5, "diet",
     "📉 **HbA1c ({v:.1f}%) indicates diabetes.** "
     "Spread carbohydrate intake evenly across 3 meals. Aim for 45-60g carbs per meal."),
    ("BMXBMI", lambda v: v >= 30, "exercise",
     "🏃 **BMI ({v:.1f}) indicates obesity.** "
     "Start with 30 min brisk walking daily. Even 5-10% weight loss significantly "
     "improves insulin sensitivity."),
    ("BMXBMI", lambda v: 25 <= v < 30, "exercise",
     "🚶 **BMI ({v:.1f}) is overweight.** "
     "Aim for 30 min moderate exercise 5 days/week combined with resistance training."),
    ("SBP_mean", lambda v: v >= 130, "lifestyle",
     "🧂 **Systolic BP ({v:.0f} mmHg) is elevated.** "
     "Reduce sodium below 2,300 mg/day. The DASH diet is clinically proven to lower BP."),
    ("LBXGLU", lambda v: v > 90, "lifestyle",
     "😴 **Poor sleep raises cortisol and elevates blood glucose.** "
     "Aim for 7-9 hours of quality sleep."),
    ("RIDAGEYR", lambda v: v > 45, "lifestyle",
     "🩺 **At age {v:.0f}, annual HbA1c and glucose screening is strongly recommended** "
     "even without symptoms."),
]

_HEART_RULES = [
    ("Cholesterol", lambda v: v >= 240, "diet",
     "🫀 **Cholesterol ({v:.0f} mg/dL) is high.** "
     "Eat oily fish 2x/week, walnuts, avocado, olive oil. Limit saturated fat to < 7% of calories."),
    ("Cholesterol", lambda v: 200 <= v < 240, "diet",
     "🥦 **Cholesterol ({v:.0f} mg/dL) is borderline.** "
     "Increase soluble fibre (oats, beans, apples) to actively lower LDL."),
    ("RestingBP", lambda v: v >= 130, "diet",
     "🧂 **Resting BP ({v:.0f} mmHg) is elevated.** "
     "Restrict sodium to < 2,300 mg/day. The DASH diet can lower systolic BP by 8-14 mmHg."),
    ("MaxHR", lambda v: v < 120, "exercise",
     "🚴 **Low max heart rate ({v:.0f}) suggests deconditioning.** "
     "Start with 20-30 min gentle walking daily, gradually increasing intensity."),
    ("Oldpeak", lambda v: v > 2.0, "lifestyle",
     "⚕️ **ST depression ({v:.1f}) is clinically significant.** "
     "Seek urgent cardiologist review before starting any new exercise programme."),
    ("Cholesterol", lambda v: v >= 200, "exercise",
     "🏋️ **Resistance training 2-3x/week** raises HDL and lowers LDL. "
     "Combine with 150 min/week aerobic exercise."),
    ("RestingBP", lambda v: v >= 120, "lifestyle",
     "🧘 **Chronic stress raises blood pressure.** "
     "Practice 10 min daily deep breathing or mindfulness."),
]

_PARK_RULES = [
    (None, None, "exercise",
     "🥊 **Regular vigorous aerobic exercise** (boxing, cycling, running) has strong evidence "
     "for slowing Parkinson's motor symptom progression. Aim for 30-60 min, 3x/week."),
    (None, None, "exercise",
     "🧘 **Tai chi and yoga** significantly improve balance, flexibility, and fall prevention. "
     "Even 2 sessions/week shows measurable benefit."),
    (None, None, "diet",
     "🥦 **Mediterranean diet** (olive oil, fish, vegetables, legumes, whole grains) "
     "is associated with lower Parkinson's risk via anti-inflammatory effects."),
    (None, None, "lifestyle",
     "🎵 **Rhythmic auditory stimulation** (music therapy, dancing) helps with gait "
     "and motor coordination — the brain responds to rhythm in ways that bypass motor deficits."),
    (None, None, "lifestyle",
     "💊 **If on levodopa, avoid large protein meals at medication time** — "
     "protein competes with levodopa absorption. Take medication 30 min before meals."),
]

DISEASE_RULES = {
    "Diabetes":            _DIAB_RULES,
    "Heart Disease":       _HEART_RULES,
    "Parkinson's Disease": _PARK_RULES,
}


def _get_rule_recs(result: dict) -> dict:
    """Match patient values against rule database. Returns {category: [tip_strings]}."""
    disease  = result.get("disease", "")
    features = result.get("features", {})
    rules    = DISEASE_RULES.get(disease, [])
    output   = {"diet": [], "exercise": [], "lifestyle": []}

    for feat_key, condition, category, msg_template in rules:
        if feat_key is None:
            output[category].append(msg_template)
        elif feat_key in features:
            try:
                val = float(features[feat_key])
                if condition(val):
                    tip = (msg_template
                           .replace("{v:.0f}", f"{val:.0f}")
                           .replace("{v:.1f}", f"{val:.1f}")
                           .replace("{v:.3f}", f"{val:.3f}"))
                    output[category].append(tip)
            except Exception:
                pass

    return output


def _shap_prompt(result: dict) -> str:
    """Build the structured prompt that gets sent to Groq for AI recommendations."""
    disease    = result.get("disease", "")
    risk_pct   = result.get("risk_percent", 0)
    risk_level = result.get("risk_level", "")
    features   = result.get("features", {})
    labels     = result.get("labels", {})
    shap_vals  = result.get("shap_values", {})

    feat_lines = "\n".join(
        f"  {labels.get(k, k)}: {round(float(v), 3)}" for k, v in features.items()
    )
    shap_lines = ""
    if shap_vals:
        top5 = sorted(shap_vals.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        shap_lines = "\nTop SHAP risk drivers:\n" + "\n".join(
            f"  {labels.get(k, k)}: {v:+.5f} "
            f"({'increases risk' if v > 0 else 'decreases risk'})"
            for k, v in top5
        )

    return (
        f"Patient: {disease} | {risk_level} ({risk_pct:.1f}% risk)\n\n"
        f"Feature values:\n{feat_lines}\n{shap_lines}\n\n"
        "Generate a personalised health plan with:\n"
        "1. DIET - 3 specific recommendations based on the highest-impact features\n"
        "2. EXERCISE - 2-3 appropriate exercises for this patient\n"
        "3. LIFESTYLE - 2-3 changes addressing the top risk drivers\n\n"
        "Reference the patient's ACTUAL numbers throughout. "
        "Be warm and encouraging. End with a reminder to consult a doctor.\n"
        "Format clearly with Diet, Exercise, Lifestyle headings."
    )


def show():
    st.title("🥗 Personalised Recommendations")
    st.caption("SHAP-driven rules + Groq LLM - Tailored to your exact values")

    api_key = st.session_state.get("api_key", GROQ_API_KEY)
    result  = st.session_state.get("last_result")

    if not result:
        st.warning("No prediction found. Run a prediction first.")
        st.info("Go to Diabetes, Heart Disease, or Parkinson's in the sidebar.")
        return

    disease    = result.get("disease", "")
    risk_pct   = result.get("risk_percent", 0)
    risk_level = result.get("risk_level", "")
    _, color, icon = (("", "#22c55e", "🟢") if risk_pct < 30 else
                      ("", "#eab308", "🟡") if risk_pct < 50 else
                      ("", "#f97316", "🟠") if risk_pct < 70 else
                      ("", "#ef4444", "🔴"))

    st.markdown(
        f"""<div style="background:{color}22;border-left:5px solid {color};
        border-radius:0 8px 8px 0;padding:10px 16px;margin-bottom:12px">
        <strong>{icon} Based on: {disease} - {risk_level} ({risk_pct:.1f}%)</strong>
        </div>""",
        unsafe_allow_html=True,
    )

    # Show SHAP drivers upfront
    shap_vals = result.get("shap_values", {})
    labels    = result.get("labels", {})
    if shap_vals:
        st.subheader("🔍 Your Top Risk Drivers (from SHAP)")
        top5 = sorted(shap_vals.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        for k, v in top5:
            direction = "🔴 Increases risk" if v > 0 else "🟢 Decreases risk"
            feat_val  = result.get("features", {}).get(k, "")
            try:
                fval_str = f"`{round(float(feat_val), 3)}`"
            except Exception:
                fval_str = "N/A"
            st.markdown(
                f"- **{labels.get(k, k)}** = {fval_str} - {direction} (SHAP: {v:+.4f})"
            )
        st.divider()

    tab_rules, tab_ai = st.tabs(["📋 Evidence-Based Tips", "🤖 AI-Generated Plan (Groq)"])

    with tab_rules:
        rule_recs = _get_rule_recs(result)
        if not any(rule_recs.values()):
            st.info("No specific rule-based tips for this profile.")

        for category, emoji, header in [
            ("diet",      "🥗", "Diet Recommendations"),
            ("exercise",  "🏃", "Exercise Plan"),
            ("lifestyle", "🌿", "Lifestyle Tips"),
        ]:
            tips = rule_recs.get(category, [])
            if tips:
                st.subheader(f"{emoji} {header}")
                for tip in tips:
                    st.markdown(tip)
                    st.markdown("")

        st.divider()
        st.caption(
            "These are evidence-based guidelines, not personal medical advice. "
            "Always consult a qualified healthcare professional."
        )

    with tab_ai:
        st.markdown(
            "The AI reads your **exact values and SHAP drivers** to generate "
            "a fully personalised plan referencing your actual numbers."
        )
        if not api_key:
            st.warning(
                "Enter your Groq API key in the AI Health Assistant sidebar, "
                "or paste it into GROQ_API_KEY in app.py."
            )
        else:
            if st.button("✨ Generate My Personalised Plan", type="primary",
                         use_container_width=True):
                system = (
                    "You are a preventive health advisor and nutritionist. "
                    "Generate specific, personalised, medically accurate recommendations. "
                    "Always reference the patient's actual numbers. "
                    "End with a reminder to consult a healthcare professional."
                )
                with st.spinner("Generating your personalised plan…"):
                    reply = groq_call(_shap_prompt(result), system, api_key)
                st.markdown(reply)
                st.caption(
                    "AI-generated content is for educational purposes only. "
                    "Consult a qualified healthcare professional for medical decisions."
                )
