"""
utils/xai.py
============
Reusable chart helpers for SHAP and LIME explanations.
All functions return PNG bytes (for st.image) or render directly via Streamlit.

XAI = Explainable AI
  SHAP → tells you HOW MUCH each feature pushed the prediction up or down
  LIME → locally approximates the model around one specific patient
"""

import io
import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ─────────────────────────────────────────────────────────────────
# SHAP CHARTS
# ─────────────────────────────────────────────────────────────────

def shap_waterfall_chart(shap_vals: dict, labels: dict, title: str) -> bytes:
    """
    Horizontal bar chart showing each feature's SHAP contribution.

    Red bars   → feature pushed prediction TOWARDS disease (increases risk)
    Green bars → feature pushed prediction AWAY from disease (decreases risk)
    Bar length  → how strongly that feature influenced this prediction

    Args:
        shap_vals : {feature_key: shap_value}  (from _get_shap_vals)
        labels    : {feature_key: display_name}
        title     : chart title string

    Returns:
        PNG image as bytes (pass directly to st.image)
    """
    items  = sorted(shap_vals.items(), key=lambda x: abs(x[1]), reverse=True)[:12]
    labs   = [labels.get(k, k) for k, _ in items]
    vals   = [v for _, v in items]
    colors = ["#dc2626" if v > 0 else "#16a34a" for v in vals]

    fig, ax = plt.subplots(figsize=(8, max(3.5, len(vals) * 0.48)))
    bars = ax.barh(labs, vals, color=colors, height=0.6, edgecolor="none")

    # Value labels on each bar
    for bar, val in zip(bars, vals):
        x = bar.get_width()
        ax.text(
            x + (0.001 if x >= 0 else -0.001),
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.4f}",
            va="center",
            ha="left" if x >= 0 else "right",
            fontsize=8,
        )

    ax.axvline(0, color="#94a3b8", linewidth=0.8)
    ax.set_xlabel("SHAP value", fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=8)
    ax.legend(
        handles=[
            mpatches.Patch(color="#dc2626", label="Increases risk"),
            mpatches.Patch(color="#16a34a", label="Decreases risk"),
        ],
        fontsize=8, loc="lower right",
    )
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def shap_importance_chart(shap_vals: dict, labels: dict, title: str) -> bytes:
    """
    Bar chart of mean absolute SHAP values = global feature importance.
    Unlike the waterfall, direction doesn't matter here — just overall impact size.
    """
    mag_vals = {k: abs(v) for k, v in shap_vals.items()}
    return shap_waterfall_chart(mag_vals, labels, title)


# ─────────────────────────────────────────────────────────────────
# LIME CHART
# ─────────────────────────────────────────────────────────────────

def lime_bar_chart(lime_vals: dict, title: str) -> bytes:
    """
    Horizontal bar chart for LIME feature weights.

    Orange bars → condition pushes prediction TOWARDS disease
    Blue bars   → condition pushes prediction AWAY from disease

    LIME works differently from SHAP:
      - It fits a simple linear model around this one patient
      - Each 'condition' is a value range (e.g. "Glucose > 110")
      - Weights show how that range affects the local linear model

    Args:
        lime_vals : {condition_string: weight}  (from lime.as_list)
        title     : chart title string

    Returns:
        PNG image as bytes
    """
    items  = sorted(lime_vals.items(), key=lambda x: abs(x[1]), reverse=True)[:12]
    labs   = [k for k, _ in items]
    vals   = [v for _, v in items]
    colors = ["#f97316" if v > 0 else "#0ea5e9" for v in vals]

    fig, ax = plt.subplots(figsize=(8, max(3.5, len(vals) * 0.48)))
    ax.barh(labs, vals, color=colors, height=0.6, edgecolor="none")
    ax.axvline(0, color="#94a3b8", linewidth=0.8)
    ax.set_xlabel("LIME weight", fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=7)
    ax.legend(
        handles=[
            mpatches.Patch(color="#f97316", label="Towards disease"),
            mpatches.Patch(color="#0ea5e9", label="Away from disease"),
        ],
        fontsize=8, loc="lower right",
    )
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ─────────────────────────────────────────────────────────────────
# STREAMLIT DISPLAY
# ─────────────────────────────────────────────────────────────────

def show_xai_tabs(shap_vals: dict, lime_vals: dict,
                  labels: dict, disease: str, show_lime: bool):
    """
    Render SHAP Waterfall / SHAP Importance / LIME tabs in Streamlit.
    Called from every prediction page after a prediction is made.
    """
    tab1, tab2, tab3 = st.tabs(["SHAP Waterfall", "SHAP Importance", "LIME"])

    with tab1:
        if shap_vals:
            st.image(
                shap_waterfall_chart(shap_vals, labels, f"SHAP — {disease}"),
                use_container_width=True,
            )
            st.caption("Red = increases risk  ·  Green = decreases risk")
        else:
            st.info("SHAP not available for this model.")

    with tab2:
        if shap_vals:
            st.image(
                shap_importance_chart(shap_vals, labels, f"Feature Importance — {disease}"),
                use_container_width=True,
            )
            st.caption("Longer bar = greater overall impact on this prediction")

    with tab3:
        if show_lime and lime_vals:
            st.image(
                lime_bar_chart(lime_vals, f"LIME — {disease}"),
                use_container_width=True,
            )
            st.caption("Orange = towards disease  ·  Blue = away from disease")
        elif not show_lime:
            st.info("Enable **Show LIME** in the sidebar to compute LIME explanations.")
        else:
            st.info("LIME not available. Check if lime_bg.npy exists for this model.")


def show_result_banner(risk_pct: float, risk_label: str, color: str,
                       icon: str, disease: str, prediction: int, confidence: float):
    """
    Coloured result banner shown at the top of every prediction result.
    Also draws a progress bar showing risk percentage.
    """
    outcome = "Positive ⚠️" if prediction == 1 else "Negative ✅"
    st.markdown(
        f"""<div style="background:{color}22;border-left:5px solid {color};
        border-radius:0 8px 8px 0;padding:12px 18px;margin:8px 0">
        <strong style="font-size:1.1rem">{icon} {disease} — {outcome}</strong><br>
        Risk Score: <strong>{risk_pct:.1f}%</strong> &nbsp;·&nbsp;
        {risk_label} &nbsp;·&nbsp; Confidence: {confidence:.1f}%
        </div>""",
        unsafe_allow_html=True,
    )
    st.progress(int(risk_pct))
