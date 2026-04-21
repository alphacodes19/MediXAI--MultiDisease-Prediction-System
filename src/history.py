"""
pages/history.py
================
History & Tracker page.

Shows all predictions made by the logged-in user, with:
  - Summary metrics (total, positive count, avg risk, max risk)
  - Risk % over time line chart (Plotly if installed, Matplotlib fallback)
  - Risk level pie chart
  - Predictions by disease bar chart
  - Full table of all records
  - CSV export
  - Delete by ID
"""

import io
import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.database import load_history, delete_record


def show():
    st.title("📊 History & Tracker")
    st.caption("All your predictions — auto-saved, per user, with trend charts.")

    try:
        import plotly.express as px
        PLOTLY = True
    except ImportError:
        PLOTLY = False

    uid = st.session_state.get("user_id", 0)

    # Filters
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        disease_filter = st.selectbox("Filter by disease",
                                      ["All", "Diabetes", "Heart Disease", "Parkinson's Disease"])
    with col_f2:
        limit = st.slider("Max records to show", 20, 500, 100, 20)

    disease_arg = None if disease_filter == "All" else disease_filter
    records     = load_history(user_id=uid, disease=disease_arg, limit=limit)

    if not records:
        st.info("No predictions saved yet. Make a prediction — results are saved automatically.")
        return

    df = pd.DataFrame([{
        "ID":        r["id"],
        "Date":      r["ts"],
        "Disease":   r["disease"],
        "Result":    "Positive" if r["prediction"] == 1 else "Negative",
        "Risk %":    r["risk_pct"],
        "Risk Level":r["risk_level"],
        "Confidence":r["confidence"],
    } for r in records])

    # Summary metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Predictions", len(df))
    c2.metric("Positive",          int((df["Result"] == "Positive").sum()))
    c3.metric("Avg Risk %",        f"{df['Risk %'].mean():.1f}%")
    c4.metric("Max Risk %",        f"{df['Risk %'].max():.1f}%")

    st.divider()
    st.subheader("📈 Risk % Over Time")
    df["Date"] = pd.to_datetime(df["Date"])

    if PLOTLY:
        fig = px.line(df.sort_values("Date"), x="Date", y="Risk %",
                      color="Disease", markers=True,
                      color_discrete_sequence=["#3b82f6", "#ef4444", "#a855f7"])
        fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig, ax = plt.subplots(figsize=(10, 3.5))
        for dis in df["Disease"].unique():
            sub = df[df["Disease"] == dis].sort_values("Date")
            ax.plot(sub["Date"], sub["Risk %"], marker="o", label=dis, linewidth=1.5)
        ax.set_ylabel("Risk %")
        ax.legend()
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        st.image(buf.read(), use_container_width=True)

    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("Risk Level Distribution")
        rc        = df["Risk Level"].value_counts()
        COLOR_MAP = {"Low Risk": "#22c55e", "Moderate Risk": "#eab308",
                     "High Risk": "#f97316", "Very High Risk": "#ef4444"}
        if PLOTLY:
            fig2 = px.pie(values=rc.values, names=rc.index, hole=0.4,
                          color=rc.index, color_discrete_map=COLOR_MAP)
            fig2.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.bar_chart(rc)

    with col_r:
        st.subheader("Predictions by Disease")
        st.bar_chart(df["Disease"].value_counts())

    st.divider()
    st.subheader("All Records")
    df_show              = df.copy()
    df_show["Date"]      = df_show["Date"].dt.strftime("%Y-%m-%d %H:%M")
    df_show["Risk %"]    = df_show["Risk %"].map("{:.1f}%".format)
    df_show["Confidence"]= df_show["Confidence"].map("{:.1f}%".format)
    st.dataframe(df_show, hide_index=True, use_container_width=True)

    col_exp, col_del = st.columns(2)
    with col_exp:
        csv = pd.DataFrame([{
            "ID": r["id"], "Date": r["ts"], "Disease": r["disease"],
            "Prediction": "Positive" if r["prediction"] == 1 else "Negative",
            "Risk %": r["risk_pct"], "Risk Level": r["risk_level"],
            "Confidence": r["confidence"],
        } for r in records]).to_csv(index=False).encode()
        st.download_button("📥 Export History CSV", data=csv,
                           file_name="medixai_history.csv", mime="text/csv",
                           use_container_width=True)

    with col_del:
        del_id = st.number_input("Delete record by ID", min_value=1, step=1, value=1)
        if st.button("🗑️ Delete Record", type="secondary", use_container_width=True):
            delete_record(int(del_id))
            st.success(f"Record #{del_id} deleted.")
            st.rerun()
