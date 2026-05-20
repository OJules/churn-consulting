import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import joblib
import shap
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.inference import score_customers, assign_segment
from src.config import MODELS_DIR, TARGET

# ─── Config ───────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

# ─── Load data ────────────────────────────────────────────────────────────────
@st.cache_data
def load_scored_data():
    df = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "df_scored.csv")
    return df

@st.cache_resource
def load_model_artifacts():
    preprocessor     = joblib.load(MODELS_DIR / "preprocessor.pkl")
    model            = joblib.load(MODELS_DIR / "xgb_calibrated.pkl")
    explainer        = joblib.load(MODELS_DIR / "shap_explainer.pkl")
    X_train, X_test, y_train, y_test, feature_names = joblib.load(
        MODELS_DIR / "train_test_data.pkl"
    )
    return preprocessor, model, explainer, X_test, feature_names

df = load_scored_data()
preprocessor, model, explainer, X_test, feature_names = load_model_artifacts()

# ─── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/combo-chart.png", width=60)
st.sidebar.title("Churn Consulting")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    ["Executive Overview", "Segment Explorer", "Individual Scoring", "ROI Simulator"]
)
st.sidebar.markdown("---")
st.sidebar.caption("Telco Churn · XGBoost · AUC 0.84")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — EXECUTIVE OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "Executive Overview":
    st.title("📊 Executive Overview")
    st.markdown("Vue synthétique des indicateurs clés pour le management.")
    st.markdown("---")

    seg_a = df[df["Segment"] == "A — High Priority"]
    churn_rate = df[TARGET].mean()
    revenue_at_risk = seg_a["MonthlyCharges"].sum() * 12
    n_high_risk = len(seg_a)
    roi_20 = (revenue_at_risk * 0.20 - n_high_risk * 15) / (n_high_risk * 15) * 100

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Churn Rate Global", f"{churn_rate:.1%}", delta="-26% clients/an")
    col2.metric("Clients Haut Risque", f"{n_high_risk}", delta="Segment A")
    col3.metric("Revenu Annuel à Risque", f"${revenue_at_risk:,.0f}")
    col4.metric("ROI Campagne (20%)", f"{roi_20:.0f}%", delta="conversion estimée")

    st.markdown("---")
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Distribution du Churn")
        churn_counts = df[TARGET].value_counts().reset_index()
        churn_counts.columns = ["Churn", "Count"]
        churn_counts["Churn"] = churn_counts["Churn"].map({0: "No Churn", 1: "Churn"})
        fig = px.bar(churn_counts, x="Churn", y="Count",
                     color="Churn",
                     color_discrete_map={"No Churn": "#4A90D9", "Churn": "#E74C3C"})
        fig.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader("Répartition par Segment CRM")
        seg_counts = df["Segment"].value_counts().reset_index()
        seg_counts.columns = ["Segment", "Count"]
        colors = {"A — High Priority": "#E74C3C",
                  "B — Medium Priority": "#F39C12",
                  "C — Low Priority": "#4A90D9"}
        fig2 = px.pie(seg_counts, names="Segment", values="Count",
                      color="Segment", color_discrete_map=colors,
                      hole=0.4)
        fig2.update_layout(height=350)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.subheader("Top Insights")
    c1, c2, c3 = st.columns(3)
    c1.info("📋 **Contrat mensuel** → churn rate 42% vs 3% en contrat 2 ans")
    c2.warning("🌐 **Fiber optic** → 41% de churn, le service premium le plus risqué")
    c3.error("⏱️ **0–12 mois** → 48% de churn, fenêtre critique pour la rétention")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — SEGMENT EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Segment Explorer":
    st.title("🎯 Segment Explorer")
    st.markdown("Explorez la distribution des scores de risque par segment CRM.")
    st.markdown("---")

    col_f1, col_f2, col_f3 = st.columns(3)
    selected_segments = col_f1.multiselect(
        "Segments", df["Segment"].unique().tolist(),
        default=df["Segment"].unique().tolist()
    )
    contract_filter = col_f2.multiselect(
        "Type de contrat", df["Contract"].unique().tolist(),
        default=df["Contract"].unique().tolist()
    )
    internet_filter = col_f3.multiselect(
        "Internet Service", df["InternetService"].unique().tolist(),
        default=df["InternetService"].unique().tolist()
    )

    filtered = df[
        df["Segment"].isin(selected_segments) &
        df["Contract"].isin(contract_filter) &
        df["InternetService"].isin(internet_filter)
    ]
    st.caption(f"{len(filtered)} clients affichés")
    st.markdown("---")

    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Distribution ChurnProbability")
        fig = px.histogram(filtered, x="ChurnProbability", color="Segment",
                           nbins=40, barmode="overlay", opacity=0.7,
                           color_discrete_map={
                               "A — High Priority": "#E74C3C",
                               "B — Medium Priority": "#F39C12",
                               "C — Low Priority": "#4A90D9"
                           })
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader("CLV × ChurnProbability")
        fig2 = px.scatter(filtered, x="ChurnProbability", y="CLV",
                          color="Segment", opacity=0.5, size_max=8,
                          color_discrete_map={
                              "A — High Priority": "#E74C3C",
                              "B — Medium Priority": "#F39C12",
                              "C — Low Priority": "#4A90D9"
                          })
        fig2.update_layout(height=350)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.subheader("Statistiques par Segment")
    summary = filtered.groupby("Segment").agg(
        Clients=("ChurnProbability", "count"),
        Churn_Rate=(TARGET, "mean"),
        Avg_ChurnProba=("ChurnProbability", "mean"),
        Avg_CLV=("CLV", "mean"),
        Avg_RetentionPriority=("RetentionPriority", "mean"),
    ).round(3).reset_index()
    st.dataframe(summary, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — INDIVIDUAL SCORING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Individual Scoring":
    st.title("🔍 Individual Scoring")
    st.markdown("Analysez le profil de risque d'un client individuel.")
    st.markdown("---")

    col_sel, col_info = st.columns([1, 2])

    with col_sel:
        idx = st.number_input(
            "Index client (0 à {})".format(len(df) - 1),
            min_value=0, max_value=len(df) - 1, value=1
        )
        customer = df.iloc[idx]
        st.markdown("**Profil client**")
        st.write({
            "Contract": customer["Contract"],
            "Tenure": f"{customer['tenure']} mois",
            "MonthlyCharges": f"${customer['MonthlyCharges']:.2f}",
            "InternetService": customer["InternetService"],
            "PaymentMethod": customer["PaymentMethod"],
        })

    with col_info:
        churn_p = customer["ChurnProbability"]
        clv = customer["CLV"]
        segment = customer["Segment"]
        priority = customer["RetentionPriority"]

        color = "#E74C3C" if segment == "A — High Priority" else \
                "#F39C12" if segment == "B — Medium Priority" else "#27AE60"

        st.markdown(f"### Churn Probability : `{churn_p:.1%}`")
        st.progress(float(churn_p))

        c1, c2, c3 = st.columns(3)
        c1.metric("CLV", f"${clv:,.0f}")
        c2.metric("Retention Priority", f"{priority:,.0f}")
        c3.markdown(f"**Segment**<br><span style='color:{color}; font-size:18px'>{segment}</span>",
                    unsafe_allow_html=True)

        st.markdown("---")
        if segment == "A — High Priority":
            st.error("🚨 **Action immédiate recommandée** : appel commercial, offre de rétention personnalisée")
        elif segment == "B — Medium Priority":
            st.warning("⚠️ **Nurturing recommandé** : email ciblé, upgrade de service")
        else:
            st.success("✅ **Fidélisation standard** : programme de loyauté")

    st.markdown("---")
    st.subheader("Explication SHAP locale")

    from src.config import NUMERICAL_COLS, CATEGORICAL_COLS
    customer_raw = df.iloc[[idx]][NUMERICAL_COLS + CATEGORICAL_COLS].copy()
    customer_raw["SeniorCitizen"] = customer_raw["SeniorCitizen"].astype(str)
    X_customer = preprocessor.transform(customer_raw)
    shap_vals = explainer.shap_values(X_customer)

    fig, ax = plt.subplots(figsize=(10, 4))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_vals[0],
            base_values=explainer.expected_value,
            feature_names=feature_names
        ),
        max_display=10,
        show=False
    )
    st.pyplot(fig)
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — ROI SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "ROI Simulator":
    st.title("💰 ROI Simulator")
    st.markdown("Simulez le retour sur investissement d'une campagne de rétention ciblée.")
    st.markdown("---")

    seg_a = df[df["Segment"] == "A — High Priority"]
    revenue_at_risk_annual = seg_a["MonthlyCharges"].sum() * 12
    n_clients = len(seg_a)

    col_p1, col_p2 = st.columns(2)
    cost_per_client = col_p1.slider(
        "Coût par client contacté ($)", 5, 50, 15, step=5
    )
    target_segment = col_p2.selectbox(
        "Segment ciblé",
        ["A — High Priority", "A + B (tous à risque)"]
    )

    if target_segment == "A + B (tous à risque)":
        seg_target = df[df["Segment"].isin(["A — High Priority", "B — Medium Priority"])]
    else:
        seg_target = seg_a

    n_target = len(seg_target)
    rev_at_risk = seg_target["MonthlyCharges"].sum() * 12
    campaign_cost = n_target * cost_per_client

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    c1.metric("Clients ciblés", f"{n_target}")
    c2.metric("Revenu annuel à risque", f"${rev_at_risk:,.0f}")
    c3.metric("Coût campagne", f"${campaign_cost:,.0f}")

    st.markdown("---")
    st.subheader("Courbe ROI selon le taux de conversion")

    conversion_rates = np.arange(0.05, 0.55, 0.05)
    revenues_saved = rev_at_risk * conversion_rates
    rois = (revenues_saved - campaign_cost) / campaign_cost * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=conversion_rates * 100, y=rois,
        mode="lines+markers",
        line=dict(color="#4A90D9", width=3),
        marker=dict(size=8),
        name="ROI (%)"
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="red",
                  annotation_text="Break-even")
    fig.update_layout(
        xaxis_title="Taux de conversion (%)",
        yaxis_title="ROI (%)",
        title="ROI de la campagne de rétention",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Tableau détaillé")
    roi_df = pd.DataFrame({
        "Conversion (%)": (conversion_rates * 100).astype(int),
        "Revenu sauvé ($)": revenues_saved.astype(int),
        "Coût campagne ($)": int(campaign_cost),
        "Gain net ($)": (revenues_saved - campaign_cost).astype(int),
        "ROI (%)": rois.astype(int)
    })
    st.dataframe(roi_df, use_container_width=True)