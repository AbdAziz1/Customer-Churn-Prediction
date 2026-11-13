import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import shap

from utils.shap_explainer import make_explainer, clean_feature_names
from utils.model_loader import load_artifact
from utils.data_prep import model_frame
from utils.styles import apply_styles

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "telco.csv"

apply_styles()

st.title("Batch Risk Segmentation")
st.caption("Score a batch of customers and group them into low, medium and high churn risk.")

artifact = load_artifact()
model = artifact["model"]
features = artifact["features"]
threshold = float(artifact["threshold"])

# Tenure Group and Total Services are calculated, so the batch needs the other 29 fields.
required = [f for f in features if f not in ("Tenure Group", "Total Services")]

# ── Data source ───────────────────────────────────────────
source = st.radio(
    "Data source",
    ["Project dataset (telco.csv)", "Upload a CSV"],
    horizontal=True,
)

if source == "Upload a CSV":
    uploaded = st.file_uploader("Customer CSV", type="csv")
    if uploaded is None:
        st.info("Upload a CSV that contains the original customer fields used by the model.")
        st.stop()
    raw = pd.read_csv(uploaded)
else:
    if not DATA_PATH.exists():
        st.error("telco.csv is missing from the data folder.")
        st.stop()
    raw = pd.read_csv(DATA_PATH)

missing = [c for c in required if c not in raw.columns]
if missing:
    st.error(f"The data is missing {len(missing)} required column(s): {', '.join(missing)}")
    st.stop()

# ── Risk tier cutoffs ─────────────────────────────────────
with st.expander("Risk tier cutoffs", expanded=False):
    c1, c2 = st.columns(2)
    low_cut = c1.slider("Medium risk starts at", 0.05, 0.95, 0.30, 0.01)
    high_cut = c2.slider("High risk starts at", 0.05, 0.95, round(threshold, 2), 0.01)
    st.caption(
        f"High risk defaults to the model's decision threshold ({threshold:.1%}), "
        "so every customer the model classifies as churn lands in the High tier."
    )

if low_cut >= high_cut:
    st.error("The Medium cutoff must be lower than the High cutoff.")
    st.stop()

# ── Score and segment ─────────────────────────────────────
X = model_frame(raw, features)
probs = model.predict_proba(X)[:, 1]

scored = raw.copy()
scored["Churn probability"] = probs
scored["Risk tier"] = np.select(
    [probs >= high_cut, probs >= low_cut], ["High", "Medium"], default="Low"
)

order = ["Low", "Medium", "High"]
counts = scored["Risk tier"].value_counts().reindex(order, fill_value=0)

st.subheader("Risk segments")
cols = st.columns(3)
for col, tier in zip(cols, order):
    share = counts[tier] / len(scored)
    col.metric(f"{tier} risk ({share:.0%})", f"{counts[tier]:,}")

agg = {
    "Customers": ("Risk tier", "size"),
    "Avg churn probability": ("Churn probability", "mean"),
}
if "Churn Value" in scored.columns:
    agg["Actual churn rate"] = ("Churn Value", "mean")

summary = scored.groupby("Risk tier").agg(**agg).reindex(order)
for name in ("Avg churn probability", "Actual churn rate"):
    if name in summary.columns:
        summary[name] = summary[name].map("{:.1%}".format)
st.dataframe(summary, use_container_width=True)

caption = (
    "Probabilities come from a model trained with balanced class weights, so they run higher "
    "than real-world churn rates. The tiers rank customers, they are not forecasts of the churn rate."
)
if source == "Project dataset (telco.csv)":
    caption += (
        " This dataset includes the rows the model was trained on, so the results look "
        "better than they would on new customers."
    )
st.caption(caption)

# ── Customer list ─────────────────────────────────────────
st.subheader("Customers by tier")
chosen = st.multiselect("Show tiers", order, default=["High"])
view = scored[scored["Risk tier"].isin(chosen)].sort_values("Churn probability", ascending=False)

id_cols = [c for c in ["Customer ID"] if c in view.columns]
detail_cols = [
    c for c in ["Contract", "Tenure in Months", "Monthly Charge", "Internet Type"]
    if c in view.columns
]
table = view[id_cols + detail_cols].copy()
table.insert(len(id_cols), "Risk tier", view["Risk tier"])
table.insert(len(id_cols) + 1, "Churn probability (%)", view["Churn probability"] * 100)

st.dataframe(
    table,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Churn probability (%)": st.column_config.ProgressColumn(
            "Churn probability", min_value=0, max_value=100, format="%.1f%%"
        )
    },
)

st.download_button(
    "Download scored customers (CSV)",
    data=view.to_csv(index=False).encode("utf-8"),
    file_name="scored_customers.csv",
    mime="text/csv",
)

# ── SHAP drivers by tier ──────────────────────────────────
st.subheader("What drives risk in each tier")

s1, s2 = st.columns(2)
shap_tier = s1.selectbox("Tier to explain", order, index=2)
sample_cap = s2.slider("Customers to explain (sample)", 50, 500, 200, 50)

tier_idx = scored.index[scored["Risk tier"] == shap_tier]

if len(tier_idx) == 0:
    st.info(f"No customers fall in the {shap_tier} tier with the current cutoffs.")
else:
    explain_raw = X.loc[tier_idx].sample(n=min(sample_cap, len(tier_idx)), random_state=42)
    background_raw = X.sample(n=min(100, len(X)), random_state=42)

    preprocessor = model.named_steps["preprocessor"]
    inner_model = model.named_steps["clf"].estimator_

    with st.spinner("Computing SHAP values..."):
        background = np.asarray(preprocessor.transform(background_raw))
        explain_processed = np.asarray(preprocessor.transform(explain_raw))
        explainer = make_explainer(artifact, background)
        # The additivity check is off here so one awkward row can't block the page.
        # The largest gap is reported below instead.
        shap_out = explainer(explain_processed, check_additivity=False)

    vals = np.asarray(shap_out.values)
    if vals.ndim == 3:
        vals = vals[:, :, 1]
    base = float(np.asarray(explainer.expected_value).reshape(-1)[0])
    gap = float(np.abs(
        base + vals.sum(axis=1) - inner_model.decision_function(explain_processed)
    ).max())

    names = clean_feature_names(preprocessor.get_feature_names_out())
    explanation = shap.Explanation(
        values=vals,
        base_values=np.full(len(vals), base),
        data=explain_processed,
        feature_names=names,
    )

    st.caption(
        f"Explaining {len(explain_raw):,} of {len(tier_idx):,} {shap_tier}-risk customers. "
        "Values are in raw model score (log-odds). Positive values push toward churn."
    )

    tab_bar, tab_swarm = st.tabs(["Bar plot", "Beeswarm"])

    with tab_bar:
        st.caption("The 15 features with the largest average push on this tier's risk score.")
        imp = pd.DataFrame({"Feature": names, "Mean |SHAP|": np.abs(vals).mean(axis=0)})
        imp = imp.nlargest(15, "Mean |SHAP|").sort_values("Mean |SHAP|")
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh(imp["Feature"], imp["Mean |SHAP|"], color="#2563EB", height=0.65)
        ax.set_xlabel("Mean absolute SHAP value (raw model score)", color="#475569")
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#CBD5E1")
        ax.tick_params(colors="#475569", length=0)
        ax.xaxis.grid(True, color="#E2E8F0", linewidth=0.8)
        ax.set_axisbelow(True)
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)

    with tab_swarm:
        st.caption(
            "Each dot is one customer. Dots to the right push toward churn, dots to the left "
            "push toward staying. Red means a high value of that feature and blue a low value. "
            "Values are the scaled model inputs, and for one-hot features red means the category applies."
        )
        try:
            plt.figure(figsize=(10, 6))
            shap.plots.beeswarm(explanation, max_display=15, show=False)
            fig = plt.gcf()
            st.pyplot(fig, clear_figure=True)
            plt.close(fig)
        except Exception as exc:
            st.warning(f"Beeswarm unavailable: {exc}")

    st.caption(
        f"Largest gap between SHAP's total and the model's score in this sample: {gap:.3f} "
        "(log-odds). Averages across many customers are far less sensitive to a gap like this "
        "than a single customer's explanation."
    )