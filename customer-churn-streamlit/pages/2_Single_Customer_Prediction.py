import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from pathlib import Path
from utils.model_loader import load_artifact
from utils.data_prep import engineer_features, model_frame
from utils.prediction import predict_one
from utils.shap_explainer import explain_one, readable_label
from utils.styles import apply_styles     
from utils.styles import apply_styles, render_risk_card 


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "telco.csv"


apply_styles()

st.title("Single Customer Prediction")
st.caption("Enter the customer's profile to estimate churn probability and explain the result.")

artifact = load_artifact()
features = artifact["features"]
if not DATA_PATH.exists():
    st.error("telco.csv is missing from the data folder. It is needed to populate valid input choices and SHAP background data.")
    st.stop()

raw_data = pd.read_csv(DATA_PATH)
engineered = engineer_features(raw_data)
training_frame = model_frame(engineered, features)

# Build a reasonable default record from a real row, while allowing every field to be edited.
default_row = training_frame.iloc[0].copy()
numeric = training_frame.select_dtypes(include=["number"]).columns.tolist()
categorical = [c for c in features if c not in numeric]

st.info("All 31 model features are required. 'Tenure Group' and 'Total Services' are calculated automatically from tenure and service selections.")

AUTO_FIELDS = {"Tenure Group", "Total Services"}

FIELD_GROUPS = {
    "Customer demographics": [
        "Gender", "Age", "Under 30", "Senior Citizen", "Married",
        "Dependents", "Number of Dependents", "Referred a Friend",
        "Number of Referrals",
    ],
    "Contract & billing": [
        "Tenure in Months", "Offer", "Contract", "Paperless Billing",
        "Payment Method", "Monthly Charge",
    ],
    "Phone & internet": [
        "Phone Service", "Multiple Lines", "Avg Monthly Long Distance Charges",
        "Internet Service", "Internet Type", "Avg Monthly GB Download",
        "Unlimited Data",
    ],
    "Add-on services": [
        "Online Security", "Online Backup", "Device Protection Plan",
        "Premium Tech Support", "Streaming TV", "Streaming Movies",
        "Streaming Music",
    ],
}


def render_field(container, feature):
    series = training_frame[feature]
    if feature in numeric:
        clean = pd.to_numeric(series, errors="coerce").dropna()
        minimum = float(clean.min())
        maximum = float(clean.max())
        default = float(default_row[feature])
        if pd.api.types.is_integer_dtype(series.dtype):
            return container.number_input(
                feature, min_value=int(minimum), max_value=int(maximum),
                value=int(default), step=1
            )
        return container.number_input(
            feature, min_value=float(minimum), max_value=float(maximum),
            value=float(default), step=0.01, format="%.2f"
        )
    options = sorted(series.dropna().astype(str).unique().tolist())
    default = str(default_row[feature])
    index = options.index(default) if default in options else 0
    return container.selectbox(feature, options, index=index)


# Any feature not listed above falls into "Other" so nothing is dropped.
placed = {f for names in FIELD_GROUPS.values() for f in names}
sections = dict(FIELD_GROUPS)
leftover = [f for f in features if f not in placed and f not in AUTO_FIELDS]
if leftover:
    sections["Other"] = leftover

with st.form("customer_input_form"):
    st.subheader("Customer profile")
    values = {}
    for title, names in sections.items():
        names = [n for n in names if n in features and n not in AUTO_FIELDS]
        if not names:
            continue
        with st.container(border=True):
            st.markdown(f"**{title}**")
            columns = st.columns(3)
            for i, name in enumerate(names):
                values[name] = render_field(columns[i % 3], name)

    submitted = st.form_submit_button("Predict churn", type="primary", use_container_width=True)

if submitted:
    raw_customer = pd.DataFrame([values])
    customer = engineer_features(raw_customer)
    customer = model_frame(customer, features)
    result = predict_one(customer, artifact)

    st.divider()
    st.subheader("Prediction result")
    render_risk_card(result["probability"], result["threshold"], result["label"])

    # SHAP background is sampled from the original training feature space.
    background = training_frame.sample(n=min(100, len(training_frame)), random_state=42)
    
    try:
        explanation = explain_one(artifact, customer, background)


                # ── Alternative SHAP views ────────────────────────
        view_row = customer.iloc[0]
        view_labels = [
            readable_label(n, view_row, v)
            for n, v in zip(explanation.feature_names, explanation.data)
        ]
        view_values = np.asarray(explanation.values)
        view_base = float(np.asarray(explanation.base_values).reshape(-1)[0])

        st.subheader("More ways to read this prediction")
        tab_bar, tab_decision, tab_force = st.tabs(["Bar plot", "Decision plot", "Force plot"])

        with tab_bar:
            try:
                st.caption(
                    "The 15 strongest drivers ranked by size. Red pushes toward churn, "
                    "blue pushes toward staying. Values are in raw model score (log-odds)."
                )
                view = shap.Explanation(
                    values=view_values, base_values=view_base, feature_names=view_labels
                )
                plt.figure(figsize=(10, 6))
                shap.plots.bar(view, max_display=15, show=False)
                fig = plt.gcf()
                st.pyplot(fig, clear_figure=True)
                plt.close(fig)
            except Exception as exc:
                st.warning(f"Bar plot unavailable: {exc}")

        with tab_decision:
            try:
                st.caption(
                    "Starts at the average customer's probability at the top and adds each "
                    "feature's push in turn, ending at this customer's probability at the bottom. "
                    "The axis is in probability, so the path shows how risk builds up."
                )
                plt.figure(figsize=(10, 6))
                shap.decision_plot(
                    view_base,
                    view_values.reshape(1, -1),
                    feature_names=view_labels,
                    link="logit",
                    feature_display_range=slice(-1, -16, -1),
                    color_bar=False,
                    auto_size_plot=False,
                    show=False,
                )
                fig = plt.gcf()
                st.pyplot(fig, clear_figure=True)
                plt.close(fig)
            except Exception as exc:
                st.warning(f"Decision plot unavailable: {exc}")

        with tab_force:
            try:
                st.caption(
                    "All pushes on a single line. Red features push the probability up, "
                    "blue features push it down, and the bold number is this customer's "
                    "predicted probability. Hover over a segment to see its value."
                )
                force = shap.force_plot(
                    view_base,
                    view_values,
                    feature_names=view_labels,
                    link="logit",
                    matplotlib=False,
                )
                components.html(
                    f"<head>{shap.getjs()}</head><body>{force.html()}</body>",
                    height=200,
                    scrolling=True,
                )
            except Exception as exc:
                st.warning(f"Force plot unavailable: {exc}")

        impact = pd.DataFrame({
            "Feature": explanation.feature_names,
            "SHAP value": explanation.values,
        })
        impact["Direction"] = np.where(impact["SHAP value"] > 0, "Increases churn score", "Decreases churn score")
        impact["Absolute impact"] = impact["SHAP value"].abs()
        impact = impact.sort_values("Absolute impact", ascending=False).head(10)
        st.markdown("**Largest contributing features**")
        st.dataframe(impact[["Feature", "SHAP value", "Direction"]], use_container_width=True, hide_index=True)


    except Exception as exc:
        st.warning(f"The prediction completed, but SHAP explanation failed: {exc}")
        st.caption("Check the pinned dependency versions in requirements.txt.")

        # ── What-if scenarios ─────────────────────────────
    ACTIONABLE = [
        "Contract", "Payment Method", "Paperless Billing", "Offer",
        "Online Security", "Online Backup", "Device Protection Plan",
        "Premium Tech Support", "Streaming TV", "Streaming Movies",
        "Streaming Music", "Unlimited Data",
    ]
    INTERNET_ONLY = {
        "Online Security", "Online Backup", "Device Protection Plan",
        "Premium Tech Support", "Streaming TV", "Streaming Movies",
        "Streaming Music", "Unlimited Data",
    }
    has_internet = (
        "Internet Service" not in raw_customer.columns
        or str(raw_customer["Internet Service"].iloc[0]) == "Yes"
    )

    scenarios = []
    for feature in ACTIONABLE:
        if feature not in raw_customer.columns:
            continue
        if feature in INTERNET_ONLY and not has_internet:
            continue
        current = str(raw_customer[feature].iloc[0])
        for option in sorted(training_frame[feature].dropna().astype(str).unique()):
            if option == current:
                continue
            changed = raw_customer.copy()
            changed[feature] = option
            scenarios.append((feature, current, option, changed))

    st.subheader("What-if scenarios")
    if scenarios:
        batch = model_frame(pd.concat([s[3] for s in scenarios], ignore_index=True), features)
        new_prob = artifact["model"].predict_proba(batch)[:, 1]
        base_prob = result["probability"]
        threshold = result["threshold"]

        table = pd.DataFrame({
            "Feature": [s[0] for s in scenarios],
            "Current": [s[1] for s in scenarios],
            "Change to": [s[2] for s in scenarios],
            "New prob": new_prob,
            "Change": new_prob - base_prob,
        })
        best = (
            table.sort_values("Change")
            .groupby("Feature", as_index=False)
            .first()
            .sort_values("Change")
        )
        best = best[best["Change"] < 0].head(5)

        if best.empty:
            st.info("No single change to these fields lowers this customer's churn probability.")
        else:
            top_row = best.iloc[0]
            st.markdown(
                f"The single change with the largest effect is **{top_row['Feature']}: "
                f"{top_row['Current']} → {top_row['Change to']}**, which moves the churn "
                f"probability from {base_prob:.1%} to {top_row['New prob']:.1%}."
            )
            shown = pd.DataFrame({
                "Feature": best["Feature"],
                "Current": best["Current"],
                "Change to": best["Change to"],
                "New probability": best["New prob"].map("{:.1%}".format),
                "Change": (best["Change"] * 100).map("{:+.1f} pts".format),
                "Classified as": np.where(best["New prob"] >= threshold, "Churn", "No Churn"),
            })
            st.dataframe(shown, use_container_width=True, hide_index=True)
    else:
        st.info("No adjustable fields were found for what-if scenarios.")

    st.caption(
        "Each row changes one field and keeps everything else the same, including the "
        "monthly charge. It shows how the model's score responds, not a guaranteed effect "
        "of a real retention action. Internet add-ons are only tested for customers who have internet service."
    )
    

        # ── Compared with the typical customer ────────────
    NUMERIC_KEYS = [
        "Tenure in Months", "Monthly Charge", "Age", "Number of Referrals",
        "Avg Monthly GB Download", "Total Services",
    ]
    CATEGORY_KEYS = ["Contract", "Internet Type", "Payment Method", "Paperless Billing", "Offer"]
    this_row = customer.iloc[0]

    def fmt(x):
        return f"{x:,.0f}" if float(x).is_integer() else f"{x:,.2f}"

    st.subheader("Compared with the typical customer")

    num_rows = []
    for f in NUMERIC_KEYS:
        if f not in numeric:
            continue
        col = pd.to_numeric(training_frame[f], errors="coerce").dropna()
        value = float(this_row[f])
        # Percentile counts ties as half, so a value shared by many customers lands near its true rank.
        percentile = ((col < value).mean() + (col <= value).mean()) / 2 * 100
        num_rows.append({
            "Feature": f,
            "This customer": fmt(value),
            "Typical (median)": fmt(float(col.median())),
            "Percentile": float(percentile),
        })

    if num_rows:
        standout = max(num_rows, key=lambda r: abs(r["Percentile"] - 50))
        p = standout["Percentile"]
        direction = f"higher than {p:.0f}%" if p >= 50 else f"lower than {100 - p:.0f}%"
        st.markdown(
            f"This customer stands out most on **{standout['Feature']}** "
            f"({standout['This customer']} vs a typical {standout['Typical (median)']}), "
            f"which is {direction} of customers."
        )
        st.dataframe(
            pd.DataFrame(num_rows),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Percentile": st.column_config.ProgressColumn(
                    "Percentile", min_value=0, max_value=100, format="%.0f"
                )
            },
        )

    cat_rows = []
    for f in CATEGORY_KEYS:
        if f not in training_frame.columns:
            continue
        series = training_frame[f].astype(str).replace("nan", "None")
        value = str(this_row[f])
        cat_rows.append({
            "Feature": f,
            "This customer": value,
            "Customers with this value": float((series == value).mean() * 100),
            "Most common value": series.mode().iloc[0],
        })

    if cat_rows:
        st.dataframe(
            pd.DataFrame(cat_rows),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Customers with this value": st.column_config.ProgressColumn(
                    "Customers with this value", min_value=0, max_value=100, format="%.0f%%"
                )
            },
        )

    st.caption(
        "Percentile is the share of customers with a lower value (50 means typical). "
        "The comparison uses the full customer dataset. A customer far from typical is not "
        "necessarily at risk. Use the SHAP views above to see which differences the model acts on."
    )