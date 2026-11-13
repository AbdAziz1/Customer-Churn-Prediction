import streamlit as st
from utils.styles import apply_styles  
from utils.model_loader import load_artifact

st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="📉",
    layout="wide",
)

apply_styles()   

artifact = load_artifact()

# ── Header ──────────────────────────────────────────────
st.title("Customer Churn Prediction")
st.caption(

    "Identify at-risk customers and understand what drives their churn, "
    "powered by a Weighted Gradient Boosting model."
)
st.divider()

with st.sidebar:
    st.subheader("Navigation")
    st.markdown("Use the pages to review the model or score an individual customer.")
    st.divider()
    st.metric("Model", artifact.get("model_name", "Saved model"))
    st.metric("Decision threshold", f'{artifact["threshold"]:.3f}')
    st.metric("Input features", len(artifact["features"]))

# ── Workflow cards ──────────────────────────────────────
st.subheader("Get started")

card1, card2, card3 = st.columns(3)

with card1:
    with st.container(border=True):
        st.markdown("#### 📊 Model Overview")
        st.write(
            "Review hold-out performance, the decision threshold, "
            "the input schema, and global feature importance."
        )
        st.page_link("pages/1_Model_Overview.py", label="Open Model Overview", icon="➡️")

with card2:
    with st.container(border=True):
        st.markdown("#### 🧑‍💼 Single Customer Prediction")
        st.write(
            "Enter a customer profile to get a churn probability, "
            "a classification, and a SHAP explanation."
        )
        st.page_link(
            "pages/2_Single_Customer_Prediction.py",
            label="Open Single Customer Prediction",
            icon="➡️",
        )


with card3:
    with st.container(border=True):
        st.markdown("#### 🧑‍💼 Batch Risk Segmentation")
        st.write(
            "Analyze customers in bulk to identify at-risk segments and understand the factors driving churn."
        )
        st.page_link(
            "pages/3_Batch_Risk_Segmentation.py",
            label="Open Batch Risk Segmentation",
            icon="➡️",
        )

st.caption(
    "The saved pipeline receives original, unencoded feature values. "
    "Imputation, scaling, and categorical encoding are handled by the fitted preprocessing steps."
)