import streamlit as st

from utils.model_loader import load_artifact
from utils.styles import apply_styles

apply_styles()

st.title("Model Overview")
st.caption("Saved model metadata and final test set performance.")

artifact = load_artifact()
features = artifact["features"]
model_name = artifact.get("model_name", "Weighted Gradient Boosting")

# ── Model summary ─────────────────────────────────────────
c1, c2, c3 = st.columns(3)
c1.metric("Algorithm", model_name)
c2.metric("Decision threshold", f'{artifact["threshold"]:.4f}')
c3.metric("Model inputs", len(features))

# ── Final test set performance ────────────────────────────
st.subheader("Final test set performance")
st.caption(f"**{model_name}**: final evaluation on the held-out test set.")

# Final values from the training notebook ("FINAL TEST SET PERFORMANCE").
metrics = {
    "Accuracy": 0.8332,
    "Precision": 0.6514,
    "Recall": 0.7995,
    "F1-score": 0.7179,
    "ROC-AUC": 0.9147,
    "PR-AUC": 0.7919,
}

metric_help = {
    "Accuracy": "Percentage of all customers classified correctly. "
                "Can be misleading when churn classes are imbalanced.",
    "Precision": "Of customers flagged as likely to churn, how many "
                 "actually churned. Higher precision means less wasted "
                 "retention outreach.",
    "Recall": "Of all customers who actually churned, how many the "
              "model identified. Higher recall means fewer missed "
              "churners.",
    "F1-score": "Balances precision and recall. Useful when both "
                "unnecessary outreach and missed churners matter.",
    "ROC-AUC": "Measures how well the model ranks churners above "
               "non-churners across different thresholds.",
    "PR-AUC": "Measures the precision-recall trade-off, particularly "
              "useful when churn is the minority class.",
}

cols = st.columns(3)
for i, (name, value) in enumerate(metrics.items()):
    cols[i % 3].metric(name, f"{value:.4f}", help=metric_help[name])

# ── What the numbers mean for customer churn ──────────────
precision, recall = metrics["Precision"], metrics["Recall"]

# Test-set churn rate, derived from the metrics above:
# 1 - accuracy = FP + FN, with TP = recall * rate, FP = TP * (1 - precision) / precision.
churn_rate = (1 - metrics["Accuracy"]) / (
    recall * (1 - precision) / precision + (1 - recall)
)

st.subheader("What these numbers mean for customer churn")

st.markdown(f"""
**Accuracy ({metrics["Accuracy"]:.1%}): overall correctness.**
About {metrics["Accuracy"] * 100:.0f} of every 100 customers are classified correctly,
whether they stay or leave. Only about {churn_rate:.0%} of customers churn, so a model
that predicted "nobody leaves" would already score about {1 - churn_rate:.0%}.
Read accuracy against that baseline, not on its own.

**Precision ({precision:.1%}): how trustworthy a churn flag is.**
Of every 100 customers the model flags, about {precision * 100:.0f} really would have left.
The other {100 - precision * 100:.0f} would have stayed anyway, so retention offers
sent to them are wasted spend.

**Recall ({recall:.1%}): how many leavers we catch.**
Of every 100 customers who actually leave, the model catches about {recall * 100:.0f}
and misses about {100 - recall * 100:.0f}. Missed customers leave without any retention attempt.

**F1-score ({metrics["F1-score"]:.3f}): the balance between the two.**
It sits between precision and recall, closer to the lower one, so it only rises
when the model is good at both catching churners and avoiding false alarms.

**ROC-AUC ({metrics["ROC-AUC"]:.3f}): ranking ability.**
Pick one customer who left and one who stayed at random. About
{metrics["ROC-AUC"] * 100:.0f}% of the time, the model gives the leaver the higher risk score.
This is why the model works well for prioritising a call list.

**PR-AUC ({metrics["PR-AUC"]:.3f}): quality of the flagged list.**
It tracks precision as you flag more customers. A model with no skill would score about
{churn_rate:.2f} here (the churn rate), so {metrics["PR-AUC"]:.2f} is a large improvement.
""")

churners = 1000 * churn_rate
caught = recall * churners
flagged = caught / precision

st.info(
    f"**Worked example: 1,000 customers.** About {churners:.0f} would leave. "
    f"The model flags about {flagged:.0f} customers, of which about {caught:.0f} are real churners "
    f"and about {flagged - caught:.0f} are false alarms. "
    f"About {churners - caught:.0f} churners are missed."
)