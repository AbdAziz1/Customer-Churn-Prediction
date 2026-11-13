import numpy as np
import shap
import streamlit as st

def clean_feature_names(names):
    """Drop the transformer prefix (e.g. 'num__', 'cat__') from feature names."""
    return [str(n).split("__", 1)[-1] for n in names]

def readable_label(name, raw_row, processed_value):
    """Turn a processed feature name into 'Feature = value' using the customer's raw inputs.

    - Numeric features: 'Tenure in Months = 5'
    - One-hot features: 'Contract = Month-to-Month' if active, 'Contract ≠ Two Year' if not
    """
    if name in raw_row.index:
        return f"{name} = {raw_row[name]}"
    for col in sorted(raw_row.index, key=len, reverse=True):
        if name.startswith(col + "_"):
            category = name[len(col) + 1:]
            sign = "=" if processed_value >= 0.5 else "≠"
            return f"{col} {sign} {category}"
    return name

@st.cache_resource(show_spinner=False)
def make_explainer(_artifact, background: np.ndarray):
    pipeline = _artifact["model"]
    # Explain the fitted GradientBoostingClassifier inside the custom wrapper.
    inner_model = pipeline.named_steps["clf"].estimator_
    return shap.TreeExplainer(
        inner_model,
        data=background,
        feature_perturbation="interventional",
        model_output="raw",
    )


def explain_one(artifact: dict, raw_customer, background_raw):
    pipeline = artifact["model"]
    preprocessor = pipeline.named_steps["preprocessor"]
    inner_model = pipeline.named_steps["clf"].estimator_

    # Round to float32 first. scikit-learn trees compare float32 values, so
    # float64 inputs sitting near a split threshold can take a different branch.
    background = np.asarray(preprocessor.transform(background_raw), dtype=np.float32)
    customer_processed = np.asarray(preprocessor.transform(raw_customer), dtype=np.float32)
    explainer = make_explainer(artifact, background)
    values = explainer(customer_processed)

    # For binary GradientBoostingClassifier, raw SHAP output is log-odds.
    expected = float(np.asarray(explainer.expected_value).reshape(-1)[0])
    shap_values = np.asarray(values.values)
    if shap_values.ndim == 3:
        shap_values = shap_values[0, :, 1]
    else:
        shap_values = shap_values[0]

    base_value = expected
    feature_names = clean_feature_names(preprocessor.get_feature_names_out())
    explanation = shap.Explanation(
        values=shap_values,
        base_values=base_value,
        data=np.asarray(customer_processed)[0],
        feature_names=feature_names,
    )
    raw_score = float(inner_model.decision_function(customer_processed)[0])
    reconstructed = base_value + float(np.sum(shap_values))
    if not np.isclose(raw_score, reconstructed, atol=1e-3):
        raise RuntimeError(
            "SHAP additivity check failed. Check SHAP/scikit-learn compatibility."
        )
    return explanation
