import pandas as pd

def predict_one(customer: pd.DataFrame, artifact: dict) -> dict:
    feature_names = artifact["features"]
    missing = [c for c in feature_names if c not in customer.columns]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")
    X = customer.loc[:, feature_names]
    probability = float(artifact["model"].predict_proba(X)[0, 1])
    threshold = float(artifact["threshold"])
    prediction = int(probability >= threshold)
    return {
        "probability": probability,
        "prediction": prediction,
        "label": "Churn" if prediction else "No Churn",
        "threshold": threshold,
    }
