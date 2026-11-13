import pandas as pd

EXCLUDED = [
    "Customer ID", "Churn Label", "Churn Value", "Churn Target",
    "Customer Status", "Churn Score", "Churn Category", "Churn Reason",
    "Satisfaction Score", "CLTV", "Total Charges", "Total Revenue",
    "Total Refunds", "Total Long Distance Charges",
    "Total Extra Data Charges", "Country", "State", "City", "Zip Code",
    "Lat Long", "Latitude", "Longitude", "Population", "Quarter",
]

SERVICE_COLUMNS = [
    "Phone Service", "Multiple Lines", "Internet Service", "Online Security",
    "Online Backup", "Device Protection Plan", "Premium Tech Support",
    "Streaming TV", "Streaming Movies", "Streaming Music", "Unlimited Data",
]

def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the feature engineering used before model training."""
    data = frame.copy()
    data["Tenure Group"] = pd.cut(
        data["Tenure in Months"],
        bins=[-1, 12, 24, 48, 60, 72],
        labels=[
            "0-12 Months", "13-24 Months", "25-48 Months",
            "49-60 Months", "61-72 Months",
        ],
    )
    service_cols = [c for c in SERVICE_COLUMNS if c in data.columns]
    data["Total Services"] = data[service_cols].eq("Yes").sum(axis=1)
    return data

def model_frame(raw_frame: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    engineered = engineer_features(raw_frame)
    missing = [c for c in feature_names if c not in engineered.columns]
    if missing:
        raise ValueError(f"Missing model inputs: {missing}")
    return engineered.loc[:, feature_names]
