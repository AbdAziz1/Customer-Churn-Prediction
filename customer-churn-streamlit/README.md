# Customer Churn Streamlit App

A Streamlit app that scores customer churn risk with a saved Weighted Gradient Boosting pipeline and explains the results with SHAP.

## Pages

| Page | What it does |
|------|--------------|
| **Home** (`app.py`) | Overview of the app with links to the pages and a sidebar summary of the saved model. |
| **Model Overview** | Model name, decision threshold and final test set performance (accuracy, precision, recall, F1, ROC-AUC, PR-AUC), with a plain-English explanation of each metric in terms of customer churn and a worked 1,000-customer example. |
| **Single Customer Prediction** | Grouped input form, risk card with a probability bar and the decision threshold, SHAP views (bar, decision and force plots), a largest-contributors table, a "Why this prediction" summary, what-if scenarios and a comparison with the typical customer. |
| **Batch Risk Segmentation** | Scores a batch of customers (the project dataset or an uploaded CSV), splits them into Low, Medium and High risk with adjustable cutoffs, lists customers by tier with a CSV download, and shows SHAP bar and beeswarm plots for a chosen tier. |

## Project structure

```text
customer-churn-streamlit/
├── .streamlit/
│   └── config.toml
├── app.py
├── pages/
│   ├── 1_Model_Overview.py
│   ├── 2_Single_Customer_Prediction.py
│   └── 3_Batch_Risk_Segmentation.py
├── utils/
│   ├── model_loader.py
│   ├── data_prep.py
│   ├── prediction.py
│   ├── shap_explainer.py
│   └── styles.py
├── artifacts/
│   └── customer_churn_model.joblib
├── data/
│   └── telco.csv
└── requirements.txt
```

## Setup

Use Python 3.11 or 3.12 if available. The model artifact was serialized with scikit-learn 1.6.1, so the dependency is pinned to that version.

```bash
python -m venv .venv
```

Activate the environment:

**Windows**
```bash
.venv\\Scripts\\activate
```

**macOS/Linux**
```bash
source .venv/bin/activate
```

Install dependencies and run:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Run the command from inside the `customer-churn-streamlit/` folder. Restart the app after changing `.streamlit/config.toml`, because theme changes only apply on a full restart.

## Styling

- `.streamlit/config.toml` sets the global light theme and blue accent color.
- `utils/styles.py` holds the shared CSS for the metric cards and the risk card renderer. Every page calls `apply_styles()` near the top, because Streamlit runs each file in `pages/` as its own script.

## Important deployment notes

- Keep the saved artifact in `artifacts/customer_churn_model.joblib`.
- Keep the original `telco.csv` in `data/telco.csv`. The Single Customer Prediction page uses it for valid input options, default values, the SHAP background sample, what-if options and the typical-customer comparison. The Batch Risk Segmentation page uses it as its default data source. The Model Overview page does not need it.
- The prediction page asks for the original 29 fields and calculates `Tenure Group` and `Total Services` automatically, producing all 31 model inputs.
- Do not change the feature names or their order.
- The Model Overview metrics are fixed values copied from the training notebook's "Final test set performance" output. If you retrain the model, update the `metrics` dictionary in `pages/1_Model_Overview.py`.
- The tier summary on the batch page also shows the actual churn rate per tier when the data has a `Churn Value` column, as `telco.csv` does.

## Batch data format

An uploaded CSV must contain the 29 original customer fields used by the model (the same fields as the prediction form). `Tenure Group` and `Total Services` are calculated and should not be included. Any missing columns are listed on the page.

## How to read the results

- **Probabilities:** the model is trained with balanced class weights, so its probabilities run higher than real-world churn rates. Compare a customer against the decision threshold, not against the actual churn rate.
- **Risk tiers:** High starts at the decision threshold by default, and Medium starts at 30%. Both cutoffs are adjustable and are not dictated by the model. The tiers rank customers and are not forecasts of the churn rate.
- **Telco dataset results:** `telco.csv` includes the rows the model was trained on, so batch results on it look better than they would on new customers.
- **SHAP:** it explains the inner fitted GradientBoostingClassifier after the pipeline's preprocessor. The bar plot is in raw-score (log-odds) space, while the decision and force plots are drawn on a probability axis. Features are labelled with the customer's actual values (for example `Contract = Month-to-Month`).
- **What-if scenarios:** each scenario changes one field and keeps everything else the same, including the monthly charge. They show how the model's score responds, not the guaranteed effect of a real retention action.

## Known issue

On the Single Customer Prediction page, SHAP's additivity check can fail for some inputs ("Additivity check failed in TreeExplainer"). The prediction and risk card still appear, because only the SHAP section is blocked. This is under investigation, so remove this note once it is resolved.