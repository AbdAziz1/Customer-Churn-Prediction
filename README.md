# Customer Churn Prediction
 
A Streamlit application that identifies telecom customers at risk of churning, explains the drivers behind each prediction, and segments customer bases into risk tiers for targeted retention.
 
The model is a class-weighted Gradient Boosting classifier, tuned with stratified cross-validation and a decision threshold selected to maximise F1-score on out-of-fold predictions.
It also explores the factors influencing churn, analyzes customer behavior, and presents business insights through an interactive Power BI dashboard.

## 🔗 Live Project

<https://abdaziz1-customer-churn-pred-customer-churn-streamlitapp-f1q2br.streamlit.app/>

## Features
- **PowerBI Dashboard:** The dashboard enables stakeholders to explore churn trends and identify potential areas for customer retention improvement. 
- **Model Overview:** saved model metadata, the decision threshold, final held-out test metrics, and a plain-language explanation of each metric in terms of customer outcomes.
- **Single Customer Prediction:** enter a customer profile to get a churn probability and classification, with SHAP explanations (bar, decision and force plots), a ranked list of the features that contribute most, and what-if scenarios that show which single changes would lower churn risk.
- **Batch Risk Segmentation:** score the bundled dataset or an uploaded CSV, group customers into Low, Medium and High risk tiers with adjustable cutoffs, review the largest drivers within each tier, and download the scored results.





# Power BI Dashboard

Developed an interactive Power BI dashboard to transform customer data into actionable business insights.
The dashboard contains three analytical sections:

## Page 1 – Executive Overview

- Overall customer and churn KPIs.
- Customer churn rate.
- Revenue-related metrics.
- High-level business performance.

<img width="1312" height="736" alt="Screenshot 2026-11-14 135034" src="https://github.com/user-attachments/assets/be6a859e-86e0-4f30-a3cb-f6f3dfe18608" />


## Page 2 – Customer Segmentation Analysis

- Customer demographic analysis.
- Customer segmentation.
- Contract and service comparisons.
- Churn patterns across customer groups.

<img width="1308" height="737" alt="Screenshot 2026-11-14 135513" src="https://github.com/user-attachments/assets/b0419ce6-ae0e-4366-ae90-0939b83ea76e" />


## Page 3 – Churn Drivers & Revenue Analysis

- Key factors associated with customer churn.
- Revenue and monthly charge analysis.
- Contract-related churn patterns.
- Service-level analysis.

<img width="1302" height="732" alt="Screenshot 2026-11-14 135548" src="https://github.com/user-attachments/assets/64344d43-1d0d-4db2-aaa6-d5f0f598f6c6" />


## Model Summary
 
| Item | Detail |
|---|---|
| Algorithm | Weighted Gradient Boosting (scikit-learn `GradientBoostingClassifier` with balanced sample weights) |
| Inputs | 31 features, including engineered `Tenure Group` and `Total Services` |
| Hyperparameter search | `RandomizedSearchCV`, 25 iterations, optimising F1-score |
| Validation | 5-fold stratified cross-validation on the training set |
| Decision threshold | Selected on out-of-fold predictions to maximise F1-score |
| Train / test split | 80% / 20%, stratified, `random_state=42` |
 
### Final Test Set Performance
 
| Metric | Value |
|---|---|
| Accuracy | 0.8332 |
| Precision | 0.6514 |
| Recall | 0.7995 |
| F1-score | 0.7179 |
| ROC-AUC | 0.9147 |
| PR-AUC | 0.7919 |
 
## Methodology
 
**Feature selection.** Identifier columns, the target and its direct leakage fields (for example, churn score, category and reason), geographic identifiers, the reporting quarter, and cumulative financial totals were excluded to prevent leakage and to keep predictors available at the time of scoring. Constant features were removed.
 
**Preprocessing.** Numeric features are median-imputed and standardised. Categorical features are imputed with the most frequent value and ordinal-encoded, with unseen categories mapped to a sentinel value at inference time. Preprocessing is fitted on the training data only and stored inside the model pipeline, so the app accepts raw, unencoded customer fields.
 
**Imbalance handling.** Churners are the minority class. The final model uses balanced sample weights computed within each training fold, which avoids validation-fold leakage. Class-weighted and random-oversampling variants were compared against the baseline during experimentation.
 
**Threshold selection.** The default 0.50 cutoff is replaced by a threshold chosen from out-of-fold probabilities, which maximises F1-score. The threshold is stored with the model.
 
**Explainability.** SHAP `TreeExplainer` values are computed against a background sample drawn from the training data. Explanations are reported in log-odds (raw model score).

## Notes and Limitations
 
- Scores from the balanced model run higher than real-world churn rates. The risk tiers rank customers; they are not forecasts of the churn rate.
- Results on the bundled dataset include rows the model was trained on and will look better than performance on new customers. Use the held-out test metrics above as the reference.
- What-if scenarios show how the model's score responds to a change in one field. They do not guarantee the effect of a real retention action.
- A customer far from the typical profile is not necessarily at risk. Use the SHAP views to see which differences the model relies on.
