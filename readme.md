# Medical Claims Denial Prediction using Machine Learning & Generative AI

## Project Overview

This project was completed as part of the Ensemble Health AI/ML Take-Home Assessment.

The objective is to predict insurance claim denials before claim submission and prioritize high-risk claims for manual review. In addition, Large Language Models (LLMs) are used to generate concise, actionable explanations for the highest-risk claims.

---

## Problem Statement

Hospitals can manually review only the top 25% of claims before submission. Therefore, the solution focuses on identifying and prioritizing the highest-risk claims rather than maximizing overall classification accuracy.

---

## Project Structure

```
├── data/
│   ├── claims_history.csv
│   └── current_claims.csv
│
├── models/
│   ├── logistic_model.pkl
│   └── threshold.pkl
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Model_Building.ipynb
│   └── 03_Claim_Scoring_and_LLM_Explanations.ipynb
│
├── outputs/
│
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── preprocessing.py
│   ├── model.py
│   ├── evaluation.py
│   ├── explainability.py
│   ├── llm.py
│   ├── train.py
│   └── score.py
│
├── requirements.txt
└── README.md
```

---

## Approach

### 1. Exploratory Data Analysis

- Missing value analysis
- Target distribution
- Numerical feature analysis
- Categorical feature analysis
- Business insights

### 2. Feature Engineering

The following business features were engineered:

- Payment Ratio
- Payment Gap
- Missing Authorization
- Missing Referral
- Late Submission
- Service Month Number

### 3. Data Preprocessing

- Used predefined Train / Validation / Test split provided in the dataset.
- Median imputation for numerical features.
- Most-frequent imputation for categorical features.
- Standard scaling for numerical variables.
- One-Hot Encoding for categorical variables.
- Preprocessing implemented using Scikit-learn Pipeline and ColumnTransformer.

### 4. Models Evaluated

Three classification models were trained and compared:

- Logistic Regression
- Random Forest
- XGBoost

Evaluation metrics included:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

Logistic Regression was selected as the final model based on validation performance and model interpretability.

---

## Threshold Strategy

The review team can inspect only the highest-risk 25% of claims before submission.

Instead of using the default probability threshold of 0.5, claims were ranked by predicted denial probability and an operational threshold was selected to prioritize approximately the top 25% highest-risk claims for manual review.

Risk tiers were assigned as:

- High Risk – Top 25%
- Medium Risk – 50–75%
- Low Risk – Bottom 50%

---

## Explainability

SHAP values were used to identify the most influential features contributing to each prediction.

For the Top 10 highest-risk claims, these SHAP feature contributions together with the original claim attributes were provided to the Gemini LLM to generate concise, grounded and actionable explanations for analysts.

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

```bash
python src/train.py
```

This command will:

- Load historical claims
- Perform feature engineering
- Train Logistic Regression, Random Forest and XGBoost models
- Evaluate all models
- Select the operational threshold
- Retrain the final Logistic Regression model on Train + Validation data
- Save:
  - `models/logistic_model.pkl`
  - `models/threshold.pkl`
  - evaluation metrics and plots in the `outputs` folder

### 3. Score current claims

```bash
python src/score.py
```

This command will:

- Load the trained model
- Score current claims
- Assign risk tiers
- Generate SHAP explanations
- Generate Gemini explanations for the Top 10 highest-risk claims
- Save:

```
outputs/predictions_current_claims.csv
outputs/shap_summary.png
```

---

## Output

The final prediction file contains:

- claim_id
- denial_probability
- predicted_denial
- risk_tier
- top_risk_factors
- explanation

Claims are sorted in descending order of denial probability to prioritize manual review.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- SHAP
- Matplotlib
- OpenAI Python SDK (Gemini API compatible)
- Google Gemini 3.1 Flash Lite

---

## Future Improvements

- Improve model recall through additional feature engineering and hyperparameter tuning.
- Explore techniques to reduce feature collinearity.
- Experiment with ensemble and cost-sensitive learning approaches.
- Generate explanations for all claims using asynchronous or batch LLM inference while handling API rate limits.