# Medical Claims Denial Prediction using Machine Learning & Generative AI

## Project Overview

This project was completed as part of the Ensemble Health AI/ML Take-Home Assessment.

The objective is to predict insurance claim denials before claim submission and prioritize high-risk claims for manual review. In addition, Large Language Models (LLMs) are used to generate concise, actionable explanations for the highest-risk claims.

---

## Problem Statement

Hospitals can manually review only the top 25% of claims before submission. Therefore, the solution focuses on identifying and prioritizing the highest-risk claims rather than maximizing overall classification accuracy.

---



## Approach

### 1. Exploratory Data Analysis

- Missing value analysis
- Target distribution
- Numerical feature analysis
- Categorical feature analysis
- Business insights

### 2. Feature Engineering

Engineered features include:

- Payment Ratio
- Payment Gap
- Missing Authorization
- Missing Referral
- Late Submission
- Service Month Number

---

## Models Evaluated

- Logistic Regression
- Random Forest
- XGBoost

Logistic Regression was selected as the final model based on validation performance and model interpretability.

---

## Threshold Strategy

The review team can inspect only the highest-risk 25% of claims.

Claims were ranked by predicted denial probability, and the operational threshold was selected to prioritize approximately the top 25% highest-risk claims for manual review.

Risk tiers were assigned as:

- High: Top 25%
- Medium: 50–75%
- Low: Bottom 50%

---

## Explainability

SHAP values were used to identify the top contributing features for each claim.

These feature contributions, along with the original claim attributes, were provided to Gemini to generate grounded, plain-English explanations for the highest-risk claims.

---



## How to Run

1. Install dependencies

```bash
pip install -r requirements.txt

Run notebooks in sequence

01_EDA.ipynb
↓

02_Model_Building.ipynb
↓

03_Claim_Scoring_and_LLM_Explanations.ipynb

Final output
outputs/predictions_current_claims.csv