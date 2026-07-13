import pandas as pd

from joblib import load

from config import (
    CURRENT_DATA,
    MODEL_PATH,
    THRESHOLD_PATH,
    OUTPUT_PATH, 
    SHAP_SUMMARY_PATH
)

from data_loader import load_scoring_data

from feature_engineering import engineer_features

from explainability import (
    calculate_shap,
    save_shap_summary,
    get_top_risk_factors
)

from llm import generate_explanation


def main():

    # --------------------------------------------------
    # Load trained model and threshold
    # --------------------------------------------------

    model = load(MODEL_PATH)

    threshold = load(THRESHOLD_PATH)

    print("Model Loaded")

    # --------------------------------------------------
    # Load current claims
    # --------------------------------------------------

    current_claims = load_scoring_data(
        CURRENT_DATA
    )

    # --------------------------------------------------
    # Feature Engineering
    # --------------------------------------------------

    current_claims = engineer_features(
        current_claims
    )

    # --------------------------------------------------
    # Preserve original columns
    # --------------------------------------------------

    scoring_output = current_claims.copy()

    # --------------------------------------------------
    # Create feature dataframe
    # (same columns used during training)
    # --------------------------------------------------

    X = current_claims.drop(
        columns=[
            "claim_id",
            "service_month"
        ],
        errors="ignore"
    )

    # --------------------------------------------------
    # Predict probabilities
    # --------------------------------------------------

    probabilities = model.predict_proba(X)[:, 1]

    scoring_output["denial_probability"] = probabilities

    # --------------------------------------------------
    # Predict denial
    # --------------------------------------------------

    scoring_output["predicted_denial"] = (
        scoring_output["denial_probability"] >= threshold
    ).astype(int)

    # --------------------------------------------------
    # Risk Tier
    # --------------------------------------------------

    scoring_output["risk_tier"] = pd.qcut(
        scoring_output["denial_probability"],
        q=[0, 0.50, 0.75, 1],
        labels=[
            "Low",
            "Medium",
            "High"
        ]
    )

    # --------------------------------------------------
    # SHAP Explainability
    # --------------------------------------------------

    shap_values, transformed_features, feature_names = calculate_shap(
        model,
        X
    )

    save_shap_summary(
        shap_values,
        transformed_features,
        feature_names,
        SHAP_SUMMARY_PATH
    )

    scoring_output["top_risk_factors"] = (
        get_top_risk_factors(
            shap_values,
            feature_names
        )
    )

    # --------------------------------------------------
    # LLM Explanations
    # --------------------------------------------------

    print("Generating LLM explanations...")

# Initialize explanation column
    scoring_output["explanation"] = ""

    # Top 10 highest-risk claims
    top10_claims = scoring_output.nlargest(
        10,
        "denial_probability"
    )

    print("Generating LLM explanations for Top 10 highest-risk claims...")

    for idx in top10_claims.index:

        try:
            scoring_output.loc[idx, "explanation"] = generate_explanation(
                scoring_output.loc[idx]
            )

        except Exception as e:
            scoring_output.loc[idx, "explanation"] = (
                f"Explanation unavailable: {str(e)}"
            )

    # --------------------------------------------------
    # Save Output
    # --------------------------------------------------

    scoring_output.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("Scoring completed successfully.")

    print(f"Output saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()