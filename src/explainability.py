import shap
import pandas as pd
import matplotlib.pyplot as plt


def calculate_shap(model, X):
    """
    Calculates SHAP values for a trained pipeline.

    Parameters
    ----------
    model : sklearn Pipeline
        Trained ML pipeline.

    X : DataFrame
        Input features.

    Returns
    -------
    shap_values
    transformed_features
    feature_names
    """

    # Extract preprocessing and classifier
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    # Transform features
    transformed_features = preprocessor.transform(X)

    # Get transformed feature names
    feature_names = preprocessor.get_feature_names_out()

    # SHAP Explainer
    explainer = shap.Explainer(
        classifier,
        transformed_features
    )

    shap_values = explainer(transformed_features)

    return (
        shap_values,
        transformed_features,
        feature_names
    )


def save_shap_summary(
    shap_values,
    transformed_features,
    feature_names,
    save_path
):
    """
    Saves SHAP summary plot.
    """

    plt.figure(figsize=(10, 6))

    shap.summary_plot(
        shap_values,
        transformed_features,
        feature_names=feature_names,
        show=False
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def get_top_risk_factors(
    shap_values,
    feature_names,
    top_n=3
):
    """
    Returns top SHAP features for every observation.

    Returns
    -------
    list
        List of strings.
    """

    top_features = []

    for row in shap_values.values:

        importance = abs(row)

        top_idx = importance.argsort()[-top_n:][::-1]

        feature_list = [
            feature_names[i]
            for i in top_idx
        ]

        top_features.append(
            ", ".join(feature_list)
        )

    return top_features