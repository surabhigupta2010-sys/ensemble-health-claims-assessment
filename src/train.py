
import pandas as pd

from joblib import dump

from data_loader import load_training_data

from feature_engineering import engineer_features

from preprocessing import (
    split_data,
    get_feature_columns,
    create_preprocessor,
    combine_train_validation
)

from model import (
    train_logistic,
    train_random_forest,
    train_xgboost
)

from evaluation import (
    evaluate_model,
    evaluate_with_threshold,
    find_probability_threshold,
    save_metrics,
    plot_confusion_matrix,
    plot_roc_curve
)

from config import *

print("Running train.py")

def main():

    print("Inside main")

    df = load_training_data(
    TRAIN_DATA
    )

    df = engineer_features(df)

    (
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test
    ) = split_data(df)
    
    numerical_columns, categorical_columns = get_feature_columns(
        X_train
    )

    preprocessor = create_preprocessor(
    numerical_columns,
    categorical_columns
    )

    logistic_pipeline = train_logistic(
    preprocessor,
    X_train,
    y_train
    )

    rf_pipeline = train_random_forest(
    preprocessor,
    X_train,
    y_train
    )

    xgb_pipeline = train_xgboost(
    preprocessor,
    X_train,
    y_train
    )


    logistic_results = evaluate_model(
    logistic_pipeline,
    X_val,
    y_val,
    "Logistic Regression"
    )

    rf_results = evaluate_model(
    rf_pipeline,
    X_val,
    y_val,
    "Random Forest"
    )

    xgb_results = evaluate_model(
    xgb_pipeline,
    X_val,
    y_val,
    "XGBoost"
    )

    results = pd.concat(
    [
        logistic_results,
        rf_results,
        xgb_results
    ],
    ignore_index=True
    )

    print(results)

    save_metrics(
    results,
    MODEL_METRICS_PATH
    )

    plot_confusion_matrix(
    logistic_pipeline,
    X_val,
    y_val,
    CONFUSION_MATRIX_PATH
    )

    plot_roc_curve(
    logistic_pipeline,
    X_val,
    y_val,
    ROC_CURVE_PATH
    )

    threshold = find_probability_threshold(
    logistic_pipeline,
    X_val
    )

    print(threshold)

    threshold_results = evaluate_with_threshold(
    logistic_pipeline,
    X_val,
    y_val,
    threshold,
    "Logistic Regression"
    )

    print(threshold_results)

    X_train_final, y_train_final = combine_train_validation(
    X_train,
    X_val,
    y_train,
    y_val
    )

    final_model = train_logistic(
    preprocessor,
    X_train_final,
    y_train_final
    )

    from joblib import dump

    dump(
    final_model,
    MODEL_PATH
    )

    dump(
    threshold,
    THRESHOLD_PATH
    )

    print("Training completed successfully.")


if __name__ == "__main__":
    main()   