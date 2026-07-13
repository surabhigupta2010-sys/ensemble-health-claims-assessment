from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)

import numpy as np

import pandas as pd


def split_data(
    df,
    target_column="is_denied"
):
    """
    Uses the predefined train/validation/test split.
    """

    train_df = df[
        df["split"] == "train"
    ]

    val_df = df[
        df["split"] == "validation"
    ]

    test_df = df[
        df["split"] == "test"
    ]

    drop_columns = [
    "claim_id",
    "split",
    "service_month",
    "is_denied",
    "denial_reason"
    ]

    X_train = train_df.drop(
        columns=drop_columns
    )

    X_val = val_df.drop(
        columns=drop_columns
    )

    X_test = test_df.drop(
        columns=drop_columns
    )

    y_train = train_df[target_column]

    y_val = val_df[target_column]

    y_test = test_df[target_column]

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    )


def get_feature_columns(X_train):
    """
    Identify numerical and categorical columns.
    """
    numerical_features = X_train.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_features = X_train.select_dtypes(
        include=["object"]
    ).columns.tolist()

    return (
        numerical_features,
        categorical_features
    )

def create_preprocessor(
    numerical_features,
    categorical_features
):
    """
    Creates preprocessing pipeline.
    """

    numeric_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numeric_transformer,
                numerical_features
            ),
            (
                "cat",
                categorical_transformer,
                categorical_features
            )
        ]
    )

    return preprocessor

def combine_train_validation(
    X_train,
    X_val,
    y_train,
    y_val
):

    X_train_final = pd.concat(
        [X_train, X_val],
        axis=0
    )

    y_train_final = pd.concat(
        [y_train, y_val],
        axis=0
    )

    return X_train_final, y_train_final