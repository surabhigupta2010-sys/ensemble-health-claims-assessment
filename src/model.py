from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

def train_logistic(
    preprocessor,
    X_train,
    y_train,
    random_state=42
):
    

    logistic_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "classifier",
                LogisticRegression(
                    random_state=random_state,
                    max_iter=1000
                )
            )
        ]
    )

    logistic_pipeline.fit(
        X_train,
        y_train
    )

    return logistic_pipeline



def train_random_forest(
    preprocessor,
    X_train,
    y_train,
    random_state=42
):
    rf_pipeline = Pipeline(

    steps=[

        ("preprocessor",preprocessor),

        (

            "classifier",

            RandomForestClassifier(

                n_estimators=200,

                max_depth=10,

                random_state=random_state,

                class_weight="balanced"

            )

            )

        ]

    )

    rf_pipeline.fit(X_train, y_train)

    return rf_pipeline

def train_xgboost(
    preprocessor,
    X_train,
    y_train,
    random_state=42
):
    
    xgb_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            XGBClassifier(
                random_state=random_state,
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss"
            )
        )
    ]
    )

    xgb_pipeline.fit(X_train, y_train)

    return xgb_pipeline


def train_final_model(
    preprocessor,
    X_train,
    y_train,
    random_state=42
):
    model = train_logistic(
    preprocessor,
    X_train,
    y_train,
    random_state
    )

    return model