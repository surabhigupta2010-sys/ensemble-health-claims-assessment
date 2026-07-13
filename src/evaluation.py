import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    precision_recall_curve
)


def evaluate_model(
    model,
    X,
    y,
    model_name
):

    y_pred = model.predict(X)

    y_prob = model.predict_proba(X)[:,1]

    accuracy = accuracy_score(
    y,
    y_pred
    )

    precision = precision_score(
        y,
        y_pred
    )

    recall = recall_score(
        y,
        y_pred
    )

    f1 = f1_score(
        y,
        y_pred
    )

    roc_auc = roc_auc_score(
        y,
        y_prob
    )

    results = pd.DataFrame({
    "Model":[model_name],
    "Accuracy":[accuracy],
    "Precision":[precision],
    "Recall":[recall],
    "F1 Score":[f1],
    "ROC AUC":[roc_auc]
    })

    return results


def plot_confusion_matrix(
    model,
    X,
    y,
    save_path
):
    
    y_pred = model.predict(X)
    cm = confusion_matrix(
    y,
    y_pred
    )

    disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
    )

    disp.plot(cmap="Blues")

    plt.savefig(
    save_path,
    dpi=300,
    bbox_inches="tight"
)

    plt.close()

def plot_roc_curve(
    model,
    X,
    y,
    save_path
):
        
    RocCurveDisplay.from_estimator(
    model,
    X,
    y
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

def save_metrics(
    results,
    save_path
):
    results.to_csv(
    save_path,
    index=False
    )    

def find_probability_threshold(
    model,
    X_val,
    target_review_rate=0.25
):
    
    probabilities = model.predict_proba(
    X_val
    )[:,1]
    
    sorted_probabilities = np.sort(
    probabilities
    )

    threshold = np.percentile(
    probabilities,
    75
    )
    return threshold

def evaluate_with_threshold(
    model,
    X,
    y,
    threshold,
    model_name
):

    y_prob = model.predict_proba(X)[:,1]

    y_pred = (y_prob >= threshold).astype(int)
    
        # Calculate evaluation metrics
    accuracy = accuracy_score(
        y,
        y_pred
    )

    precision = precision_score(
        y,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y,
        y_prob
    )

    # Create results dataframe
    results = pd.DataFrame({
        "Model": [model_name],
        "Threshold": [threshold],
        "Accuracy": [accuracy],
        "Precision": [precision],
        "Recall": [recall],
        "F1 Score": [f1],
        "ROC AUC": [roc_auc]
    })

    return results,y_pred, y_prob