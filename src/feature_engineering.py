import pandas as pd
import numpy as np

def engineer_features(df):
    """
    Creates business-driven engineered features.
    """

    df = df.copy()

    # -------------------------
    # Payment Ratio
    # -------------------------
    df["payment_ratio"] = (
        df["expected_payment"] /
        df["total_billed"]
    )

    # -------------------------
    # Payment Gap
    # -------------------------
    df["payment_gap"] = (
        df["total_billed"] -
        df["expected_payment"]
    )

    # -------------------------
    # Missing Authorization
    # -------------------------
    df["auth_missing"] = (
        (df["prior_auth_required"] == 1) &
        (df["has_prior_auth"] == 0)
    ).astype(int)

    # -------------------------
    # Missing Referral
    # -------------------------
    df["referral_missing"] = (
        (df["referral_required"] == 1) &
        (df["referral_present"] == 0)
    ).astype(int)

    # -------------------------
    # Late Submission
    # -------------------------
    df["late_submission"] = (
        df["days_to_submit"] > 30
    ).astype(int)

    # -------------------------
    # Service Month
    # -------------------------
    df["service_month"] = pd.to_datetime(
        df["service_month"]
    )

    df["service_month_num"] = (
        df["service_month"].dt.month
    )

    return df