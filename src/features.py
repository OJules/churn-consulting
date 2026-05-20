import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute les features métier au dataset pandas.
    À appeler après drop_id et cast SeniorCitizen.
    """
    df = df.copy()

    # CLV simplifiée
    df["CLV"] = df["MonthlyCharges"] * df["tenure"]
    df["CLV"] = df.apply(
        lambda r: r["MonthlyCharges"] if r["CLV"] == 0 else r["CLV"], axis=1
    )

    # Ratio charges
    df["ChargesRatio"] = df["MonthlyCharges"] / (df["TotalCharges"] + 1)

    # Nombre de services souscrits
    service_cols = [
        "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    df["NbServices"] = df[service_cols].apply(
        lambda row: sum(v not in ["No", "No internet service",
                                   "No phone service"] for v in row),
        axis=1
    )

    # Tenure bucket (années)
    df["TenureBucket"] = (df["tenure"] // 12).astype(str)

    return df