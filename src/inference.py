import pandas as pd
import joblib
from src.config import MODELS_DIR
from src.features import engineer_features


def load_artifacts():
    preprocessor     = joblib.load(MODELS_DIR / "preprocessor.pkl")
    calibrated_model = joblib.load(MODELS_DIR / "xgb_calibrated.pkl")
    return preprocessor, calibrated_model


def score_customers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prend un DataFrame pandas propre (sans TARGET),
    retourne le même DataFrame avec ChurnProbability,
    CLV et RetentionPriority ajoutés.
    """
    preprocessor, model = load_artifacts()

    df = df.copy()
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(str)

    # Features engineered
    df = engineer_features(df)

    # Colonnes pour le preprocessor uniquement
    from src.config import NUMERICAL_COLS, CATEGORICAL_COLS
    X = df[NUMERICAL_COLS + CATEGORICAL_COLS]
    X_proc = preprocessor.transform(X)

    df["ChurnProbability"] = model.predict_proba(X_proc)[:, 1]
    df["CLV"] = df["MonthlyCharges"] * df["tenure"]
    df["CLV"] = df.apply(
        lambda r: r["MonthlyCharges"] if r["CLV"] == 0 else r["CLV"], axis=1
    )
    df["RetentionPriority"] = df["ChurnProbability"] * df["CLV"]

    return df


def assign_segment(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute la colonne Segment au DataFrame scoré."""
    p75 = df["RetentionPriority"].quantile(0.75)
    p40 = df["RetentionPriority"].quantile(0.40)

    def _segment(row):
        if row["ChurnProbability"] >= 0.5 and row["RetentionPriority"] >= p75:
            return "A - High Priority"
        elif row["ChurnProbability"] >= 0.3 or row["RetentionPriority"] >= p40:
            return "B - Medium Priority"
        else:
            return "C - Low Priority"

    df["Segment"] = df.apply(_segment, axis=1)
    return df