import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

from src.config import (
    NUMERICAL_COLS, CATEGORICAL_COLS, TARGET,
    RANDOM_STATE, TEST_SIZE, MODELS_DIR
)


def build_preprocessor() -> ColumnTransformer:
    numerical_pipeline = Pipeline([("scaler", StandardScaler())])
    categorical_pipeline = Pipeline([
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    return ColumnTransformer([
        ("num", numerical_pipeline, NUMERICAL_COLS),
        ("cat", categorical_pipeline, CATEGORICAL_COLS),
    ])


def build_xgboost(scale_pos_weight: float) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        eval_metric="auc",
        verbosity=0
    )


def train_full_pipeline(df):
    """
    Entraîne le pipeline complet sur df pandas.
    Retourne le modèle calibré et le preprocessor fittés.
    """
    df = df.copy()
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(str)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    preprocessor = build_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc  = preprocessor.transform(X_test)

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    xgb = build_xgboost(scale_pos_weight)
    xgb.fit(X_train_proc, y_train)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    calibrated = CalibratedClassifierCV(xgb, method="isotonic", cv=cv)
    calibrated.fit(X_train_proc, y_train)

    # Feature names
    cat_features = (preprocessor
                    .named_transformers_["cat"]["encoder"]
                    .get_feature_names_out(CATEGORICAL_COLS))
    feature_names = NUMERICAL_COLS + list(cat_features)

    # Sauvegarde
    joblib.dump(preprocessor,  MODELS_DIR / "preprocessor.pkl")
    joblib.dump(calibrated,    MODELS_DIR / "xgb_calibrated.pkl")
    joblib.dump((X_train_proc, X_test_proc, y_train, y_test, feature_names),
                MODELS_DIR / "train_test_data.pkl")

    return calibrated, preprocessor, X_test_proc, y_test, feature_names