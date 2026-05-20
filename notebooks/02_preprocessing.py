# %% Imports
import polars as pl
import sys
sys.path.append("..")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib

from src.data_loader import load_data, drop_id
from src.config import (
    TARGET, NUMERICAL_COLS, CATEGORICAL_COLS,
    RANDOM_STATE, TEST_SIZE, MODELS_DIR
)

df = load_data()
df = drop_id(df)
print(df.shape)

# %% Check for missing values
nulls = df.null_count()
print(nulls)

# TotalCharges : combien de nulls après cast ?
print(f"\nTotalCharges nulls: {df['TotalCharges'].null_count()}")


# %% Imputation TotalCharges
# Les nulls TotalCharges correspondent à tenure=0 → on impute par MonthlyCharges
df = df.with_columns(
    pl.when(pl.col("TotalCharges").is_null())
    .then(pl.col("MonthlyCharges"))
    .otherwise(pl.col("TotalCharges"))
    .alias("TotalCharges")
)

print(f"TotalCharges nulls after imputation: {df['TotalCharges'].null_count()}")


# %% SeniorCitizen : cast en string pour encodage uniforme

df = df.with_columns(
    pl.col("SeniorCitizen").cast(pl.Utf8)
)
print(df["SeniorCitizen"].value_counts())


# %% train-test split
df_pd = df.to_pandas()

X = df_pd.drop(columns=[TARGET])
y = df_pd[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Churn rate train: {y_train.mean():.2%}")
print(f"Churn rate test:  {y_test.mean():.2%}")


# %%  Pipeline sklearn
numerical_pipeline = Pipeline([
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer([
    ("num", numerical_pipeline, NUMERICAL_COLS),
    ("cat", categorical_pipeline, CATEGORICAL_COLS),
])

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed  = preprocessor.transform(X_test)

print(f"X_train_processed shape: {X_train_processed.shape}")
print(f"X_test_processed shape:  {X_test_processed.shape}")

# %% Feature names
cat_feature_names = preprocessor.named_transformers_["cat"]["encoder"].get_feature_names_out(CATEGORICAL_COLS)
feature_names = NUMERICAL_COLS + list(cat_feature_names)
print(f"Total features: {len(feature_names)}")
print(feature_names[:10])

# %% Sauvegarde preprocessor
joblib.dump(preprocessor, MODELS_DIR / "preprocessor.pkl")
joblib.dump((X_train_processed, X_test_processed, y_train, y_test, feature_names),
            MODELS_DIR / "train_test_data.pkl")
print("Preprocessor and data saved.")
# %%
