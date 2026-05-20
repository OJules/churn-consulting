import polars as pl
import pandas as pd
from src.config import DATA_FILE, TARGET, ID_COL


def load_data() -> pl.DataFrame:
    df = pl.read_csv(DATA_FILE)
    df = df.with_columns(
        pl.col("TotalCharges").cast(pl.Float64, strict=False)
    )
    df = df.with_columns(
        pl.col(TARGET).map_elements(
            lambda x: 1 if x == "Yes" else 0, return_dtype=pl.Int8
        ).alias(TARGET)
    )
    return df


def drop_id(df: pl.DataFrame) -> pl.DataFrame:
    return df.drop(ID_COL)


def load_data_pandas() -> pd.DataFrame:
    """Version pandas avec tous les casts prêts pour sklearn."""
    df = load_data().to_pandas()
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(str)
    df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"])
    return df