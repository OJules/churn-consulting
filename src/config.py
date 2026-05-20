 
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "report"

DATA_FILE = DATA_DIR / "Telco_Cusomer_Churn.csv"

# Column roles
TARGET = "Churn"
ID_COL = "customerID"

NUMERICAL_COLS = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

CATEGORICAL_COLS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

# ML
RANDOM_STATE = 42
TEST_SIZE = 0.2

