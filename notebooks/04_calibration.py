# %% Imports

import matplotlib.pyplot as plt
import joblib
import sys
sys.path.append("..")

from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
from src.config import MODELS_DIR, RANDOM_STATE

X_train, X_test, y_train, y_test, feature_names = joblib.load(
    MODELS_DIR / "train_test_data.pkl"
)
xgb_w = joblib.load(MODELS_DIR / "xgb_model.pkl")
print("Data and model loaded.")


# %% Calibation curve avant calibration

y_proba = xgb_w.predict_proba(X_test)[:, 1]

fraction_of_positives, mean_predicted_value = calibration_curve(
    y_test, y_proba, n_bins=10
)

plt.figure(figsize=(7, 6))
plt.plot(mean_predicted_value, fraction_of_positives,
         "s-", label="XGBoost Weighted (before)", color="tomato")
plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
plt.xlabel("Mean predicted probability")
plt.ylabel("Fraction of positives")
plt.title("Calibration Curve — Before")
plt.legend()
plt.tight_layout()
plt.show()

brier_before = brier_score_loss(y_test, y_proba)
print(f"Brier Score (before): {brier_before:.4f}")

# %% Calibration isotonic

from sklearn.model_selection import StratifiedKFold  # noqa: E402

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

calibrated_model = CalibratedClassifierCV(
    xgb_w, method="isotonic", cv=cv
)
calibrated_model.fit(X_train, y_train)

y_proba_cal = calibrated_model.predict_proba(X_test)[:, 1]

fraction_cal, mean_cal = calibration_curve(
    y_test, y_proba_cal, n_bins=10
)

plt.figure(figsize=(7, 6))
plt.plot(mean_predicted_value, fraction_of_positives,
         "s-", label="Before calibration", color="tomato", alpha=0.7)
plt.plot(mean_cal, fraction_cal,
         "s-", label="After calibration (isotonic)", color="steelblue")
plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
plt.xlabel("Mean predicted probability")
plt.ylabel("Fraction of positives")
plt.title("Calibration Curve — Before vs After")
plt.legend()
plt.tight_layout()
plt.show()

brier_after = brier_score_loss(y_test, y_proba_cal)
print(f"Brier Score (before): {brier_before:.4f}")
print(f"Brier Score (after):  {brier_after:.4f}")
print(f"Improvement: {(brier_before - brier_after) / brier_before:.1%}")

# %% Save calibrated model
joblib.dump(calibrated_model, MODELS_DIR / "xgb_calibrated.pkl")
joblib.dump(y_proba_cal, MODELS_DIR / "y_proba_calibrated.pkl")
print("Calibrated model saved.")

