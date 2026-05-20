import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, classification_report,
    brier_score_loss
)
from sklearn.calibration import calibration_curve


def print_metrics(y_test, y_pred, y_proba, model_name: str = "Model"):
    auc = roc_auc_score(y_test, y_proba)
    brier = brier_score_loss(y_test, y_proba)
    print(f"\n=== {model_name} ===")
    print(f"AUC-ROC:     {auc:.4f}")
    print(f"Brier Score: {brier:.4f}")
    print(classification_report(y_test, y_pred,
                                 target_names=["No Churn", "Churn"]))


def plot_calibration(y_test, y_proba, label: str = "Model"):
    fraction, mean_pred = calibration_curve(y_test, y_proba, n_bins=10)
    plt.figure(figsize=(7, 6))
    plt.plot(mean_pred, fraction, "s-", label=label, color="steelblue")
    plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title(f"Calibration Curve — {label}")
    plt.legend()
    plt.tight_layout()
    plt.show()