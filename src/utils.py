import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
import mlflow

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def log_confusion_matrix(y_true, y_pred, model_name):
    ensure_dir("plots")
    cm = confusion_matrix(y_true, y_pred)

    labels = ["Not Passed", "Passed"]

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=labels, yticklabels=labels)
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted", fontweight="bold"); plt.ylabel("Actual", fontweight="bold")
    path = f"plots/{model_name}_confusion.png"
    plt.savefig(path); plt.close()
    mlflow.log_artifact(path)

def log_residual_plot(y_true, y_pred, model_name):
    ensure_dir("plots")

    # Residual vs Predicted
    residuals = y_true - y_pred
    plt.figure(figsize=(7, 4))
    sns.scatterplot(x=y_pred, y=residuals, s=25, alpha=0.7, edgecolor="k")
    plt.axhline(0, color='red', linestyle='--', linewidth=1)
    plt.title(f"{model_name} Residuals vs Predicted")
    plt.xlabel("Predicted G3")
    plt.ylabel("Residuals (Actual - Predicted)")
    path1 = f"plots/{model_name}_residuals.png"
    plt.tight_layout()
    plt.savefig(path1)
    plt.close()
    mlflow.log_artifact(path1)

    # Predicted vs Actual
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.6, edgecolor="k", linewidth=0.5)
    mn, mx = y_true.min(), y_true.max()
    plt.plot([mn, mx], [mn, mx], "r--", linewidth=2, label="Perfect fit")
    plt.xlabel("Actual Final Grade (G3)")
    plt.ylabel("Predicted Final Grade (G3)")
    plt.title(f"{model_name} Predicted vs Actual (Regression)")
    plt.legend()
    path2 = f"plots/{model_name}_pred_vs_actual.png"
    plt.tight_layout()
    plt.savefig(path2)
    plt.close()
    mlflow.log_artifact(path2)
