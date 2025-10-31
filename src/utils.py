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
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    path = f"plots/{model_name}_confusion.png"
    plt.savefig(path); plt.close()
    mlflow.log_artifact(path)

def log_residual_plot(y_true, y_pred, model_name):
    ensure_dir("plots")
    residuals = y_true - y_pred
    plt.figure(figsize=(7, 4))
    sns.scatterplot(x=y_pred, y=residuals)
    plt.axhline(0, color='red', linestyle='--')
    plt.title(f"{model_name} Residuals vs Predicted")
    plt.xlabel("Predicted"); plt.ylabel("Residuals")
    path = f"plots/{model_name}_residuals.png"
    plt.savefig(path); plt.close()
    mlflow.log_artifact(path)
