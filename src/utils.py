import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
import mlflow
import random
import random
random.seed(42)
from tensorflow.keras import layers, models, regularizers, optimizers
from tensorflow.keras.layers import LeakyReLU
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
import numpy as np
np.random.seed(42)
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.inspection import permutation_importance





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

def log_classification_learning_curve(history, model_name):
    ensure_dir("plots")
    
    plt.figure(figsize=(7, 5))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    if 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f"{model_name} Learning Curve (Classification)")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    path = f"plots/{model_name}_learning_curve_classification.png"
    plt.savefig(path)
    plt.close()
    mlflow.log_artifact(path)

def log_regression_learning_curve(history, model_name):
    ensure_dir("plots")
    
    plt.figure(figsize=(7, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f"{model_name} Learning Curve (Regression)")
    plt.xlabel("Epochs")
    plt.ylabel("Loss (MSE)")
    plt.legend()
    path = f"plots/{model_name}_learning_curve_regression.png"
    plt.savefig(path)
    plt.close()
    mlflow.log_artifact(path)

def log_nn_ablation(model, X_test, y_test, model_name, baseline_metric):
    ensure_dir("plots")
    feature_losses = []
    for i in range(X_test.shape[1]):
        X_ablate = X_test.copy()
        X_ablate[:, i] = 0
        y_pred = model.predict(X_ablate, verbose=0).flatten()
        loss = mean_squared_error(y_test, y_pred)
        feature_losses.append(loss - baseline_metric)
    
    plt.figure(figsize=(8, 6))
    plt.bar(range(X_test.shape[1]), feature_losses)
    plt.title(f"{model_name} NN Feature Ablation")
    plt.xlabel("Feature Index")
    plt.ylabel("Increase in Loss")
    path = f"plots/{model_name}_feature_ablation.png"
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    mlflow.log_artifact(path)


def log_feature_importance(model, X_test, y_test, model_name, metric):
    ensure_dir("plots")
    
    result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, scoring=metric)
    
    importances = result.importances_mean
    sorted_idx = result.importances_mean.argsort()[::-1]

    if hasattr(X_test, "columns"):
        feature_names = np.array(X_test.columns)
    else:
        feature_names = np.array([f"feat_{i}" for i in range(X_test.shape[1])])

    sorted_names = feature_names[sorted_idx]

    plt.figure(figsize=(10, 6))
    plt.barh(sorted_names, importances[sorted_idx])
    plt.gca().invert_yaxis()  # Most important at the top
    plt.title(f"{model_name} Feature Importance (Permutation)")
    plt.xlabel("Importance")
    plt.tight_layout()

    path = f"plots/{model_name}_feature_importance.png"
    plt.savefig(path)
    plt.close()

    mlflow.log_artifact(path)

def random_search_trial_c(X_train, y_train, param_space, k=5):
    # Pick random hyperparameters
    lr = random.choice(param_space["learning_rate"])
    batch_size = random.choice(param_space["batch_size"])
    hidden_layer_sizes = random.choice(param_space["hidden_layer_sizes"])
    dropout = random.choice(param_space["dropout"])
    l2_reg = random.choice(param_space["l2"])
    activation = random.choice(param_space["activation"])
    epochs = random.choice(param_space["epochs"])


    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    f1_scores = []

    for train_idx, val_idx in kf.split(X_train):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        # Build the model
        model = models.Sequential()
        model.add(layers.Input(shape=(X_train.shape[1],)))
        
        for size in hidden_layer_sizes:
            model.add(layers.Dense(size, kernel_regularizer=regularizers.l2(l2_reg)))
            if activation == "leaky_relu":
                model.add(LeakyReLU(negative_slope=0.1))
            else:
                model.add(layers.Activation(activation))
            model.add(layers.Dropout(dropout))

        model.add(layers.Dense(1, activation='sigmoid'))



        model.compile(
            optimizer=optimizers.Adam(learning_rate=lr),
            loss="binary_crossentropy"
        )

        model.fit(
            X_tr, y_tr,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
        )

        y_pred_prob = model.predict(X_val, verbose=0)
        y_pred = (y_pred_prob > 0.5).astype(int)

        f1_scores.append(f1_score(y_val, y_pred))

    avg_f1 = np.mean(f1_scores)

    return {
        "avg_f1": avg_f1,
        "params": {
            "lr": lr, 
            "batch_size": batch_size, 
            "hidden_layer_sizes": hidden_layer_sizes, 
            "dropout": dropout, 
            "l2": l2_reg, 
            "activation": activation, 
            "epochs": epochs
        }
    }

def random_search_trial_r(X_train, y_train, param_space, k=5):
    # Pick random hyperparameters
    lr = random.choice(param_space["learning_rate"])
    batch_size = random.choice(param_space["batch_size"])
    hidden_layer_sizes = random.choice(param_space["hidden_layer_sizes"])
    dropout = random.choice(param_space["dropout"])
    l2_reg = random.choice(param_space["l2"])
    activation = random.choice(param_space["activation"])
    epochs = random.choice(param_space["epochs"])

    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    mae_scores = []

    for train_idx, val_idx in kf.split(X_train):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        # Build the model
        model = models.Sequential()
        model.add(layers.Input(shape=(X_train.shape[1],)))
        
        for size in hidden_layer_sizes:
            model.add(layers.Dense(size, kernel_regularizer=regularizers.l2(l2_reg)))
            if activation == "leaky_relu":
                model.add(LeakyReLU(negative_slope=0.1))
            else:
                model.add(layers.Activation(activation))
            model.add(layers.Dropout(dropout))

        model.add(layers.Dense(1))

        # Compile
        model.compile(
            optimizer=optimizers.Adam(learning_rate=lr),
            loss="mse",
            metrics=["mae"]
        )

        model.fit(
            X_tr, y_tr,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
        )

        y_pred = model.predict(X_val, verbose=0).flatten()
        mae_scores.append(mean_absolute_error(y_val, y_pred))

    avg_mae = np.mean(mae_scores)

    return {
        "avg_mae": avg_mae,
        "params": {
            "lr": lr, 
            "batch_size": batch_size, 
            "hidden_layer_sizes": hidden_layer_sizes, 
            "dropout": dropout, 
            "l2": l2_reg, 
            "activation": activation, 
            "epochs": epochs
        }
    }