import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
import mlflow
import random
from tensorflow.keras import layers, models, regularizers, optimizers
from tensorflow.keras.layers import LeakyReLU

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



def random_search_trial_c(X_train, y_train, X_val, y_val, param_space):
    # Pick random hyperparameters
    lr = random.choice(param_space["learning_rate"])
    batch_size = random.choice(param_space["batch_size"])
    layers_size = random.choice(param_space["hidden_layer_sizes"])
    dropout = random.choice(param_space["dropout"])
    l2_reg = random.choice(param_space["l2"])
    activation = random.choice(param_space["activation"])
    epochs = random.choice(param_space["epochs"])

    # Build the model dynamically
    model = models.Sequential()
    model.add(layers.Input(shape=(X_train.shape[1],)))
    
    for size in layers_size:
        model.add(layers.Dense(size, kernel_regularizer=regularizers.l2(l2_reg)))
        if activation == "leaky_relu":
            model.add(LeakyReLU(alpha=0.1))
        else:
            model.add(layers.Activation(activation))
        model.add(layers.Dropout(dropout))

    
    model.add(layers.Dense(1, activation="sigmoid"))
    
    # Compile
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy"] 
    )
    
    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0
    )
    
    # Evaluate
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    
    return {"val_loss": val_loss, "val_accuracy": val_acc, "params": {
        "lr": lr, "batch_size": batch_size, "layers": layers_size, 
        "dropout": dropout, "l2": l2_reg, "activation": activation, "epochs": epochs
    }}

def random_search_trial_r(X_train, y_train, X_val, y_val, param_space):
    # Pick random hyperparameters
    lr = random.choice(param_space["learning_rate"])
    batch_size = random.choice(param_space["batch_size"])
    layers_size = random.choice(param_space["hidden_layer_sizes"])
    dropout = random.choice(param_space["dropout"])
    l2_reg = random.choice(param_space["l2"])
    activation = random.choice(param_space["activation"])
    epochs = random.choice(param_space["epochs"])

    # Build the model dynamically
    model = models.Sequential()
    model.add(layers.Input(shape=(X_train.shape[1],)))
    
    for size in layers_size:
        model.add(layers.Dense(size, kernel_regularizer=regularizers.l2(l2_reg)))
        if activation == "leaky_relu":
            model.add(LeakyReLU(alpha=0.1))
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

    
    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0
    )
    
    # Evaluate
    val_loss, val_mae = model.evaluate(X_val, y_val, verbose=0)
    
    return {"val_loss": val_loss, "val_mae": val_mae, "params": {
        "lr": lr, "batch_size": batch_size, "layers": layers_size, 
        "dropout": dropout, "l2": l2_reg, "activation": activation, "epochs": epochs
    }}