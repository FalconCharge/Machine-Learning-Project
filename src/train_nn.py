import mlflow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.metrics import F1Score
import random


from data import runDataProcessing, load_processed_data
from evaluate import classification_metrics, regression_metrics
from utils import log_confusion_matrix, log_residual_plot, random_search_trial_c, random_search_trial_r

mlflow.set_tracking_uri("http://127.0.0.1:5000")






def train_nn_classifier():
    # Build a simple feed-forward NN
    model = keras.Sequential([
        keras.Input(shape=(X_train_c.shape[1],)),  
        layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.2),
        layers.Dense(16, activation="relu", kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid", kernel_regularizer=regularizers.l2(0.001))
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["f1_score"]
    )

    # Train
    history = model.fit(
        X_train_c, y_train_c,
        validation_data=(X_val_c, y_val_c),
        epochs=30,
        batch_size=32,
        verbose=0
    )

    # Evaluate on test
    test_loss, test_acc = model.evaluate(X_test_c, y_test_c, verbose=0)
    preds = (model.predict(X_test_c) > 0.5).astype(int).flatten()

    metrics = classification_metrics(y_test_c, preds)

    # Log with MLflow
    with mlflow.start_run(run_name="NeuralNetwork_Classification"):
        mlflow.log_params({"model_type": "Keras_FFNN"})
        mlflow.log_metrics(metrics)
        log_confusion_matrix(y_test_c, preds, "Keras_NN_Classification")
        mlflow.keras.log_model(model, "KerasNN_Classifier")

def train_nn_regression():
    model = keras.Sequential([
        keras.Input(shape=(X_train_r.shape[1],)),  
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.1),
        layers.Dense(16, activation="relu"),
        layers.Dense(1) 
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0005),
        loss="mse",
        metrics=["mae"]
    )

    history = model.fit(
        X_train_r, y_train_r,
        validation_data=(X_val_r, y_val_r),
        epochs=40,
        batch_size=32,
        verbose=0
    )

    # Predict
    preds = model.predict(X_test_r).flatten()
    metrics = regression_metrics(y_test_r, preds)

    # MLflow logging
    with mlflow.start_run(run_name="NeuralNetwork_Regression"):
        mlflow.log_params({"model_type": "KerasRegressionNN"})
        mlflow.log_metrics(metrics)
        log_residual_plot(y_test_r, preds, "Keras_NN_Regression")
        mlflow.keras.log_model(model, "KerasNN_Regressor")

def train_nn_classifier(X_train, y_train, X_val, y_val, X_test, y_test, best_params):
    model = keras.Sequential()
    model.add(layers.Input(shape=(X_train.shape[1],)))

    for size in best_params["layers"]:
        if best_params["activation"] == "leaky_relu":
            model.add(layers.Dense(size, kernel_regularizer=regularizers.l2(best_params["l2"])))
            model.add(layers.LeakyReLU(alpha=0.1))
        else:
            model.add(layers.Dense(size, activation=best_params["activation"],
                                   kernel_regularizer=regularizers.l2(best_params["l2"])))
        model.add(layers.Dropout(best_params["dropout"]))

    model.add(layers.Dense(1, activation="sigmoid"))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=best_params["lr"]),
        loss="binary_crossentropy",
        metrics=["accuracy", F1Score(name="f1_score")]
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=best_params["epochs"],
        batch_size=best_params["batch_size"],
        verbose=0
    )

    preds = (model.predict(X_test) > 0.5).astype(int).flatten()
    metrics = classification_metrics(y_test, preds)

    with mlflow.start_run(run_name="Final_KerasNN_Classifier"):
        mlflow.log_params(best_params)
        mlflow.log_metrics(metrics)
        log_confusion_matrix(y_test, preds, "Final_KerasNN_Classifier")
        mlflow.keras.log_model(model, "Final_KerasNN_Classifier")

    return model

def train_nn_regression(X_train, y_train, X_val, y_val, X_test, y_test, best_params):
    model = keras.Sequential()
    model.add(layers.Input(shape=(X_train.shape[1],)))

    for size in best_params["layers"]:
        if best_params["activation"] == "leaky_relu":
            model.add(layers.Dense(size, kernel_regularizer=regularizers.l2(best_params["l2"])))
            model.add(layers.LeakyReLU(alpha=0.1))
        else:
            model.add(layers.Dense(size, activation=best_params["activation"],
                                   kernel_regularizer=regularizers.l2(best_params["l2"])))
        model.add(layers.Dropout(best_params["dropout"]))

    model.add(layers.Dense(1))  # regression output

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=best_params["lr"]),
        loss="mse",
        metrics=["mae"]
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=best_params["epochs"],
        batch_size=best_params["batch_size"],
        verbose=0
    )

    preds = model.predict(X_test).flatten()
    metrics = regression_metrics(y_test, preds)

    with mlflow.start_run(run_name="Final_KerasNN_Regression"):
        mlflow.log_params(best_params)
        mlflow.log_metrics(metrics)
        log_residual_plot(y_test, preds, "Final_KerasNN_Regression")
        mlflow.keras.log_model(model, "Final_KerasNN_Regression")

    return model




def main():

    # 1. Process data
    runDataProcessing()

    # 2. Load classification data
    train_class = load_processed_data("data/processed/splits/train_classification.csv")
    val_class = load_processed_data("data/processed/splits/val_classification.csv")
    test_class = load_processed_data("data/processed/splits/test_classification.csv")

    # 3. Load regression data
    train_regres = load_processed_data("data/processed/splits/train_regression.csv")
    val_regres = load_processed_data("data/processed/splits/val_regression.csv")
    test_regres = load_processed_data("data/processed/splits/test_regression.csv")

    # 4. Build classification splits
    global X_train_c, y_train_c, X_val_c, y_val_c, X_test_c, y_test_c
    X_train_c = train_class.drop(columns=["G3", "Pass"])
    y_train_c = train_class["Pass"]

    X_val_c = val_class.drop(columns=["G3", "Pass"])
    y_val_c = val_class["Pass"]

    X_test_c = test_class.drop(columns=["G3", "Pass"])
    y_test_c = test_class["Pass"]

    # 5. Build regression splits
    global X_train_r, y_train_r, X_val_r, y_val_r, X_test_r, y_test_r
    X_train_r = train_regres.drop(columns=["G3", "Pass"])
    y_train_r = train_regres["G3"]

    X_val_r = val_regres.drop(columns=["G3", "Pass"])
    y_val_r = val_regres["G3"]

    X_test_r = test_regres.drop(columns=["G3", "Pass"])
    y_test_r = test_regres["G3"]


    param_space = {
        "learning_rate": [0.0001, 0.001, 0.005, 0.01],
        "batch_size": [16, 32, 64],
        "hidden_layer_sizes": [(64, 32), (128, 64), (32, 32)],
        "dropout": [0.1, 0.2, 0.3, 0.5],
        "l2": [0.0001, 0.001, 0.01],
        "activation": ["relu", "tanh", "leaky_relu"],
        "epochs": [50, 100, 150, 200]
    }

    best_c = None
    for _ in range(10):
        result = random_search_trial_c(X_train_c, y_train_c, X_val_c, y_val_c, param_space)
        print(result)
        if best_c is None or result["val_accuracy"] > best_c["val_accuracy"]:
            best_c = result

    print("Best hyperparameters:", best_c)

    best_r = None
    for _ in range(10):
        result = random_search_trial_r(X_train_r, y_train_r, X_val_r, y_val_r, param_space)
        print(result)
        if best_r is None or result["val_mae"] < best_r["val_mae"]:
            best_r = result


    print("Best hyperparameters:", best_r)

    # Train final models using best hyperparameters
    final_class_model = train_nn_classifier(X_train_c, y_train_c, X_val_c, y_val_c, X_test_c, y_test_c, best_c["params"])
    final_reg_model = train_nn_regression(X_train_r, y_train_r, X_val_r, y_val_r, X_test_r, y_test_r, best_r["params"])


if __name__ == "__main__":
    main()


