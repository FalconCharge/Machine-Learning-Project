import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeRegressor

from data import runDataProcessing, load_processed_data
from evaluate import classification_metrics, regression_metrics
from utils import log_confusion_matrix, log_residual_plot

# Run data.py to process and split the data with the features.py for normalization and encoding
runDataProcessing()

# Load classification data
train_class = load_processed_data("data/processed/splits/train_classification.csv")
val_class = load_processed_data("data/processed/splits/val_classification.csv")
test_class = load_processed_data("data/processed/splits/test_classification.csv")

# Load regression data
train_regres = load_processed_data("data/processed/splits/train_regression.csv")
val_regres = load_processed_data("data/processed/splits/val_regression.csv")
test_regres = load_processed_data("data/processed/splits/test_regression.csv")



# For classification
X_train_c = train_class.drop(columns=["G3", "Pass"])
y_train_c = train_class["Pass"]

X_val_c = val_class.drop(columns=["G3", "Pass"])
y_val_c = val_class["Pass"]

X_test_c = test_class.drop(columns=["G3", "Pass"])
y_test_c = test_class["Pass"]

# For regression

X_train_r = train_regres.drop(columns=["G3", "Pass"])
y_train_r = train_regres["G3"]

X_val_r = val_regres.drop(columns=["G3", "Pass"])
y_val_r = val_regres["G3"]

X_test_r = test_regres.drop(columns=["G3", "Pass"])
y_test_r = test_regres["G3"]

def train_decision_tree_regressor():
    # Hyperparameter tuning for Decision Tree Regressor
    best_model = None
    best_depth = None
    best_val_mae = float("inf")
    best_val_rmse = float("inf")

    for depth in range(1, 12):
        model = DecisionTreeRegressor(max_depth=depth, random_state=42)
        model.fit(X_train_r, y_train_r)

        val_preds = model.predict(X_val_r)
        val_metrics = regression_metrics(y_val_r, val_preds)
        val_mae = val_metrics["MAE"]
        val_rmse = val_metrics["RMSE"]

        print(f"depth={depth:2d}  val_MAE={val_mae:.4f}  val_RMSE={val_rmse:.4f}")

        # Select best by MAE, tie-break by RMSE
        if val_mae < best_val_mae or (val_mae == best_val_mae and val_rmse < best_val_rmse):
            best_val_mae = val_mae
            best_val_rmse = val_rmse
            best_depth = depth
            best_model = model

    print(f"\nBest max_depth={best_depth} | Val MAE={best_val_mae:.4f}, Val RMSE={best_val_rmse:.4f}")

    # Log the best model to MLflow
    with mlflow.start_run(run_name=f"DecisionTree_Regression_best"):
        model = best_model
        preds = model.predict(X_test_r)
        metrics = regression_metrics(y_test_r, preds)
        
        mlflow.log_params({"model_type": "DecisionTreeRegressor", "max_depth": best_depth})
        mlflow.log_metrics(metrics)
        log_residual_plot(y_test_r, preds, "DecisionTreeRegressor_Test")
        mlflow.sklearn.log_model(model, name="DecisionTreeRegressor", input_example=X_test_r)

    # Log the best validation model to MLflow
    with mlflow.start_run(run_name=f"DecisionTree_Regression_best_validation"):
        model = best_model
        preds = model.predict(X_val_r)
        metrics = regression_metrics(y_val_r, preds)
        
        mlflow.log_params({"model_type": "DecisionTreeRegressor", "max_depth": best_depth})
        mlflow.log_metrics(metrics)
        log_residual_plot(y_val_r, preds, "DecisionTreeRegressor_Validation")
        mlflow.sklearn.log_model(model, name="DecisionTreeRegressor", input_example=X_val_r)

def train_logistic_regression():
    # Hyperparameter tuning for Logistic Regression
    best_val_acc = 0
    best_score = 0
    best_model = None
    best_c = None

    for c in [0.2, 0.25, 0.27, 0.3, 0.4]:
        model = LogisticRegression(C=c, max_iter=1000)
        model.fit(X_train_c, y_train_c)

        preds_val = model.predict(X_val_c)
        metrics = classification_metrics(y_val_c, preds_val)
        val_acc = metrics["accuracy"]
        val_f1 = metrics["f1_score"]

        print(f"C={c}, val_acc={val_acc:.3f}, val_f1={val_f1:.3f}")

        if val_f1 > best_score or (val_f1 == best_score and val_acc > best_val_acc):
            best_score = val_f1
            best_val_acc = val_acc
            best_c = c
            best_model = model

    print(f"Best C: {best_c} with validation accuracy: {best_val_acc:.4f}")
    print(f"Best C: {best_c} with Fl-Score: {best_score:.4f}")


    # Logistic Regression
    with mlflow.start_run(run_name="LogisticRegression_T_Classification"):
        model = best_model
        model.fit(X_train_c, y_train_c)
        preds = model.predict(X_test_c)

        metrics = classification_metrics(y_test_c, preds)
        mlflow.log_params({"model_type": "LogisticRegression"})
        mlflow.log_metrics(metrics)
        log_confusion_matrix(y_test_c, preds, "LogisticRegression_test")
        mlflow.sklearn.log_model(model, name="LogisticRegression", input_example=X_test_c[:5])
    
    # Logistic Regression validation
    with mlflow.start_run(run_name="LogisticRegression_V_Classification"):
        model = best_model
        model.fit(X_train_c, y_train_c)
        preds = model.predict(X_val_c)

        metrics = classification_metrics(y_val_c, preds)
        mlflow.log_params({"model_type": "LogisticRegression"})
        mlflow.log_metrics(metrics)
        log_confusion_matrix(y_val_c, preds, "LogisticRegression_val")
        mlflow.sklearn.log_model(model, name="LogisticRegression", input_example=X_val_c[:5])

def train_linear_regression():
    # Isn't really any parameters to tune for Linear Regression
    # Linear Regression
    with mlflow.start_run(run_name="LinearRegression_T_Regression"):
        model = LinearRegression()
        model.fit(X_train_r, y_train_r)
        preds = model.predict(X_test_r)

        metrics = regression_metrics(y_test_r, preds)
        mlflow.log_params({"model_type": "LinearRegression"})
        mlflow.log_metrics(metrics)
        log_residual_plot(y_test_r, preds, "LinearRegression_test")
        mlflow.sklearn.log_model(model, name="LinearRegression", input_example=X_test_r)

    # Linear Regression validation
    with mlflow.start_run(run_name="LinearRegression_V_Regression"):
        model = LinearRegression()
        model.fit(X_train_r, y_train_r)
        preds = model.predict(X_val_r)

        metrics = regression_metrics(y_val_r, preds)
        mlflow.log_params({"model_type": "LinearRegression"})
        mlflow.log_metrics(metrics)
        log_residual_plot(y_val_r, preds, "LinearRegression_Validation")
        mlflow.sklearn.log_model(model, name="LinearRegression", input_example=X_val_r)

def train_naive_bayes():
    # Isn't really any parameters to tune for Naive Bayes
    # Naive Bayes
    with mlflow.start_run(run_name="NaiveBayes_T_Classification"):
        model = GaussianNB()
        model.fit(X_train_c, y_train_c)
        preds = model.predict(X_test_c)

        metrics = classification_metrics(y_test_c, preds)
        mlflow.log_params({"model_type": "NaiveBayes"})
        mlflow.log_metrics(metrics)
        log_confusion_matrix(y_test_c, preds, "NaiveBayes_Test")
        mlflow.sklearn.log_model(model, name="NaiveBayes", input_example=X_test_c[:5])

    # Naive Bayes validation
    with mlflow.start_run(run_name="NaiveBayes_V_Classification"):
        model = GaussianNB()
        model.fit(X_train_c, y_train_c)
        preds = model.predict(X_val_c)

        metrics = classification_metrics(y_val_c, preds)
        mlflow.log_params({"model_type": "NaiveBayes"})
        mlflow.log_metrics(metrics)
        log_confusion_matrix(y_val_c, preds, "NaiveBayes_val")
        mlflow.sklearn.log_model(model, name="NaiveBayes", input_example=X_val_c[:5])

# Start MLflow experiment
mlflow.set_experiment("Student Performance Baselines")

# Train them all
train_logistic_regression()
train_decision_tree_regressor()
train_naive_bayes()
train_linear_regression()
