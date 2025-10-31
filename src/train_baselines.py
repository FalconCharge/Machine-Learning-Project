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




# Start MLflow experiment
mlflow.set_experiment("Student Performance Baselines")


# Hyperparameter tuning for Logistic Regression
best_val_acc = 0
best_model = None
best_c = None

for c in [0.2, 0.24, 0.25, 0.26, 0.3]:
    model = LogisticRegression(C=c, max_iter=1000, random_state=42)
    model.fit(X_train_c, y_train_c)
    val_preds = model.predict(X_val_c)
    val_acc = classification_metrics(y_val_c, val_preds)["accuracy"]

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model = model
        best_c = c

print(f"Best C: {best_c} with validation accuracy: {best_val_acc:.4f}")


# Logistic Regression
with mlflow.start_run(run_name="LogisticRegression_Classification"):
    model = LogisticRegression(C=best_c, max_iter=1000, random_state=42)
    model.fit(X_train_c, y_train_c)
    preds = model.predict(X_test_c)

    metrics = classification_metrics(y_test_c, preds)
    mlflow.log_params({"model_type": "LogisticRegression"})
    mlflow.log_metrics(metrics)
    log_confusion_matrix(y_test_c, preds, "LogisticRegression")
    mlflow.sklearn.log_model(model, name="LogisticRegression", input_example=X_test_c[:5])


# Isn't really any parameters to tune for Naive Bayes

# Naive Bayes
with mlflow.start_run(run_name="NaiveBayes_Classification"):
    model = GaussianNB()
    model.fit(X_train_c, y_train_c)
    preds = model.predict(X_test_c)

    metrics = classification_metrics(y_test_c, preds)
    mlflow.log_params({"model_type": "NaiveBayes"})
    mlflow.log_metrics(metrics)
    log_confusion_matrix(y_test_c, preds, "NaiveBayes")
    mlflow.sklearn.log_model(model, name="NaiveBayes", input_example=X_test_c[:5])

# Isn't really any parameters to tune for Linear Regression

# Linear Regression
with mlflow.start_run(run_name="LinearRegression_Regression"):
    model = LinearRegression()
    model.fit(X_train_r, y_train_r)
    preds = model.predict(X_test_r)

    metrics = regression_metrics(y_test_r, preds)
    mlflow.log_params({"model_type": "LinearRegression"})
    mlflow.log_metrics(metrics)
    log_residual_plot(y_test_r, preds, "LinearRegression")
    mlflow.sklearn.log_model(model, name="LinearRegression", input_example=X_test_r)

# Hyperparameter tuning for Decision Tree Regressor
best_val_mae = float("inf")
best_model = None
best_depth = None

for depth in range(1, 12):
    
    model = DecisionTreeRegressor(max_depth=depth, random_state=42)
    model.fit(X_train_r, y_train_r)
    
    val_preds = model.predict(X_val_r)
    val_metrics = regression_metrics(y_val_r, val_preds)
    val_mae = val_metrics["MAE"]
        
    if val_mae < best_val_mae:
        best_val_mae = val_mae
        best_model = model
        best_depth = depth

print(f"\nBest max_depth: {best_depth} with Validation MAE: {best_val_mae:.4f}")

# Log the best model to MLflow
with mlflow.start_run(run_name=f"DecisionTree_Regression_best"):
    preds = best_model.predict(X_test_r)
    metrics = regression_metrics(y_test_r, preds)
    
    mlflow.log_params({"model_type": "DecisionTreeRegressor", "max_depth": best_depth})
    mlflow.log_metrics(metrics)
    log_residual_plot(y_test_r, preds, "DecisionTreeRegressor")
    mlflow.sklearn.log_model(best_model, name="DecisionTreeRegressor", input_example=X_test_r)
