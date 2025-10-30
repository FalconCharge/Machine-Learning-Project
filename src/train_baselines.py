import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeRegressor

import pandas as pd

from data import load_student_data, split_data
from features import select_features, preprocess_features
from evaluate import classification_metrics, regression_metrics
from utils import log_confusion_matrix, log_residual_plot

# Load data
df = load_student_data()
df = select_features(df)

# Classification
X_train_c, X_test_c, y_train_c, y_test_c = split_data(df, target_col="Pass", test_size=0.2, stratify=df["Pass"])
X_train_c, scaler_c = preprocess_features(X_train_c, target_col=None)
X_test_c, _ = preprocess_features(X_test_c, target_col=None, scaler=scaler_c)

# Regression
X_train_r, X_test_r, y_train_r, y_test_r = split_data(df, target_col="G3", test_size=0.2)
X_train_r, scaler_r = preprocess_features(X_train_r, target_col=None)
X_test_r, _ = preprocess_features(X_test_r, target_col=None, scaler=scaler_r)


# Start MLflow experiment
mlflow.set_experiment("Student Performance Baselines")

# Logistic Regression
with mlflow.start_run(run_name="LogisticRegression_Classification"):
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_c, y_train_c)
    preds = model.predict(X_test_c)

    metrics = classification_metrics(y_test_c, preds)
    mlflow.log_params({"model_type": "LogisticRegression"})
    mlflow.log_metrics(metrics)
    log_confusion_matrix(y_test_c, preds, "LogisticRegression")
    mlflow.sklearn.log_model(model, name="LogisticRegression", input_example=X_test_c[:5])

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

# Linear Regression
with mlflow.start_run(run_name="LinearRegression_Regression"):
    model = LinearRegression()
    model.fit(X_train_r, y_train_r)
    preds = model.predict(X_test_r)

    metrics = regression_metrics(y_test_r, preds)
    mlflow.log_params({"model_type": "LinearRegression"})
    mlflow.log_metrics(metrics)
    log_residual_plot(y_test_r, preds, "LinearRegression")
    mlflow.sklearn.log_model(model, name="LinearRegression", input_example=X_test_c[:5])

# Decision Tree
with mlflow.start_run(run_name="DecisionTree_Regression"):
    model = DecisionTreeRegressor(max_depth=5, random_state=42)
    model.fit(X_train_r, y_train_r)
    preds = model.predict(X_test_r)

    metrics = regression_metrics(y_test_r, preds)
    mlflow.log_params({"model_type": "DecisionTreeRegressor", "max_depth": 5})
    mlflow.log_metrics(metrics)
    log_residual_plot(y_test_r, preds, "DecisionTreeRegressor")
    mlflow.sklearn.log_model(model, name="DecisionTreeRegressor", input_example=X_test_c[:5])
