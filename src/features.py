import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["avg_prev_grades"] = (df["G1"] + df["G2"]) / 2
    df['fail_abs_ratio'] = df['failures'] / (df['absences'] + 1)
    return df

def encode_binary_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    binary_cols = [
        "schoolsup", "famsup", "paid", "activities", "nursery",
        "higher", "internet", "romantic", "sex"
    ]
    for col in binary_cols:
        df[col] = df[col].map({"yes": 1, "no": 0, "M": 1, "F": 0})
    return df

def create_feature_pipeline(df: pd.DataFrame):
    df = add_engineered_features(df)
    df = encode_binary_features(df)

    categorical_cols = ["school", "address", "famsize", "Pstatus", "Mjob", "Fjob", "reason", "guardian"]

    numeric_cols = [
        "age", "Medu", "Fedu", "traveltime", "studytime", "failures",
        "famrel", "freetime", "goout", "Dalc", "Walc", "health",
        "absences", "G1", "G2", "avg_prev_grades", "fail_abs_ratio"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(drop="first", sparse_output=False), categorical_cols),
        ],
        remainder="drop"
    )
    return preprocessor

def preprocess_features(df: pd.DataFrame):
    df = add_engineered_features(df)
    df = encode_binary_features(df)
    
    # Optional: convert categorical columns to dummy variables first
    categorical_cols = ["school", "address", "famsize", "Pstatus", "Mjob", "Fjob", "reason", "guardian"]
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # Convert all columns to float to satisfy MLflow
    df = df.astype(float)
    
    # Scale numeric columns
    numeric_cols = df.columns.drop(["Pass", "G3"], errors="ignore")  # leave out targets
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    return df
