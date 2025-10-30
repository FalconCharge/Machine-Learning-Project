import pandas as pd
from sklearn.preprocessing import StandardScaler

def select_features(df):
    selected = ["traveltime", "studytime", "failures", "schoolsup", "paid",
                "activities", "higher", "internet", "freetime", "absences", "G1", "G2", "G3", "Pass"]
    return df[selected]

def preprocess_features(X, target_col=None, scaler=None):
    if target_col:
        X = X.drop(columns=[target_col])
    
    X = pd.get_dummies(X, drop_first=True)

    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    
    return X_scaled, scaler

