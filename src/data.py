import pandas as pd
from sklearn.model_selection import train_test_split

def load_student_data(path="data/raw/student-por.csv", delimiter=";"):
    df = pd.read_csv(path, delimiter=delimiter)
    df["Pass"] = (df['G3'] >= 10).astype(int)
    return df

def split_data(df, target_col, test_size=0.2, random_state=42, stratify=None):
    y = df[target_col]
    X = df.drop(columns=[target_col])
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)
