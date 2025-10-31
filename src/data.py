import os
import pandas as pd
from sklearn.model_selection import train_test_split
from features import create_feature_pipeline, add_engineered_features, preprocess_features
RAW_PATH = "data/raw/student-por.csv"
PROCESSED_PATH = "data/processed/student-por.csv"


def load_raw_data(path: str = RAW_PATH, delimiter: str = ";") -> pd.DataFrame:
    # Load the raw dataset
    df = pd.read_csv(path, delimiter=delimiter)
    return df

def load_processed_data(path: str = PROCESSED_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def load_csv_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def split_data(df: pd.DataFrame, target_col: str, test_size: float = 0.2, random_state: int = 42, stratify=None):
    y = df[target_col]
    X = df.drop(columns=[target_col])
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)


def process_and_save_data(df, out_path=PROCESSED_PATH):
    df = df.copy()
    df_target_class = (df["G3"] >= 10).astype(int)
    df_target_reg = df["G3"]

    X = df.drop(columns=["G3"])
    processed_df = preprocess_features(X)

    # Add targets back
    processed_df['Pass'] = df_target_class.values
    processed_df['G3'] = df_target_reg.values

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    processed_df.to_csv(out_path, index=False)
    return processed_df


def split_and_save_data(df: pd.DataFrame, target_col: str = "Pass", test_size: float = 0.2, val_size: float = 0.25, random_state: int = 42, stratify=None, out_dir: str = "data/processed/splits"
):

    # Split train/test first
    strat = df[target_col] if stratify is not None else None
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=strat
    )

    # Split validation from train
    strat_val = train_df[target_col] if stratify is not None else None
    val_df, train_df = train_test_split(
        train_df, train_size=val_size, random_state=random_state, stratify=strat_val
    )

    # save the csv's
    os.makedirs(out_dir, exist_ok=True)
    if target_col == "Pass":
        train_df_path = f"{out_dir}/train_classification.csv"
        val_df_path = f"{out_dir}/val_classification.csv"
        test_df_path = f"{out_dir}/test_classification.csv"
    else:
        train_df_path = f"{out_dir}/train_regression.csv"
        val_df_path = f"{out_dir}/val_regression.csv"
        test_df_path = f"{out_dir}/test_regression.csv"

    train_df.to_csv(train_df_path, index=False)
    val_df.to_csv(val_df_path, index=False)
    test_df.to_csv(test_df_path, index=False)

    print(f" Saved train/val/test to {out_dir}")
    print(f" Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    return train_df, val_df, test_df


# for use in other scripts
def runDataProcessing():
    raw_df = load_raw_data()
    processed_df = process_and_save_data(raw_df)
    split_and_save_data(processed_df, target_col="Pass", stratify=processed_df["Pass"])
    split_and_save_data(processed_df, target_col="G3", stratify=None)
    print("Data processing and splitting complete.")
    return processed_df


# Should be used before using the models Run with -> python src/data.py
if __name__ == "__main__":
    runDataProcessing()
