"""
preprocessing.py
----------------
Handles all data loading, cleaning, encoding, and scaling.
Functions are reusable and modular — import them anywhere.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

# ── Global scaler & encoders so we can reuse them during prediction ──────────
scaler = StandardScaler()
label_encoders = {}          # one LabelEncoder per categorical column


def load_data(filepath: str) -> pd.DataFrame:
    """Load CSV file from the given path and return a DataFrame."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at: {filepath}")
    df = pd.read_csv(filepath)
    print(f"[INFO] Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values:
      - Numeric columns  → median
      - Categorical cols → mode
    """
    for col in df.columns:
        if df[col].isnull().any():
            if df[col].dtype in [np.float64, np.int64, float, int]:
                df[col].fillna(df[col].median(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)
    print("[INFO] Missing values handled.")
    return df


def encode_categorical(df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
    """
    Label-encode all object (string) columns.
    Use fit=True during training, fit=False (transform only) during prediction.
    """
    global label_encoders
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()

    for col in cat_cols:
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
        else:
            if col in label_encoders:
                le = label_encoders[col]
                # Handle unseen labels gracefully
                df[col] = df[col].astype(str).apply(
                    lambda x: x if x in le.classes_ else le.classes_[0]
                )
                df[col] = le.transform(df[col])

    print(f"[INFO] Encoded columns: {cat_cols}")
    return df


def scale_features(X: pd.DataFrame, fit: bool = True) -> np.ndarray:
    """
    Standardise features using StandardScaler.
    Use fit=True during training, fit=False during prediction.
    """
    global scaler
    if fit:
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    print("[INFO] Feature scaling done.")
    return X_scaled


def preprocess(filepath: str):
    """
    Full pipeline: load → clean → encode → scale.
    Returns:
        X_scaled  - numpy array of features
        y         - pandas Series of target values
        df        - cleaned DataFrame (useful for visualisations)
    """
    df = load_data(filepath)
    df = handle_missing_values(df)
    df = encode_categorical(df, fit=True)

    # Target column
    target = 'Selling_Price'
    feature_cols = [c for c in df.columns if c != target]

    X = df[feature_cols]
    y = df[target]

    X_scaled = scale_features(X, fit=True)
    return X_scaled, y, df


def preprocess_single(input_dict: dict, feature_cols: list) -> np.ndarray:
    """
    Preprocess a single user-input row for prediction.
    input_dict  - {column_name: value, ...}
    feature_cols - ordered list of feature names used during training
    """
    row_df = pd.DataFrame([input_dict])
    row_df = handle_missing_values(row_df)
    row_df = encode_categorical(row_df, fit=False)

    # Align columns to training order; fill any missing with 0
    for col in feature_cols:
        if col not in row_df.columns:
            row_df[col] = 0
    row_df = row_df[feature_cols]

    return scale_features(row_df, fit=False)
