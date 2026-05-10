import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def load_and_preprocess(filepath):
    """
    Load dataset, clean it, encode labels, handle class imbalance,
    and scale features.
    Returns train/val/test splits as numpy arrays.
    """
    df = pd.read_csv(filepath)

    # ── 1. Drop header-like rows (first two rows are metadata in your dataset)
    df = df[df['loc'].apply(lambda x: _is_number(str(x)))]
    df = df.reset_index(drop=True)

    # ── 2. Convert all columns to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # ── 3. Drop rows with any NaN
    df = df.dropna()

    # ── 4. Encode target: TRUE → 1, FALSE → 0
    # 'defects' is already numeric (1.0 / 0.0) after step 2;
    # TRUE becomes 1.0, FALSE becomes 0.0
    X = df.drop(columns=['defects']).values.astype(np.float32)
    y = df['defects'].values.astype(np.int64)

    print(f"Dataset shape: {X.shape}")
    print(f"Class distribution — Defective: {y.sum()}, Clean: {(y==0).sum()}")

    # ── 5. Train / val / test split (70 / 15 / 15)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    # ── 6. Feature scaling (fit only on train)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    # ── 7. SMOTE on training set only (handle class imbalance)
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE — Defective: {y_train.sum()}, Clean: {(y_train==0).sum()}")

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler


def _is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
