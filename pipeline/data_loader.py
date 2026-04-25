"""
Data Loader - Load and profile CSV datasets
"""
import pandas as pd
import numpy as np


def load_and_profile(path: str, is_test: bool = False) -> tuple:
    df = pd.read_csv(path)

    profile = {
        'rows': int(len(df)),
        'cols': int(len(df.columns)),
        'columns': df.columns.tolist(),
        'dtypes': {col: str(dt) for col, dt in df.dtypes.items()},
        'missing': {col: int(v) for col, v in df.isnull().sum().items() if v > 0},
        'missing_pct': {col: round(float(v / len(df) * 100), 2)
                        for col, v in df.isnull().sum().items() if v > 0},
        'duplicates': int(df.duplicated().sum()),
        'is_test': is_test,
    }

    if not is_test and 'target' in df.columns:
        vc = df['target'].value_counts()
        profile['target_dist'] = {str(int(k)): int(v) for k, v in vc.items()}
        profile['class_balance'] = round(float(vc.min() / vc.max()), 3)

    # Numeric summary
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        desc = df[num_cols].describe().round(3)
        profile['numeric_summary'] = desc.to_dict()

    # Categorical value counts (top 5)
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    profile['cat_summary'] = {}
    for col in cat_cols:
        vc = df[col].value_counts().head(5)
        profile['cat_summary'][col] = {str(k): int(v) for k, v in vc.items()}

    return df, profile
