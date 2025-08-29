import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess(filepath, target_col="Outcome"):
    """Load dataset, clean missing values, split into train/test, and scale features."""
    df = pd.read_csv(filepath)

    # Replace 0 with NaN for specific columns
    cols_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df[cols_with_zero] = df[cols_with_zero].replace(0, np.nan)

    # Fill missing values with median
    df.fillna(df.median(), inplace=True)

    # Split features/target
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test
