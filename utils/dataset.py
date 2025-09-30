import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple

def load_dataset(path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # extract csv data
    df = pd.read_csv(path)

    # Split into feature and label dataset
    y = df[["Id", "SalePrice"]]
    X = df.drop(columns=["Id", "SalePrice"])

    # fill null values
    X = _fill_na(X)

    # split, scale and normalize
    X_train, X_cv, y_train, y_cv = _split_and_normalize_datasets(X, y, cv_split=0.3, shuffle=True)

    return (X_train, X_cv, y_train, y_cv)



def _fill_na(X: pd.DataFrame) -> pd.DataFrame:
    df = copy.deepcopy(X)

    na_columns = {col: i for i, col in enumerate(df.columns) if df[col].isna().any()}

    # Extracts columns from na_columns whose datatype is not numeric
    nan_cols = {col: idx for col, idx in na_columns.items() if df[col].dtype == type(object)}

    # Extract numerical cols with Null values
    num_cols_na = {col: idx for col, idx in na_columns.items() if nan_cols.get(col, False) is False}

    # Extract the median for each column
    na_cols_median_vals = {col: df[col].median() for col in num_cols_na.keys()}

    #Fill NaNs
    for col in num_cols_na:
        df[col] = df[col].fillna(na_cols_median_vals[col])

    # Fill cols with 'NA'
    # > if previous logic does not work, try this: For non-numeric columns, populate NULL values with the most frequent used value
    for col in nan_cols:
        df[col] = df[col].fillna("Missing")

    return df

def _split_and_normalize_datasets(X: pd.DataFrame, y: pd.DataFrame, cv_split: float, shuffle: bool) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Extract all numerical cols
    numerical_cols = [col for col in X.columns if X[col].dtype != type(object)]

    # Encode categorical values
    df = _encode_categorical_values(X)

        # Extract column names
    column_names = df.columns

    # Split dataset into train and cv
    X_train, X_cv, y_train, y_cv = train_test_split(df, y, test_size=cv_split, shuffle=shuffle)

    # Normalize train dataset
    # > First log all features with large scale
    # > Then normalize
    cols_to_log = [col for col in numerical_cols if X[col].max() >= 500]
    X_train_log = copy.deepcopy(X_train)
    scaler = StandardScaler()
    X_train_scaled = _scale_and_normalize(X_train_log, column_names, scaler, cols_to_log)

    # ***CV Dataset***
    X_cv_log = copy.deepcopy(X_cv)
    X_cv_scaled = _scale_and_normalize(X_cv_log, column_names, scaler, cols_to_log)

    return (X_train, X_cv, y_train, y_cv)


def _encode_categorical_values(X: pd.DataFrame) -> pd.DataFrame:
    df = copy.deepcopy(X)

    # Extract all categorical columns
    categorical_cols = [col for col in df.columns if df[col].dtype == type(object)]

    # Lets frequency encode moderate and high cardinality columns, and one-hot encode low cardinality columns
    for col in categorical_cols:
        num_of_categories = df[col].nunique()

        # Frequency encode if num of categories >= 5
        if num_of_categories >= 5:
            cat_frequencies = df[col].value_counts(normalize=True)
            df[col] = df[col].map(cat_frequencies)
        else: # one-hot encode all others
            one_hot_encoded = pd.get_dummies(df[col], prefix=col, dummy_na=False, dtype="uint8")
            df = pd.concat([df.drop(columns=[col]), one_hot_encoded], axis=1)

    return df

def _scale_and_normalize(X: pd.DataFrame, column_names: pd.Index, scaler: StandardScaler, columns_to_log) -> Tuple[pd.DataFrame, pd.DataFrame]:

    # > First log scale all columns in columns_to_log
    for col in columns_to_log:
        X[col] = X[col].map(lambda x: np.log1p(x))

    # Normalize train dataset with z-score
    index = X.index
    X_train_scaled = scaler.fit_transform(X) # numpy array
    X_train_scaled_df = pd.DataFrame(X_train_scaled, index=index, columns=column_names)

    return X_train_scaled_df
    