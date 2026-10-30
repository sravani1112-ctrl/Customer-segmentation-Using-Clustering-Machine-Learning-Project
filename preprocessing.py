import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder


def clean_data(df):

    df = df.copy()

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Clean column names
    df.columns = df.columns.str.strip()

    return df


def preprocess_data(df):

    df = clean_data(df)

    # Required columns
    required_columns = [
        "User_ID",
        "Name",
        "Age",
        "Country",
        "Subscription_Type",
        "Watch_Time_Hours",
        "Favorite_Genre",
        "Last_Login"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    # Fill numerical missing values
    df["Age"] = pd.to_numeric(
        df["Age"],
        errors="coerce"
    )

    df["Watch_Time_Hours"] = pd.to_numeric(
        df["Watch_Time_Hours"],
        errors="coerce"
    )

    df["Age"] = df["Age"].fillna(
        df["Age"].median()
    )

    df["Watch_Time_Hours"] = df["Watch_Time_Hours"].fillna(
        df["Watch_Time_Hours"].median()
    )

    # Fill categorical missing values
    categorical_columns = [
        "Country",
        "Subscription_Type",
        "Favorite_Genre"
    ]

    for col in categorical_columns:

        df[col] = df[col].fillna(
            df[col].mode()[0]
        )

    # Encode categorical columns
    subscription_encoder = LabelEncoder()
    genre_encoder = LabelEncoder()

    df["Subscription_Encoded"] = (
        subscription_encoder.fit_transform(
            df["Subscription_Type"].astype(str)
        )
    )

    df["Genre_Encoded"] = (
        genre_encoder.fit_transform(
            df["Favorite_Genre"].astype(str)
        )
    )

    # Features used for clustering
    features = [
        "Age",
        "Watch_Time_Hours",
        "Subscription_Encoded",
        "Genre_Encoded"
    ]

    X = df[features].copy()

    # Scaling
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    return df, X, X_scaled, features