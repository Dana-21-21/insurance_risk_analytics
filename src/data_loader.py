# src/data_loader.py

import pandas as pd


def load_data(filepath):
    """
    Load the insurance dataset.

    Parameters
    ----------
    filepath : str or Path
        Path to the dataset.

    Returns
    -------
    pd.DataFrame
        Loaded insurance dataset.
    """
    #return pd.read_csv(filepath, sep="|")
    return pd.read_csv(filepath, sep="|", low_memory=False)


def preprocess_data(df):
    """
    Basic preprocessing and feature engineering.
    """

    df = df.copy()


    # Convert date column
    df["TransactionMonth"] = pd.to_datetime(
        df["TransactionMonth"]
    )


    # Create Transaction Year

    df["TransactionYear"] = (
        df["TransactionMonth"]
        .dt.year
    )


    # Create Vehicle Age

    if "RegistrationYear" in df.columns:

        df["VehicleAge"] = (
            df["TransactionYear"]
            -
            df["RegistrationYear"]
        )


        df["VehicleAge"] = (
            df["VehicleAge"]
            .clip(lower=0)
        )


    return df