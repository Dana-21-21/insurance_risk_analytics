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
    Perform basic preprocessing.

    - Convert TransactionMonth to datetime.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """

    df = df.copy()

    df["TransactionMonth"] = pd.to_datetime(df["TransactionMonth"])

    return df
