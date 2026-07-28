# src/modeling.py

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder

from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_squared_error,
    r2_score
)



# =====================================================
# Prepare Dataset
# =====================================================

def prepare_data(
    df,
    target="TotalClaims"
):

    data=df.copy()


    # keep only claims
    data=data[
        data[target]>0
    ]


    # Vehicle Age

    data["TransactionMonth"] = pd.to_datetime(
        data["TransactionMonth"]
    )


    data["VehicleAge"] = (
        data["TransactionMonth"].dt.year
        -
        data["RegistrationYear"]
    )


    data["VehicleAge"] = data["VehicleAge"].clip(
        lower=0
    )


    drop=[
        "UnderwrittenCoverID",
        "PolicyID",
        "TransactionMonth"
    ]


    data=data.drop(
        columns=[
            c for c in drop
            if c in data.columns
        ]
    )


    X=data.drop(
        columns=[target]
    )

    y=data[target]


    return X,y



# =====================================================
# Feature Selection
# =====================================================

def select_features(X):


    features=[

        "VehicleAge",

        "VehicleType",
        "make",
        "Model",
        "bodytype",

        "kilowatts",
        "cubiccapacity",

        "Province",
        "Gender",
        "MaritalStatus",

        "CoverType",
        "CoverCategory",

        "TermFrequency",

        "SumInsured",
        "CalculatedPremiumPerTerm",
        "CustomValueEstimate",
        "CapitalOutstanding"

    ]


    features=[
        f for f in features
        if f in X.columns
    ]


    return X[features]



# =====================================================
# Split
# =====================================================

def split_data(
    X,
    y
):

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )



# =====================================================
# Preprocessing
# =====================================================

def build_preprocessor(X):


    numerical=list(
        X.select_dtypes(
            include="number"
        ).columns
    )


    categorical=list(
        X.select_dtypes(
            exclude="number"
        ).columns
    )



    num_pipe=Pipeline([

        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        )

    ])



    cat_pipe=Pipeline([

        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),

        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )

    ])



    preprocessor=ColumnTransformer([

        (
            "num",
            num_pipe,
            numerical
        ),

        (
            "cat",
            cat_pipe,
            categorical
        )

    ])


    return preprocessor



# =====================================================
# Train
# =====================================================


def train_model(
    model,
    preprocessor,
    X_train,
    y_train
):


    pipeline=Pipeline([

        (
            "preprocessor",
            preprocessor
        ),

        (
            "model",
            model
        )

    ])



    pipeline.fit(
        X_train,
        y_train
    )


    return pipeline



# =====================================================
# Evaluation
# =====================================================


def evaluate_model(
    model,
    X_test,
    y_test
):

    pred=model.predict(
        X_test
    )


    rmse=mean_squared_error(
        y_test,
        pred,
        squared=False
    )


    r2=r2_score(
        y_test,
        pred
    )


    return rmse,r2



# =====================================================
# Compare
# =====================================================


def compare_models(results):

    return pd.DataFrame(results)



# =====================================================
# Save
# =====================================================


def save_model(
    model,
    path
):

    joblib.dump(
        model,
        path
    )



# =====================================================
# Prediction Input
# =====================================================


def prepare_prediction_input(
    user_input,
    reference_df
):


    required=[

        "VehicleAge",
        "VehicleType",
        "make",
        "Model",
        "bodytype",
        "kilowatts",
        "cubiccapacity",
        "Province",
        "Gender",
        "MaritalStatus",
        "CoverType",
        "CoverCategory",
        "TermFrequency",
        "SumInsured",
        "CalculatedPremiumPerTerm",
        "CustomValueEstimate",
        "CapitalOutstanding"

    ]


    df=pd.DataFrame(
        [user_input]
    )


    for col in required:


        if col not in df.columns:


            if reference_df[col].dtype=="object":

                df[col]=reference_df[col].mode()[0]

            else:

                df[col]=reference_df[col].median()



    return df[required]


