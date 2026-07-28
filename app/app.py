# app/app.py

import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import joblib


# =====================================================
# Project Path
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.append(
    str(PROJECT_ROOT)
)



# =====================================================
# Imports
# =====================================================

from src.data_loader import (
    load_data,
    preprocess_data
)

from src.config import DATA_PATH

from src.modeling import (
    prepare_prediction_input
)



# =====================================================
# Page Configuration
# =====================================================

st.set_page_config(
    page_title="Insurance Risk Analytics",
    layout="wide"
)



# =====================================================
# Load Model
# =====================================================

MODEL_PATH = (
    PROJECT_ROOT
    /
    "models"
    /
    "best_model.pkl"
)


@st.cache_resource
def load_model():

    model = joblib.load(
        MODEL_PATH
    )

    return model


model = load_model()



# =====================================================
# Title
# =====================================================

st.title(
    "Insurance Risk Analytics Dashboard"
)


st.write(
    """
    This dashboard predicts insurance claim severity 
    and provides insights into portfolio risk,
    profitability, and pricing decisions.
    """
)



# =====================================================
# Load Dataset
# =====================================================


@st.cache_data
def get_data():

    df = load_data(
        DATA_PATH
    )

    df = preprocess_data(
        df
    )

    return df



df = get_data()



# =====================================================
# Portfolio Overview
# =====================================================


st.header(
    "Portfolio Overview"
)


col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "Total Policies",
        f"{len(df):,}"
    )


with col2:

    st.metric(
        "Total Premium",
        f"${df['TotalPremium'].sum():,.2f}"
    )


with col3:

    st.metric(
        "Total Claims",
        f"${df['TotalClaims'].sum():,.2f}"
    )



# =====================================================
# Profitability
# =====================================================


st.header(
    "Portfolio Profitability"
)


total_premium = (
    df["TotalPremium"]
    .sum()
)


total_claims = (
    df["TotalClaims"]
    .sum()
)


loss_ratio = (
    total_claims /
    total_premium
)


margin = (
    total_premium -
    total_claims
)



col1, col2 = st.columns(2)


with col1:

    st.metric(
        "Overall Loss Ratio",
        f"{loss_ratio:.2%}"
    )


with col2:

    st.metric(
        "Portfolio Margin",
        f"${margin:,.2f}"
    )



# =====================================================
# Risk by Province
# =====================================================


st.header(
    "Risk by Province"
)


province_risk = (
    df.groupby("Province")
    .agg(
        TotalPremium=(
            "TotalPremium",
            "sum"
        ),

        TotalClaims=(
            "TotalClaims",
            "sum"
        )
    )
)



province_risk["LossRatio"] = (
    province_risk["TotalClaims"]
    /
    province_risk["TotalPremium"]
)



province_risk = (
    province_risk
    .sort_values(
        "LossRatio",
        ascending=False
    )
)



st.bar_chart(
    province_risk["LossRatio"]
)



# =====================================================
# Claim Severity Distribution
# =====================================================


st.header(
    "Claim Severity Distribution"
)


claim_distribution = (
    df[
        df["TotalClaims"] > 0
    ]["TotalClaims"]
    .describe()
)


st.dataframe(
    claim_distribution
)



# =====================================================
# Prediction
# =====================================================


st.header(
    "Claim Severity Prediction"
)


st.write(
    """
    Enter policy information to estimate
    expected claim severity.
    """
)



user_input = {}



prediction_features = [

    "Province",
    "VehicleType",
    "Gender",
    "CoverType",
    "VehicleAge",
    "SumInsured"

]



for col in prediction_features:


    if col not in df.columns:
        continue


    if df[col].dtype == "object":


        user_input[col] = st.selectbox(
            col,
            sorted(
                df[col]
                .dropna()
                .unique()
            )
        )


    else:


        user_input[col] = st.number_input(
            col,
            value=float(
                df[col]
                .median()
            )
        )



# Prepare complete model input

input_df = prepare_prediction_input(
    user_input,
    df
)



if st.button(
    "Predict Claim Severity"
):

    prediction = model.predict(
        input_df
    )


    st.success(
        f"Estimated Claim Amount: ${prediction[0]:,.2f}"
    )
    
    
    
    
    








    
    
    