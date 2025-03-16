import streamlit as st
import pandas as pd
from data_loader import load_data
from data_cleaning import clean_data
from eda import perform_eda
from ml_model import train_model

st.title("📊 Automated Data Analysis & ML App")

# Upload Dataset
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file:
    df = load_data(uploaded_file)
    st.write("### Preview of Dataset", df.head())

    # Data Cleaning
    cleaned_df = clean_data(df)
    st.write("### Cleaned Data", cleaned_df.head())

    # Exploratory Data Analysis
    perform_eda(cleaned_df)

    # Machine Learning
    train_model(cleaned_df)
