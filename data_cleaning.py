import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from scipy.stats import boxcox

def clean_data(df):
    st.write("## 🛠 Data Cleaning & Preprocessing")

    df = df.copy()  # Prevent modifying the original dataset

    # Sidebar for better UI
    st.sidebar.header("⚙️ Data Cleaning Options")

     # 📊 Display Basic Data Info
    st.write("### 📌 Dataset Overview")
    st.write(f"**Total Rows:** {df.shape[0]}  |  **Total Columns:** {df.shape[1]}")
    
    # Show column data types
    st.write("**Data Types:**")
    st.write(df.dtypes.to_frame().rename(columns={0: "Type"}))

    # Show missing values count
    missing_values = df.isnull().sum()[df.isnull().sum() > 0].to_frame().rename(columns={0: "Missing Values"})
    if not missing_values.empty:
        st.write("**🔍 Missing Values Count:**")
        st.write(missing_values)
    
    # Show unique values in categorical columns
    cat_cols = df.select_dtypes(include=["object"]).columns
    if len(cat_cols) > 0:
        st.write("**🔠 Unique Values in Categorical Columns:**")
        unique_counts = df[cat_cols].nunique().to_frame().rename(columns={0: "Unique Values"})
        st.write(unique_counts)

    ### 🗑 Remove Duplicates ###
    if st.sidebar.checkbox("🗑 Remove Duplicates"):
        before = df.shape[0]
        df = df.drop_duplicates()
        after = df.shape[0]
        st.sidebar.success(f"✅ Removed {before - after} duplicate rows.")

    ### 🔍 Missing Value Handling ###
    if st.sidebar.checkbox("🧹 Handle Missing Values"):
        missing_cols = df.columns[df.isnull().any()]
        if missing_cols.empty:
            st.sidebar.success("✅ No missing values found!")
        else:
            st.sidebar.write("📌 Columns with missing values:", list(missing_cols))
            for col in missing_cols:
                method = st.sidebar.selectbox(f"⚡ Handling method for `{col}`:", ["Mean", "Median", "Mode", "Remove Rows"], key=col)
                if method == "Mean":
                    df[col].fillna(df[col].mean(), inplace=True)
                elif method == "Median":
                    df[col].fillna(df[col].median(), inplace=True)
                elif method == "Mode":
                    df[col].fillna(df[col].mode()[0], inplace=True)
                elif method == "Remove Rows":
                    df.dropna(subset=[col], inplace=True)
            st.sidebar.success("✅ Missing values handled!")

    ### 🚀 Outlier Detection & Handling ###
    if st.sidebar.checkbox("📊 Detect & Handle Outliers"):
        num_cols = df.select_dtypes(include=["number"]).columns
        outlier_method = st.sidebar.selectbox("🔍 Choose Outlier Handling Method", ["Z-score", "IQR", "None"])
        
        if outlier_method != "None":
            for col in num_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                if outlier_method == "IQR":
                    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
                elif outlier_method == "Z-score":
                    df = df[(np.abs((df[col] - df[col].mean()) / df[col].std()) < 3)]
            st.sidebar.success("✅ Outliers handled!")

    ### 🔡 Data Type Conversion ###
    if st.sidebar.checkbox("🔄 Convert Data Types"):
        for col in df.columns:
            new_type = st.sidebar.selectbox(f"Convert `{col}` to:", ["No Change", "Integer", "Float", "String"], key=col+"_dtype")
            if new_type == "Integer":
                df[col] = pd.to_numeric(df[col], errors='coerce').astype("Int64")
            elif new_type == "Float":
                df[col] = pd.to_numeric(df[col], errors='coerce').astype("float")
            elif new_type == "String":
                df[col] = df[col].astype(str)
        st.sidebar.success("✅ Data types converted!")

    ### 📝 Text Cleaning ###
    if st.sidebar.checkbox("📝 Clean Text Data"):
        text_cols = df.select_dtypes(include=["object"]).columns
        if not text_cols.empty:
            for col in text_cols:
                if st.sidebar.checkbox(f"🔠 Apply text cleaning on `{col}`"):
                    df[col] = df[col].str.lower().str.replace(r"[^a-zA-Z0-9 ]", "", regex=True)
                    st.sidebar.success(f"✅ Cleaned text in `{col}`")
        else:
            st.sidebar.warning("⚠️ No text columns found!")

    ### 📏 Feature Scaling ###
    if st.sidebar.checkbox("📊 Apply Feature Scaling"):
        scaler_type = st.sidebar.radio("Choose Scaling Method:", ["Standardization", "Normalization"])
        num_cols = df.select_dtypes(include=["number"]).columns

        if scaler_type == "Standardization":
            scaler = StandardScaler()
        else:
            scaler = MinMaxScaler()

        df[num_cols] = scaler.fit_transform(df[num_cols])
        st.sidebar.success(f"✅ Applied {scaler_type} to numerical columns!")

    ### 🔠 Encode Categorical Variables ###
    if st.sidebar.checkbox("🔠 Encode Categorical Variables"):
        encoding_type = st.sidebar.radio("Choose Encoding Type:", ["Label Encoding", "One-Hot Encoding"])
        cat_cols = df.select_dtypes(include=["object"]).columns
        
        if encoding_type == "Label Encoding":
            encoder = LabelEncoder()
            for col in cat_cols:
                df[col] = encoder.fit_transform(df[col])
        else:
            df = pd.get_dummies(df, columns=cat_cols)
        
        st.sidebar.success(f"✅ Applied {encoding_type} to categorical columns!")



    st.write("✅ **Final Processed Dataset Preview**:")
    st.write(df.head())

    return df
