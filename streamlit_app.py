import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Hospital Appointment Analysis", layout="wide")

st.title("🏥 Hospital Appointment Data Preprocessing & Analysis")

# --- Upload Dataset ---
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Raw Data Preview")
    st.dataframe(df.head())

    # --- Data Inspection ---
    st.subheader("🔍 Data Inspection")
    st.write("**Shape:**", df.shape)
    st.write("**Column Info:**")
    buffer = df.info(buf=None)
    st.text(buffer)
    st.write("**Summary Statistics:**")
    st.write(df.describe(include="all"))

    # --- Handling Missing Values ---
    st.subheader("🧩 Missing Values Handling")
    st.write(df.isnull().sum())

    # Option to fill or drop missing values
    action = st.radio("Choose how to handle missing values:", ["None", "Drop Rows", "Fill with Mean/Mode"], index=0)
    if action == "Drop Rows":
        df = df.dropna()
        st.success("Dropped rows with missing values.")
    elif action == "Fill with Mean/Mode":
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col].fillna(df[col].mode()[0], inplace=True)
            else:
                df[col].fillna(df[col].mean(), inplace=True)
        st.success("Filled missing values with mean/mode.")

    # --- Handling Duplicates ---
    st.subheader("🧹 Duplicate Handling")
    duplicate_count = df.duplicated().sum()
    st.write(f"Total Duplicates: {duplicate_count}")

    if st.button("Remove Duplicates"):
        df = df.drop_duplicates()
        st.success("Duplicates removed successfully.")

    # --- Data Type Conversion ---
    st.subheader("🔄 Data Type Conversion")
    st.write("Convert date columns (if any):")
    date_cols = st.multiselect("Select columns to convert to datetime", df.columns)
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    st.success("Date conversion completed.")

    # --- Visualization ---
    st.subheader("📈 Data Visualization")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if numeric_cols:
        selected_col = st.selectbox("Select column to visualize distribution", numeric_cols)
        fig, ax = plt.subplots()
        ax.hist(df[selected_col].dropna(), bins=30)
        ax.set_title(f"Distribution of {selected_col}")
        st.pyplot(fig)

    # --- Save Cleaned Data ---
    st.subheader("💾 Export Cleaned Data")
    if st.button("Save Cleaned Data"):
        df.to_csv("cleaned_data.csv", index=False)
        st.success("✅ Cleaned data saved as 'cleaned_data.csv'")

else:
    st.info("Please upload your hospital appointment CSV file to begin.")
