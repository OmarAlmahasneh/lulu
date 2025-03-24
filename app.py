import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from openai import OpenAI

# Function to load data from uploaded file
def load_data(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('.json'):
            df = pd.read_json(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload CSV, Excel, or JSON.")
            return None
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

# Function to clean data
def clean_data(df):
    if df is not None:
        if st.checkbox("Remove missing values"):
            df = df.dropna()
        else:
            fill_value = st.number_input("Fill missing values with:", value=0)
            df = df.fillna(fill_value)

        if st.checkbox("Remove duplicates"):
            df = df.drop_duplicates()

        return df
    return None

# Function for exploratory data analysis (EDA)
def exploratory_analysis(df):
    if df is not None:
        st.subheader("Statistical Summary")
        st.write(df.describe())

        st.subheader("Variable Distribution")
        num_cols = df.select_dtypes(include=np.number).columns
        for col in num_cols:
            fig, ax = plt.subplots()
            sns.histplot(df[col], ax=ax)
            st.pyplot(fig)

        st.subheader("Variable Relationships")
        if len(num_cols) > 1:
            fig = sns.pairplot(df[num_cols[:5]])
            st.pyplot(fig)

# Function to create an interactive dashboard
def create_dashboard(df):
    if df is not None:
        st.title("Data Analysis Dashboard")

        col1, col2 = st.columns(2)
        with col1:
            selected_col = st.selectbox("Choose a variable", df.columns)
        with col2:
            chart_type = st.selectbox("Choose chart type", ["Line Chart", "Bar Chart", "Pie Chart"])

        fig, ax = plt.subplots()
        if chart_type == "Line Chart":
            sns.lineplot(data=df, x=df.index, y=selected_col, ax=ax)
        elif chart_type == "Bar Chart":
            sns.barplot(data=df, x=df.index, y=selected_col, ax=ax)
        elif chart_type == "Pie Chart":
            df[selected_col].value_counts().plot.pie(ax=ax)
        st.pyplot(fig)

# Function to integrate AI-powered analysis
def ai_analysis(df):
    if df is not None:
        query = st.text_input("Ask AI to analyze your data:")
        if st.button("Get AI Insights"):
            try:
                openai_api_key = st.secrets["openai_api_key"]
                client = OpenAI(api_key=openai_api_key)
                response = client.completions.create(
                    model="gpt-4",
                    prompt=f"""Analyze the following dataset:

{df.head().to_string()}

{query}""",
                    max_tokens=200
                )
                st.write(response.choices[0].text)
            except Exception as e:
                st.error(f"AI analysis failed: {e}")

# Main Streamlit app
def main():
    st.title("Smart Data Analyzer")

    uploaded_file = st.file_uploader("Upload Data File", type=["csv", "xls", "xlsx", "json"])

    if uploaded_file is not None:
        df = load_data(uploaded_file)
        df = clean_data(df)

        if st.checkbox("Exploratory Data Analysis"):
            exploratory_analysis(df)

        create_dashboard(df)
        ai_analysis(df)

if __name__ == "__main__":
    main()
