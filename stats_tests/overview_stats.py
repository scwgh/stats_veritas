import streamlit as st
from utils import apply_app_styling

st.set_page_config(
    page_title="Statistical Tests Home",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_app_styling()

st.title("🏠 Statistical Tests Home")

st.markdown(
    """
    Welcome to the **Statistical Tests** section of the Validation and Verification App.

    This module provides access to **common statistical tools** used in method validation and quality assurance. 
    You can assess variance, central tendency, frequency distributions, and error tolerances across various tests.

    ---
    #### 🧭 Available Tests
    - **Chi-Squared Test**: Analyze categorical data using contingency tables.
    - **Cochran's Test**: Compare proportions of binary outcomes.
    - **CUSUM**: Detect subtle shifts or trends in QC results.
    - **F-test**: Compare the variance between two datasets.
    - **T-test**: Assess if means of two datasets differ significantly.
    - **Total Allowable Error (TEa)**: Compare error observed to acceptable limits.
    - **Z-test**: Evaluate a sample mean against a population mean.

    ---
    #### 📂 Data Upload Requirements
    Each test requires a CSV file formatted to match the test's input expectations. Make sure:
    - Headers are clearly labeled.
    - Each row represents a valid observation.
    - Numeric columns are clean (no symbols, text, or missing values).

    📌 **Tip**: Visit a test page using the buttons below to see specific data format examples and instructions.

    ---
    """
)

st.info("⬅⬇️ Use the buttons to get started.")
