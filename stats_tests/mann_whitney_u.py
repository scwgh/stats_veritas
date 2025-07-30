# mann_whitney_u.py

import streamlit as st
import pandas as pd
from scipy.stats import mannwhitneyu

def run():
    st.title("📊 Mann-Whitney U Test")

    st.markdown("""
    The **Mann-Whitney U Test** is a non-parametric test used to compare whether there is a difference in the distribution of two independent groups.
    It does not assume normality and is an alternative to the independent samples t-test.

    ### 📂 Data Upload Requirements:
    - Upload a **CSV file** with your experimental data.
    - The file must contain:
        - A **numeric column** representing the values to compare.
        - A **group column** that identifies the two groups (e.g., "Group A" vs "Group B").

    The test will be performed on these two groups to determine if their distributions differ.
    """)

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        st.subheader("Data Preview:")
        st.write(df.head())

        group_column = st.selectbox("Select the column containing group labels", df.columns)
        value_column = st.selectbox("Select the numeric column to compare", df.columns)

        if pd.api.types.is_numeric_dtype(df[value_column]):
            groups = df[group_column].dropna().unique()
            if len(groups) != 2:
                st.error("The selected group column must contain exactly two unique groups.")
                return

            # Subset the data
            group1 = df[df[group_column] == groups[0]][value_column].dropna()
            group2 = df[df[group_column] == groups[1]][value_column].dropna()

            # Perform Mann-Whitney U Test
            stat, p_value = mannwhitneyu(group1, group2, alternative='two-sided')

            st.subheader("Mann-Whitney U Test Results")
            st.write(f"Comparing: `{groups[0]}` vs `{groups[1]}`")
            st.write(f"U Statistic: {stat}")
            st.write(f"P-Value: {p_value}")

            if p_value < 0.05:
                st.error("There is a statistically significant difference between the two groups (reject H0).")
            else:
                st.success("There is no statistically significant difference between the two groups (fail to reject H0).")
        else:
            st.error("Please select a valid numeric column for the comparison.")
    else:
        st.info("Please upload a CSV file to begin.")
