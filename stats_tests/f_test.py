import streamlit as st
import pandas as pd
from scipy.stats import f
import numpy as np

def run():
    st.header("🧮 F-Test for Comparing Variances")

    with st.expander("📘 What is an F-Test?", expanded=False):
        st.markdown("""
        The **F-Test** is used to compare the variances of two populations and determine whether they are significantly different.

        **Assumptions:**
        - The populations are normally distributed.
        - The samples are independent.
        """)

    with st.expander("📘 Instructions"):
        st.markdown("""
        1. Upload a CSV file with at least two numeric columns.
        2. Select the columns you want to compare.
        3. The F-statistic is the ratio of the larger variance to the smaller.
        4. A small p-value (typically < 0.05) indicates a significant difference in variances.
        """)

    # --- File Upload ---
    with st.expander("📤 Upload Your CSV File", expanded=True):
        st.markdown("Upload a CSV containing your analyte data. Ensure it includes the following columns: `Material`, `Analyser`, and `Sample ID`.")
        uploaded_file = st.file_uploader("Choose a file to get started", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.subheader("📊 Data Preview")
        st.dataframe(df)

        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        if len(numeric_cols) < 2:
            st.warning("Please upload a dataset with at least two numeric columns.")
            return

        col1 = st.selectbox("Select first group (column):", numeric_cols, key="f_col1")
        col2 = st.selectbox("Select second group (column):", [c for c in numeric_cols if c != col1], key="f_col2")

        group1 = df[col1].dropna()
        group2 = df[col2].dropna()

        if st.button("Run F-Test"):
            try:
                var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
                dfn, dfd = len(group1) - 1, len(group2) - 1

                # Ensure F = larger variance / smaller variance
                if var1 > var2:
                    F = var1 / var2
                    dfn, dfd = dfn, dfd
                else:
                    F = var2 / var1
                    dfn, dfd = dfd, dfn

                p = 2 * (1 - f.cdf(F, dfn, dfd))  # Two-tailed

                st.success("✅ F-Test Completed")
                st.write(f"**F-Statistic:** {F:.4f}")
                st.write(f"**P-Value (two-tailed):** {p:.4f}")
                st.write(f"**Degrees of Freedom:** {dfn}, {dfd}")

                if p < 0.05:
                    st.warning("🔍 Significant difference in variances (p < 0.05).")
                else:
                    st.info("✅ No significant difference in variances (p ≥ 0.05).")
            except Exception as e:
                st.error(f"Error running F-test: {e}")
