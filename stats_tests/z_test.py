import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm

def run():
    st.header("🧮 Z-Test Between Analyzers (by Material and Analyte)")

    with st.expander("📘 What is a Z-Test?", expanded=False):
        st.markdown("""
        The **Z-Test** determines whether two population means are different when variances are known or sample sizes are large.

        **Assumptions:**
        - Data follows a normal distribution.
        - Variances are known (or sample size is large: n > 30).
        - Samples are independent.
        """)

    with st.expander("📘 Instructions"):
        st.markdown("""
        1. Upload a CSV with columns: `Analyser`, `Material`, and analyte results (from column 6 onward).
        2. Select the material, analyte, and two analyzers to compare.
        3. Estimate or input standard deviations and run the test.
        """)

        # --- File Upload ---
    with st.expander("📤 Upload Your CSV File", expanded=True):
        st.markdown("Upload a CSV containing your analyte data. Ensure it includes the following columns: `Material`, `Analyser`, and `Sample ID`.")
        uploaded_file = st.file_uploader("Choose a file to get started", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.subheader("📊 Data Preview")
        st.dataframe(df)

        required_cols = {"Analyser", "Material"}
        if not required_cols.issubset(df.columns):
            st.error("CSV must contain 'Analyser' and 'Material' columns.")
            return

        analyte_options = df.columns[5:]
        if len(analyte_options) == 0:
            st.warning("No analyte columns found (expected from column index 6 onward).")
            return

        material_options = df["Material"].dropna().unique()
        material = st.selectbox("Select Material:", material_options)

        filtered_df = df[df["Material"] == material]
        if filtered_df.empty:
            st.warning("No data for selected material.")
            return

        analyte = st.selectbox("Select Analyte:", analyte_options)
        analyser_options = filtered_df["Analyser"].dropna().unique()

        if len(analyser_options) < 2:
            st.warning("At least two analysers are required.")
            return

        analyser1 = st.selectbox("Select First Analyser:", analyser_options)
        analyser2 = st.selectbox("Select Second Analyser:", [a for a in analyser_options if a != analyser1])

        data1 = filtered_df[filtered_df["Analyser"] == analyser1][analyte].dropna()
        data2 = filtered_df[filtered_df["Analyser"] == analyser2][analyte].dropna()

        use_sample_std = st.checkbox("Estimate standard deviations from sample", value=True)

        if not use_sample_std:
            std1 = st.number_input(f"Enter known SD for {analyser1}", min_value=0.0001, value=1.0)
            std2 = st.number_input(f"Enter known SD for {analyser2}", min_value=0.0001, value=1.0)
        else:
            std1 = np.std(data1, ddof=1)
            std2 = np.std(data2, ddof=1)

        if st.button("Run Z-Test"):
            try:
                n1, n2 = len(data1), len(data2)
                if n1 < 2 or n2 < 2:
                    st.warning("Each group must have at least 2 values.")
                    return

                mean1 = np.mean(data1)
                mean2 = np.mean(data2)
                se = np.sqrt(std1**2 / n1 + std2**2 / n2)
                z = (mean1 - mean2) / se
                p = 2 * (1 - norm.cdf(abs(z)))

                st.success("✅ Z-Test Completed")
                st.write(f"**Material:** {material}")
                st.write(f"**Analyte:** {analyte}")
                st.write(f"**Analyser 1 ({analyser1}) Mean:** {mean1:.4f} | SD: {std1:.4f} | N: {n1}")
                st.write(f"**Analyser 2 ({analyser2}) Mean:** {mean2:.4f} | SD: {std2:.4f} | N: {n2}")
                st.write(f"**Z-Statistic:** {z:.4f}")
                st.write(f"**P-Value:** {p:.4f}")

                if p < 0.05:
                    st.warning("🔍 Statistically significant difference (p < 0.05).")
                else:
                    st.info("✅ No significant difference (p ≥ 0.05).")

            except Exception as e:
                st.error(f"Error performing Z-test: {e}")
