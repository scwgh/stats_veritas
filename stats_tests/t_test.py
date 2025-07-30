import streamlit as st
import pandas as pd
from scipy.stats import ttest_ind, ttest_rel

def run():
    st.header("🧮 T-Test Analysis Between Analyzers (by Material)")

    with st.expander("📘 What is a T-Test?", expanded=False):
        st.markdown("""
        A **T-Test** compares the means of two groups and determines whether they are statistically different.

        **Use Case Here:** Compare analyte measurements between two analyzers for a selected material.

        **Assumptions:**
        - Data is normally distributed.
        - Equal or similar variances (for independent t-test).
        """)

    with st.expander("📘 Instructions"):
        st.markdown("""
        1. Upload a CSV where each row contains a `Material`, an `Analyser`, and analyte values from column 6 onward.
        2. Select the material, analyte, and two analyzers you want to compare.
        3. Choose T-Test type and run the test.
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
            st.error("CSV must include 'Analyser' and 'Material' columns.")
            return

        analyte_options = df.columns[5:]
        if len(analyte_options) == 0:
            st.warning("No analyte columns found from column index 5 onward.")
            return

        material_options = df["Material"].dropna().unique()
        material = st.selectbox("Select Material:", material_options)

        filtered_df = df[df["Material"] == material]
        if filtered_df.empty:
            st.warning("No data available for the selected material.")
            return

        analyte = st.selectbox("Select Analyte:", analyte_options)
        available_analysers = filtered_df["Analyser"].dropna().unique()
        if len(available_analysers) < 2:
            st.warning("Need at least two analysers for comparison.")
            return

        analyser1 = st.selectbox("Select First Analyser:", available_analysers)
        analyser2 = st.selectbox("Select Second Analyser:", [a for a in available_analysers if a != analyser1])

        test_type = st.radio("Choose T-Test Type:", ["Independent", "Paired"], horizontal=True)

        if st.button("Run T-Test"):
            data1 = filtered_df[filtered_df["Analyser"] == analyser1][analyte].dropna()
            data2 = filtered_df[filtered_df["Analyser"] == analyser2][analyte].dropna()

            if len(data1) < 2 or len(data2) < 2:
                st.warning("Both analyzers must have at least 2 values.")
                return

            try:
                if test_type == "Independent":
                    t_stat, p_val = ttest_ind(data1, data2, equal_var=True)
                else:
                    min_len = min(len(data1), len(data2))
                    t_stat, p_val = ttest_rel(data1.iloc[:min_len], data2.iloc[:min_len])

                st.success("✅ T-Test Completed")
                st.write(f"**Material:** {material}")
                st.write(f"**Analyte:** {analyte}")
                st.write(f"**T-Statistic:** {t_stat:.4f}")
                st.write(f"**P-Value:** {p_val:.4f}")

                if p_val < 0.05:
                    st.warning("🔍 Statistically significant (p < 0.05) — the means are likely different.")
                else:
                    st.info("✅ Not statistically significant (p ≥ 0.05) — no strong evidence of difference.")
            except Exception as e:
                st.error(f"Error performing T-test: {e}")
