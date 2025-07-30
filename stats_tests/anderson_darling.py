import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import anderson

def run():
    st.header("📏 Anderson-Darling Test for Normality")

    with st.expander("📘 What is the Anderson-Darling Test?", expanded=False):
        st.write("""
        The **Anderson-Darling test** evaluates whether a sample comes from a normal distribution.
        It returns a test statistic and critical values at significance levels: 15%, 10%, 5%, 2.5%, and 1%.
        
        Null Hypothesis (H₀): *The data are normally distributed.*
        """)
        st.latex(r'''
        A^2 = -n - \frac{1}{n} \sum_{i=1}^{n} [ (2i - 1) \cdot (\ln F(x_i) + \ln(1 - F(x_{n+1-i})) ) ]
        ''')
        st.markdown("""
        **Instructions**:
        1. Upload a CSV file where:
           - `Material` is in column 4 (index 3)
           - Analyte values start from column 6 (index 5) onward.
        2. The test is run per material, for each analyte column.
        """)

    with st.expander("📄 Upload Your CSV File", expanded=True):
        st.markdown("Upload a CSV containing your analyte data. Ensure it includes the following columns: `Material`, `Analyser`, and `Sample ID`.")
        uploaded_file = st.file_uploader("Choose a file to get started", type=["csv"], label_visibility="collapsed")
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success("✅ File uploaded successfully!")

                st.markdown("### 📋 Data Preview")
                st.dataframe(df.head())

                # Identify columns
                material_col = df.columns[3]
                analyte_cols = df.columns[5:]
                df[material_col] = df[material_col].astype(str)

                if st.button("Run Anderson-Darling Test"):
                    results = []

                    for material, group_df in df.groupby(material_col):
                        analyte_subset = group_df[analyte_cols].apply(pd.to_numeric, errors='coerce')

                        for analyte in analyte_subset.columns:
                            data = analyte_subset[analyte].dropna()

                            if len(data) < 5:
                                results.append({
                                    'Material': material,
                                    'Analyte': analyte,
                                    'N': len(data),
                                    'A² Statistic': np.nan,
                                    'Critical Value (5%)': np.nan,
                                    'Normality': '⚠️ Not enough data'
                                })
                                continue

                            try:
                                result = anderson(data, dist='norm')
                                stat = result.statistic

                                # Critical value for 5% significance level
                                idx_5 = result.significance_level.tolist().index(5.0)
                                crit_5 = result.critical_values[idx_5]
                                interpretation = "✅ Normal" if stat < crit_5 else "❌ Not Normal"

                                results.append({
                                    'Material': material,
                                    'Analyte': analyte,
                                    'N': len(data),
                                    'A² Statistic': round(stat, 4),
                                    'Critical Value (5%)': round(crit_5, 4),
                                    'Normality': interpretation
                                })

                            except Exception as e:
                                results.append({
                                    'Material': material,
                                    'Analyte': analyte,
                                    'N': len(data),
                                    'A² Statistic': np.nan,
                                    'Critical Value (5%)': np.nan,
                                    'Normality': f'❌ Error: {e}'
                                })

                    results_df = pd.DataFrame(results)
                    st.subheader("📄 Anderson-Darling Test Results")
                    st.dataframe(results_df, use_container_width=True)

            except Exception as e:
                st.error(f"⚠️ Error loading or processing file: {e}")