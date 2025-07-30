import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import kruskal

def run():
    st.header("📊 Kruskal-Wallis Test for Independent Samples")

    with st.expander("📘 What is the Kruskal-Wallis Test?", expanded=False):
        st.write("""
            The **Kruskal-Wallis H-test** is a **non-parametric** method for testing whether samples originate from the same distribution.
            
            It is used as an alternative to one-way ANOVA when the assumptions of ANOVA (e.g. normality, equal variance) are not met.
            
            - Suitable for **ordinal** or **non-normally distributed** data.
            - Tests for differences **across three or more independent groups**.
        """)

    with st.expander("📘 Instructions"):
        st.markdown("""
            1. Upload a CSV file where:
               - `Material` is in column 4 (index 3)
               - Analyte values start from column 6 (index 5) onward.
            2. The Kruskal-Wallis test will be run **separately for each analyte**, comparing values **across QC levels**.
        """)

    with st.expander("📤 Upload Your CSV File", expanded=True):
        uploaded_file = st.file_uploader("Upload your CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            material_col = df.columns[3]
            analyte_cols = df.columns[5:]
            df[material_col] = df[material_col].astype(str)

            st.subheader("📋 Raw Data Preview")
            st.dataframe(df.head())

            if st.button("Run Kruskal-Wallis Test"):
                results = []

                for material in df[material_col].unique():
                    material_df = df[df[material_col] == material]

                    if material_df.empty:
                        continue

                    for analyte in analyte_cols:
                        subset = material_df[[material_col, analyte]].dropna()

                        if subset[material_col].nunique() < 2:
                            results.append({
                                "Material": material,
                                "Analyte": analyte,
                                "H Statistic": np.nan,
                                "p-value": np.nan,
                                "Result": "⚠️ Not enough groups"
                            })
                            continue

                        try:
                            groups = [group[analyte].values for _, group in subset.groupby(material_col)]
                            stat, p_value = kruskal(*groups)

                            result_text = "❌ Significant difference" if p_value < 0.05 else "✅ No significant difference"

                            results.append({
                                "Material": material,
                                "Analyte": analyte,
                                "H Statistic": round(stat, 4),
                                "p-value": round(p_value, 4),
                                "Result": result_text
                            })

                        except Exception as e:
                            results.append({
                                "Material": material,
                                "Analyte": analyte,
                                "H Statistic": np.nan,
                                "p-value": np.nan,
                                "Result": f"❌ Error: {e}"
                            })

                results_df = pd.DataFrame(results)
                st.subheader("📈 Kruskal-Wallis Test Results")
                st.dataframe(results_df)

        except Exception as e:
            st.error(f"⚠️ Error loading data: {e}")