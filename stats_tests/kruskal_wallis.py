import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import kruskal
from utils import apply_app_styling, units_list, show_footer

apply_app_styling()

st.header("📊 Kruskal-Wallis Test for Independent Samples")

with st.expander("📘 What is the Kruskal-Wallis Test?", expanded=True):
    st.write("""
        The **Kruskal-Wallis H-test** is a **non-parametric** method for testing whether samples originate from the same distribution.
        
        It is used as an alternative to one-way ANOVA when the assumptions of ANOVA (e.g. normality, equal variance) are not met.
        
        - Suitable for **ordinal** or **non-normally distributed** data.
        - Tests for differences **across three or more independent groups**.
    """)

with st.expander("📘 Instructions"):
    st.markdown("""
        1. Upload a CSV file where:
            - `Material` is in Column 4.
            - Analyte values start from Column 6 onward.
        2. The Kruskal-Wallis test will be run **separately for each analyte**, comparing values **across QC levels**.
    """)

# File uploader
with st.expander("📤 Upload Your CSV File", expanded=True):
    uploaded_file = st.file_uploader("  ", type="csv")

# Run analysis if file is uploaded
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        material_col = df.columns[3]
        analyte_cols = df.columns[5:]

        df[material_col] = df[material_col].astype(str)

        st.subheader("🔍 Data Preview")
        st.dataframe(df.head())

        if st.button("🚀 Run Kruskal-Wallis Test"):
            results = []

            for material in df[material_col].unique():
                subset_df = df[df[material_col] == material]

                for analyte in analyte_cols:
                    data = subset_df[[material_col, analyte]].dropna()

                    # Skip if fewer than 2 groups
                    if data[material_col].nunique() < 2:
                        results.append({
                            "Material": material,
                            "Analyte": analyte,
                            "H Statistic": np.nan,
                            "p-value": np.nan,
                            "Result": "⚠️ Not enough groups"
                        })
                        continue

                    try:
                        groups = [group[analyte].values for _, group in data.groupby(material_col)]
                        h_stat, p_val = kruskal(*groups)

                        results.append({
                            "Material": material,
                            "Analyte": analyte,
                            "H Statistic": round(h_stat, 4),
                            "p-value": round(p_val, 4),
                            "Result": "✅ No significant difference" if p_val >= 0.05 else "❌ Significant difference"
                        })

                    except Exception as e:
                        results.append({
                            "Material": material,
                            "Analyte": analyte,
                            "H Statistic": np.nan,
                            "p-value": np.nan,
                            "Result": f"❌ Error: {str(e)}"
                        })

            # Display results
            results_df = pd.DataFrame(results)
            st.subheader("📈 Kruskal-Wallis Test Results")
            st.dataframe(results_df)

    except Exception as e:
        st.error(f"❗ Error loading file: {e}")
        
show_footer()