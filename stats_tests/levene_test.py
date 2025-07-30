import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import levene

def run():
    st.header("🧪 Levene’s Test for Equality of Variances")

    with st.expander("📘 What is Levene’s Test?", expanded=False):
        st.write("""
            Levene’s Test is used to check whether multiple groups have **equal variances**.  
            It's an important assumption for parametric tests like ANOVA.

            - Unlike Bartlett’s test, Levene’s test does **not assume normality**.
            - A **p-value < 0.05** suggests that **variances are significantly different** between groups.
        """)

    with st.expander("📘 Instructions"):
        st.markdown("""
            1. Upload a CSV file where:
               - `Material` is in column 4 (index 3).
               - Analyte columns start from column 6 (index 5) onward.
            2. The test will compare **each analyte’s variance across QC levels**.
        """)

    with st.expander("📄 Upload Your CSV File", expanded=True):
        st.markdown("Upload a CSV containing your analyte data. Ensure it includes the following columns: `Material`, `Analyser`, and `Sample ID`.")
        uploaded_file = st.file_uploader("Choose a file to get started", type=["csv"], label_visibility="collapsed")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.success("✅ File uploaded successfully!")
            st.markdown("### 📋 Data Preview")
            st.dataframe(df.head())
        else:
            df = None
            st.info("Awaiting file upload...")

    if df is not None:
        material_options = df['Material'].unique()
        analyte_options = [col for col in df.columns if col not in ['Date', 'Material', 'Sample ID']]

        selected_material = st.selectbox("Choose Material", material_options)
        selected_analyte = st.selectbox("Choose Analyte", analyte_options)

        filtered_df = df[df['Material'] == selected_material].copy()

        if not filtered_df.empty:
            # Ensure analyte values are numeric
            filtered_df[selected_analyte] = pd.to_numeric(filtered_df[selected_analyte], errors='coerce')
            filtered_df = filtered_df.dropna(subset=[selected_analyte])

            # Group by Sample ID
            grouped = filtered_df.groupby('Sample ID')[selected_analyte].apply(list)

            if len(grouped) < 2:
                st.error("Need at least two groups with numeric values for Levene's test.")
            elif grouped.apply(len).gt(1).all():
                stat, p = levene(*grouped.tolist())
                st.write(f"**Levene’s Test Statistic:** {stat:.3f}")
                st.write(f"**p-value:** {p:.4f}")
                if p < 0.05:
                    st.warning("Variances are significantly different (p < 0.05)")
                else:
                    st.success("No significant difference in variances (p ≥ 0.05)")
            else:
                st.error("Not enough data points per group for Levene's test.")

if __name__ == "__main__":
    run()
