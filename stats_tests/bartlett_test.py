import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import bartlett

def run():
    st.header("📏 Bartlett’s Test for Equal Variances")

    with st.expander("📘 What is Bartlett’s Test?", expanded=False):
        st.write("""
            Bartlett’s test is used to assess if **multiple samples have equal variances**.
            Equal variances across samples is called "_homogeneity of variances_". Some statistical tests, for example, the analysis of variance (ANOVA), assumes that the variances are equal across groups or samples. Therefore, the Bartlett test can be used to verify this assumption. 
            If the test indicates that the variances are not equal, it suggests that the assumption of homogeneity of variances is violated. This can affect the validity of statistical tests that rely on this assumption.
            The test is sensitive to deviations from normality, so it is important to check the normality of the data before applying it. If the data is not normally distributed, consider using Levene's test or Brown-Forsythe test as alternatives.
            It assumes the data is **normally distributed**, and is sensitive to deviations from normality.
            
            **Use case:** Useful for validating assumptions before running ANOVA.
                 
            The Bartlett test is defined as:
        """)
        st.latex(r''' H_0: \sigma_1^2 = \sigma_2^2 = ... = \sigma_k^2 \text{ (homogeneity of variances)}''')
        st.latex(r''' H_a: \text{At least one variance is different}''')
        st.latex(r''' T = \frac{(N-k) \ln(\sigma^2_p)}{\sum_{i=1}^{k} (n_i - 1) \ln(\sigma^2_i)}''')
        st.latex(r''' \sigma^2_p = \frac{\sum_{i=1}^{k} (n_i - 1) \sigma^2_i}{N-k} ''')
        st.markdown(r"""
            The variances are judged to be unequal if:""")
        st.latex(r'''T > \chi^2_{1-\alpha, k-1}''')
        st.markdown("""
            1. Upload a CSV file where:
               - `Material` is in column 4 (index 3)
               - Analyte values start from column 6 (index 5) onward.
            2. Bartlett’s test will be run **separately for each Material**, across analyte columns.
        """)

    with st.expander("📤 Upload Your CSV File", expanded=True):
        uploaded_file = st.file_uploader("Upload your CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            df = df.copy()

            material_col = df.columns[3]
            analyte_cols = df.columns[5:]

            df[material_col] = df[material_col].astype(str)

            st.subheader("📊 Raw Data Preview")
            st.dataframe(df.head())

            selected_material = st.selectbox("🏷️ Choose a material to assess:", sorted(df[material_col].unique()))
            selected_analyte = st.selectbox("🧪 Choose an analyte to assess:", analyte_cols)

            group_col = st.selectbox("🧩 Choose a grouping column (e.g. BatchID, Sample ID, Group):", df.columns[:5])

            if st.button("Run Bartlett's Test"):
                filtered_df = df[df[material_col] == selected_material]

                groups = [group[selected_analyte].dropna().values for _, group in filtered_df.groupby(group_col)]
                groups = [g for g in groups if len(g) > 1]

                if len(groups) < 2:
                    st.warning("⚠️ Need at least two valid groups to perform Bartlett’s test.")
                else:
                    try:
                        stat, p_value = bartlett(*groups)
                        st.markdown(f"### Results for Material: **{selected_material}**, Analyte: **{selected_analyte}**")
                        st.write(f"**Bartlett Test Statistic:** {stat:.4f}")
                        st.write(f"**p-value:** {p_value:.4f}")

                        if p_value < 0.05:
                            st.error("❌ Variances are significantly different — heteroscedasticity detected.")
                        else:
                            st.success("✅ Variances are not significantly different — homoscedasticity assumed.")
                    except Exception as e:
                        st.error(f"❌ Error performing Bartlett’s test: {e}")
        except Exception as e:
            st.error(f"⚠️ Error loading data: {e}")