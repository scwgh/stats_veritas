import streamlit as st
import pandas as pd
import numpy as np

def cochran_test(groups):
    """
    Perform Cochran's C test for heterogeneity of variances across k groups.
    Returns the C statistic and individual variances.
    """
    k = len(groups)
    n = [len(g) for g in groups]
    
    if len(set(n)) != 1:
        raise ValueError("All groups must have the same number of replicates.")
    
    variances = [np.var(g, ddof=1) for g in groups]
    C = max(variances) / sum(variances)
    
    return C, variances

def run():
    st.header("🧮 Cochran’s C Test for Variance Homogeneity")

    with st.expander("📘 What is Cochran’s Test?", expanded=False):
        st.write("""
            Cochran’s C test is used to detect outliers in variances when comparing multiple groups.  
            It tests whether the **largest variance** is significantly different from the others.

            **Assumptions:**
            - Each group has equal sample sizes.
            - Data is approximately normally distributed.
        """)

    with st.expander("📘 Instructions"):
        st.markdown("""
            1. Upload a CSV file where:
               - `Material` is in column 4 (index 3)
               - Analyte values start from column 6 (index 5) onward.
            2. Each group (column) should have the same number of replicates (rows).
            3. Cochran's Test will be run separately for each Material.
        """)

    # --- File Upload in its own Expander ---
    with st.expander("📤 Upload Your CSV File", expanded=True):
        st.markdown("Upload a CSV containing your analyte data. Ensure it includes the following columns: `Material`, `Analyser`, and `Sample ID`.")
        uploaded_file = st.file_uploader("Choose a file to get started", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            material_col = df.columns[3]
            analyte_data = df.iloc[:, 5:]
            analyte_data = analyte_data.apply(pd.to_numeric, errors='coerce')
            df[material_col] = df[material_col].astype(str)

            st.subheader("📊 Raw Data Preview")
            st.dataframe(df.head())

            if st.button("Run Cochran's Test"):
                for material, group_df in df.groupby(material_col):
                    st.markdown(f"### 🧪 Material: **{material}**")

                    analyte_subset = group_df.iloc[:, 5:]
                    analyte_subset = analyte_subset.apply(pd.to_numeric, errors='coerce')
                    analyte_subset_clean = analyte_subset.dropna()

                    if analyte_subset_clean.empty or analyte_subset_clean.shape[0] < 2:
                        st.warning(f"Not enough valid data for Material: {material}")
                        continue

                    try:
                        groups = [analyte_subset_clean[col].values for col in analyte_subset_clean.columns]
                        C_stat, variances = cochran_test(groups)

                        st.write(f"**C Statistic:** {C_stat:.4f}")
                        st.write("**Group Variances:**")
                        for col, var in zip(analyte_subset_clean.columns, variances):
                            st.write(f"- {col}: {var:.4f}")
                    except ValueError as ve:
                        st.error(f"{material}: {ve}")

        except Exception as e:
            st.error(f"⚠️ Error loading data: {e}")
