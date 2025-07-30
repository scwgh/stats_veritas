import streamlit as st
import pandas as pd
import scipy.stats as stats

def run():
    st.header("🧮 Chi-Squared Test")

    with st.expander("📘 What is the Chi-Squared Test?", expanded=False):
        st.write("""
            The Chi-Squared test is a statistical method used to assess whether there is a significant difference 
            in the distribution of values between two analyzers for a given analyte and material.

            **When to Use:**
            - You want to see if two analyzers produce significantly different value distributions for an analyte.
        """)

    with st.expander("📘 Instructions"):
        st.markdown("""
            1. Upload your data in a CSV file containing:
                - A `Material` column (e.g., matrix or sample type),
                - An `Analyser` column (e.g., instrument name),
                - One or more analyte result columns.
            2. Select an **Analyte** and a **Material** to compare the distributions between two analyzers.
        """)

    # --- File Upload in Expander ---
    with st.expander("📤 Upload Your CSV File", expanded=True):
        st.markdown("Upload a CSV containing your analyte data. Ensure it includes the following columns: `Material`, `Analyser`, and `Sample ID`.")
        uploaded_file = st.file_uploader("Choose a file to get started", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.subheader("🔍 Raw Data Preview")
            st.dataframe(df)

            required_cols = ['Analyser', 'Material', 'Sample ID']
            if not all(col in df.columns for col in required_cols):
                st.error("❌ Your file must contain 'Analyser', 'Material', and 'Sample ID' columns.")
                return

            analyte_cols = [col for col in df.columns if col not in required_cols]
            selected_analyte = st.selectbox("🔬 Select Analyte", analyte_cols)
            selected_material = st.selectbox("🧫 Select Material", df['Material'].dropna().unique())

            # Filter for selected material
            filtered_df = df[df['Material'] == selected_material]

            analyzers = filtered_df['Analyser'].dropna().unique()
            if len(analyzers) < 2:
                st.warning("❗ Need at least two analyzers for comparison.")
                return

            selected_analyzers = st.multiselect("🧪 Select Two Analyzers to Compare", analyzers, default=analyzers[:2])
            if len(selected_analyzers) != 2:
                st.warning("Please select exactly two analyzers.")
                return

            # Filter and prepare data
            subset = filtered_df[filtered_df['Analyser'].isin(selected_analyzers)][['Analyser', selected_analyte]].dropna()
            subset[selected_analyte] = pd.to_numeric(subset[selected_analyte], errors='coerce')

            # Bin the analyte values into quartiles (or use custom bins if preferred)
            subset['Binned'] = pd.qcut(subset[selected_analyte], q=4, duplicates='drop')

            # Create contingency table
            contingency_table = pd.crosstab(subset['Analyser'], subset['Binned'])

            st.write("🔢 **Contingency Table**")
            st.dataframe(contingency_table)

            # Perform Chi-Squared Test
            chi2, p, dof, expected = stats.chi2_contingency(contingency_table)

            st.success("✅ Chi-Squared Test Completed!")
            st.write(f"**Chi-Squared Statistic:** {chi2:.3f}")
            st.write(f"**Degrees of Freedom:** {dof}")
            st.write(f"**P-Value:** {p:.4f}")

            st.write("📈 **Expected Frequencies Table**")
            expected_df = pd.DataFrame(expected, index=contingency_table.index, columns=contingency_table.columns)
            st.dataframe(expected_df)

        except Exception as e:
            st.error(f"⚠️ An error occurred: {e}")
