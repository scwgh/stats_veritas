import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from utils import apply_app_styling, units_list, show_footer

apply_app_styling()

st.set_page_config(
    page_title="Total Alowable Error (TEa)",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.header("🧮 Total Allowable Error (TEa) Analysis")

with st.expander("📘 What is TEa?", expanded=True):
    st.markdown("""
    **Total Allowable Error (TEa)** is a quality goal in laboratory medicine that defines the maximum allowable error for a test result, combining imprecision and bias.
    """)

    st.latex(r"TE = |\text{Bias}| + z \cdot SD")

    st.markdown("""
    Where:
    - **Bias** is the difference between the measured and target value
    - **SD** is the standard deviation (imprecision)
    - **z** is the z-score for the desired confidence level (e.g., 1.96 for 95%)

    **A result passes if:**
    """)

    st.latex(r"TE \leq TEa")

with st.expander("📊 Visual Explanation of TEa Components", expanded=True):
    st.markdown("This diagram illustrates how Bias and SD contribute to Total Error (TE), and how TE is compared against the Total Allowable Error (TEa).")
    
    target = 100
    bias = 5
    sd = 2
    z = 1.96
    tea = 10
    measured = target + bias
    te = abs(bias) + z * sd

    fig, ax = plt.subplots(figsize=(10, 2))
    ax.axvline(target, color='green', linestyle='--', label='Target')
    ax.plot(measured, 0, 'ro', label='Measured')
    ax.annotate('', xy=(target, 0.05), xytext=(measured, 0.05),
                arrowprops=dict(arrowstyle='<->', color='orange', lw=2))
    ax.text((target + measured) / 2, 0.08, 'Bias', color='orange', ha='center')
    sd_start = measured - sd
    sd_end = measured + sd
    ax.plot([sd_start, sd_end], [0, 0], color='blue', lw=4, alpha=0.5, label='±1 SD')
    ax.axvline(target + tea, color='purple', linestyle=':', label='TEa Threshold')
    ax.text(target + tea + 0.5, 0.02, 'TEa', color='purple', rotation=90, va='bottom')
    ax.annotate('', xy=(target, -0.05), xytext=(target + te, -0.05),
                arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    ax.text(target + te / 2, -0.08, 'Total Error (TE)', color='red', ha='center')
    ax.set_ylim(-0.15, 0.15)
    ax.set_xlim(target - 15, target + 20)
    ax.axis('off')
    ax.legend(loc='upper left')
    st.pyplot(fig)

with st.expander("📘 Instructions"):
    st.markdown("""
    1. Upload a CSV file with:
        - **Measured** results (e.g., observed values),
        - **Target** results (e.g., expected or reference values),
        - **Material**, **Analyser**, and **Sample ID**.
    2. Choose a material for analysis and provide the corresponding target value and TEa.
    3. The app will calculate Bias, SD, TE, and evaluate performance.
    """)

# --- File Upload ---
with st.expander("📤 Upload Your CSV File", expanded=True):
    uploaded_file = st.file_uploader("   ", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("📊 Data Preview")
        st.dataframe(df)

        required_columns = {'Material', 'Analyser', 'Sample ID'}
        if not required_columns.issubset(df.columns):
            st.error(f"❌ The CSV must contain the following columns: {required_columns}")
        else:
            # Let the user select the material for analysis
            materials = df['Material'].unique()
            selected_material = st.selectbox("Select Material for TEa Analysis", materials)

            # Filter by selected material
            filtered_df = df[df['Material'] == selected_material]

            numeric_cols = filtered_df.select_dtypes(include='number').columns.tolist()
            measured_col = st.selectbox("Select column for Measured values", numeric_cols)

            target_value = st.number_input("Enter the target value (expected/reference)", value=1.0, min_value=0.0001)

            tea_option = st.radio("Use fixed or column-based TEa?", ["Fixed value", "Column-based"])
            if tea_option == "Fixed value":
                tea_value = st.number_input("Enter fixed TEa (%)", min_value=0.0, value=20.0)
                filtered_df['TEa'] = tea_value
            else:
                tea_col = st.selectbox("Select TEa column", numeric_cols)
                filtered_df['TEa'] = filtered_df[tea_col]

            z_value = st.number_input("Z-value for confidence level", value=1.96)

            if st.button("Run TEa Analysis"):
                try:
                    filtered_df['Bias'] = filtered_df[measured_col] - target_value
                    filtered_df['Abs Bias %'] = 100 * filtered_df['Bias'].abs() / target_value
                    sd = filtered_df[measured_col].std()
                    filtered_df['SD'] = sd
                    filtered_df['TE'] = filtered_df['Abs Bias %'] + z_value * sd
                    filtered_df['Pass'] = filtered_df['TE'] <= filtered_df['TEa']

                    st.success("✅ TEa Evaluation Complete")
                    st.dataframe(filtered_df[['Material', measured_col, 'Bias', 'Abs Bias %', 'SD', 'TE', 'TEa', 'Pass']])

                    fail_count = (~filtered_df['Pass']).sum()
                    st.write(f"🔍 Number of failures: {fail_count} of {len(filtered_df)}")
                    if fail_count == 0:
                        st.success("🎉 All results are within Total Allowable Error!")
                    else:
                        st.warning("⚠️ Some results exceed Total Allowable Error.")

                except Exception as e:
                    st.error(f"Error during TEa analysis: {e}")

            # --- All Analytes TEa Analysis ---
            with st.expander("📊 Perform TEa Analysis for All Analytes"):
                selected_material_all = st.selectbox("Select Material (All Analytes)", materials)
                df_all = df[df['Material'] == selected_material_all]
                analyte_cols = df_all.select_dtypes(include='number').columns.tolist()
                target_value_all = st.number_input("Enter target value for all analytes", value=1.0, min_value=0.0001)

                if st.button("Run TEa Analysis for All Analytes"):
                    try:
                        results = []
                        for analyte_col in analyte_cols:
                            temp_df = df_all.copy()
                            temp_df['Bias'] = temp_df[analyte_col] - target_value_all
                            temp_df['Abs Bias %'] = 100 * temp_df['Bias'].abs() / target_value_all
                            sd = temp_df[analyte_col].std()
                            temp_df['SD'] = sd
                            if tea_option == "Fixed value":
                                temp_df['TEa'] = tea_value
                            else:
                                if tea_col in df_all.columns:
                                    temp_df['TEa'] = df_all[tea_col]
                                else:
                                    continue  # Skip if TEa column not found
                            temp_df['TE'] = temp_df['Abs Bias %'] + z_value * sd
                            temp_df['Pass'] = temp_df['TE'] <= temp_df['TEa']
                            temp_df['Analyte'] = analyte_col
                            results.append(temp_df[['Material', 'Analyte', analyte_col, 'Bias', 'Abs Bias %', 'SD', 'TE', 'TEa', 'Pass']])

                        if results:
                            all_results_df = pd.concat(results)
                            st.success("✅ TEa Analysis for All Analytes Complete")
                            st.dataframe(all_results_df)

                            csv = all_results_df.to_csv(index=False)
                            st.download_button(
                                label="Download Results as CSV",
                                data=csv,
                                file_name=f"TEa_Analysis_{selected_material_all}.csv",
                                mime="text/csv"
                            )
                        else:
                            st.warning("⚠️ No analyte results generated. Check numeric columns and TEa settings.")

                    except Exception as e:
                        st.error(f"Error during TEa analysis for all analytes: {e}")

    except Exception as e:
        st.error(f"⚠️ Could not read uploaded file: {e}")
