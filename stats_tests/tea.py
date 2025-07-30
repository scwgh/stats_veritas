import streamlit as st
import pandas as pd
import numpy as np
import io

def run():
    st.header("🧮 Total Allowable Error (TEa) Analysis")

    with st.expander("📘 What is TEa?", expanded=False):
        st.markdown("""
        **Total Allowable Error (TEa)** is a quality goal in laboratory medicine that defines the maximum allowable error for a test result, combining imprecision and bias.
        
        It is calculated as:
        \[
        \text{TE} = \left| \text{Bias} \right| + z \cdot \text{SD}
        \]
        where _z_ is the z-score for the desired confidence level (commonly 1.96 for 95%).

        **The result passes if:**
        \[
        \text{TE} \leq \text{TEa}
        \]
        """)

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
        st.markdown("Upload a CSV containing your analyte data. Ensure it includes the following columns: `Material`, `Analyser`, and `Sample ID`.")
        uploaded_file = st.file_uploader("Choose a file to get started", type=["csv"])

    if not uploaded_file:
        return  # If no file is uploaded, exit the function

    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Data Preview")
    st.dataframe(df)

    # Ensure that the required columns are present
    if 'Material' not in df.columns or 'Analyser' not in df.columns or 'Sample ID' not in df.columns:
        st.error("❌ The CSV file must contain columns for 'Material', 'Analyser', and 'Sample ID'.")
        return

    # Let the user select the material for analysis
    materials = df['Material'].unique()
    selected_material = st.selectbox("Select Material for TEa Analysis", materials)

    # Filter the data based on the selected material
    filtered_df = df[df['Material'] == selected_material]

    # Get the numeric columns for the analysis
    numeric_cols = filtered_df.select_dtypes(include='number').columns.tolist()

    # Let the user choose which column is measured
    measured_col = st.selectbox("Select column for Measured values", numeric_cols)
    
    # Let the user manually enter the target value
    target_value = st.number_input("Enter the target value (expected or reference value)", value=0.0)

    # Option to use either a fixed TEa or column-based TEa
    tea_option = st.radio("Do you want to use a fixed TEa or column-based TEa?", ["Fixed value", "Column-based"])

    # Handle fixed TEa input
    if tea_option == "Fixed value":
        tea_value = st.number_input("Enter fixed TEa (%)", min_value=0.0, value=20.0)
        filtered_df['TEa'] = tea_value
    else:  # Handle column-based TEa selection
        tea_col = st.selectbox("Select TEa column", numeric_cols)
        filtered_df['TEa'] = filtered_df[tea_col]

    # Z-value input
    z_value = st.number_input("Z-value for confidence interval (default 1.96 for 95% confidence)", value=1.96)

    # Run the analysis when the button is pressed
    if st.button("Run TEa Analysis"):
        try:
            # Calculate Bias, Abs Bias %, SD, TE, and Pass/Fail status
            filtered_df['Bias'] = filtered_df[measured_col] - target_value  # Using the manually entered target value
            filtered_df['Abs Bias %'] = 100 * filtered_df['Bias'].abs() / target_value
            filtered_df['SD'] = filtered_df[measured_col].std()
            filtered_df['TE'] = filtered_df['Abs Bias %'] + z_value * filtered_df['SD']
            filtered_df['Pass'] = filtered_df['TE'] <= filtered_df['TEa']

            st.success("✅ TEa Evaluation Complete")
            st.dataframe(filtered_df[['Material', 'Bias', 'Abs Bias %', 'SD', 'TE', 'TEa', 'Pass']])

            fail_count = (~filtered_df['Pass']).sum()
            st.write(f"🔍 Number of failures: {fail_count} out of {len(filtered_df)}")
            if fail_count == 0:
                st.success("🎉 All results are within Total Allowable Error!")
            else:
                st.warning("⚠️ Some results exceed Total Allowable Error.")

        except Exception as e:
            st.error(f"Error during TEa analysis: {e}")

    # Add a section for performing TEa analysis for all analytes
    with st.expander("📊 Perform TEa for All Analytes"):
        all_materials = df['Material'].unique()
        selected_material_all = st.selectbox("Select Material for All Analytes", all_materials)

        # Filter data for selected material
        filtered_df_all = df[df['Material'] == selected_material_all]

        # Get all analyte columns (assuming analytes are in numeric columns)
        analyte_cols = filtered_df_all.select_dtypes(include='number').columns.tolist()

        # Let the user choose which column is measured for all analytes
        measured_col_all = st.selectbox("Select column for Measured values (for all analytes)", analyte_cols)

        # Let the user manually enter the target value for all analytes
        target_value_all = st.number_input("Enter the target value for all analytes (expected or reference value)", value=0.0)

        # Perform the TEa calculation for all analytes
        if st.button("Run TEa Analysis for All Analytes"):
            try:
                # Create a new dataframe to hold the results for all analytes
                results = []

                for analyte_col in analyte_cols:
                    temp_df = filtered_df_all.copy()

                    # Set TEa for each analyte in the loop
                    if tea_option == "Fixed value":
                        temp_df['TEa'] = tea_value  # Using the fixed TEa value
                    else:
                        temp_df['TEa'] = temp_df[tea_col]  # Using the column-based TEa

                    # Calculate Bias, Abs Bias %, SD, TE, and Pass/Fail status for each analyte
                    temp_df['Bias'] = temp_df[measured_col_all] - target_value_all
                    temp_df['Abs Bias %'] = 100 * temp_df['Bias'].abs() / target_value_all
                    temp_df['SD'] = temp_df[measured_col_all].std()
                    temp_df['TE'] = temp_df['Abs Bias %'] + z_value * temp_df['SD']
                    temp_df['Pass'] = temp_df['TE'] <= temp_df['TEa']
                    temp_df['Analyte'] = analyte_col
                    results.append(temp_df[['Material', 'Analyte', 'Bias', 'Abs Bias %', 'SD', 'TE', 'TEa', 'Pass']])

                # Combine results into one dataframe
                all_results_df = pd.concat(results)

                # Display results for all analytes
                st.success("✅ TEa Analysis for All Analytes Complete")
                st.dataframe(all_results_df)

                # Provide option to download results as CSV
                csv = all_results_df.to_csv(index=False)
                st.download_button(
                    label="Download Results as CSV",
                    data=csv,
                    file_name=f"TEa_Analysis_{selected_material_all}.csv",
                    mime="text/csv"
                )

            except Exception as e:
                st.error(f"Error during TEa analysis for all analytes: {e}")
