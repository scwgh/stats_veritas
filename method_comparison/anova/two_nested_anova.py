import streamlit as st
import pandas as pd
import plotly.express as px
from io import BytesIO
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from utils import apply_app_styling

def run():
    st.title("😵‍💫 Two-Way Nested ANOVA")

    with st.expander("📘 What is Two-Way Nested ANOVA?"):
        st.markdown(""" 
        \n When it is not possible to cross every level of one factor with every level of another factor, you should consider using **Two-Way Nested ANOVA**. In nested designs, levels of one factor (e.g., `Analyser`) occur only within one level of another factor (e.g., `Material`).
        \n A nested layout describes when fewer than all levels of one factor occur within each level of the other factor (for example, if we want to study the effects of different analysers measuring urine creatinine run across different sites; but we can't change the operators at each site).
        \n The model dictates: if Factor B is nested within Factor A, then a level of Factor B can only occur within one level of Factor A and there can be no interaction. This gives the following model: 
        """)
        st.latex(r'''{y}_{ijk} = \mu + \alpha_i + \beta_{j(i)} + \epsilon_{ijk}''')
        st.markdown(""" It is important within this type of layout, since each level of one factor is only present with one level of the other factor, we can't estimate interaction between the two.
                    \n Using this format, we are testing that the twmo main effects are zero. Similar to Two-WAy Crossed ANOVA, we form a  (F0) of each main effect mean square to the appropriate mean-squared error term. (Note that the error term for Factor A is not MSE, but is MSB.) If the assumptions stated below are true then those ratios follow an F distribution and the test is performed by comparing the F0 ratios to values in an F table with the appropriate degrees of freedom and confidence level.
            - **Which factors are required for Two-Way Nested ANOVA**: 
              - Material (e.g., QC including Level), 
              - Analyser (nested within Material), 
              - Analyte, 
              - Optional: Lot Number or Batch Number
            - **Null Hypothesis**: No effect from any factor
            - **Alternative**: At least one factor has an effect
        """)

    with st.expander("📘 Instructions"):
        st.markdown(""" 
        1. Upload a CSV file with:
            - `Material` (e.g., QC1, QC2)
            - `Analyser`
            - `Sample ID`
            - One or more numeric analyte columns
            - Optional: `LotNo`
        2. The app reshapes the data and performs nested ANOVA using all factors.
        """)

    with st.expander("📤 Upload Your CSV File", expanded=True):
        uploaded_file = st.file_uploader("Upload your CSV", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.subheader("📋 Raw Data Preview")
        st.dataframe(df.head())

        # Keep only QC data
        df_qc = df[df['Material'].astype(str).str.startswith("QC")].copy()

        if df_qc.empty:
            st.warning("No QC data found. Ensure 'Material' column contains values like 'QC1', 'QC2', etc.")
            return

        required_columns = {'Material', 'Analyser', 'Sample ID'}
        if not required_columns.issubset(df_qc.columns):
            st.error("Missing one or more required columns: 'Material', 'Analyser', 'Sample ID'")
            return

        # Identify analyte columns
        exclude_cols = {'Material', 'Analyser', 'Sample ID', 'LotNo'}
        analyte_cols = [col for col in df_qc.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df_qc[col])]

        if not analyte_cols:
            st.error("No numeric analyte columns found.")
            return

        # Melt to long format
        id_vars = ['Material', 'Analyser']
        if 'LotNo' in df_qc.columns:
            id_vars.append('LotNo')
        df_long = df_qc.melt(id_vars=id_vars, value_vars=analyte_cols,
                             var_name='Analyte', value_name='Value').dropna()

        # st.subheader("📊 Two-Way Nested ANOVA")
        # st.dataframe(df_long.head())

        # Check enough levels for ANOVA
        if df_long['Material'].nunique() < 2:
            st.warning("Not enough QC levels for ANOVA.")
            return

        # Construct nested ANOVA formula
        # Analyser is nested within Material; optionally LotNo nested too
        nesting = 'C(Analyser):C(Material)'
        formula_parts = ['C(Material)', nesting, 'C(Analyte)']
        if 'LotNo' in df_qc.columns:
            formula_parts.append('C(LotNo):C(Material)')
        formula = "Value ~ " + " + ".join(formula_parts)

        try:
            model = ols(formula, data=df_long).fit()
            anova_table = anova_lm(model, typ=2)

            st.subheader("📈 ANOVA Summary Table")
            st.dataframe(anova_table.round(4))

            # Interpretation
            p_values = anova_table['PR(>F)']
            for factor in anova_table.index:
                p = p_values[factor]
                st.markdown(f"**{factor}** — p-value: `{p:.4f}` → {'✅ Significant' if p < 0.05 else '❌ Not Significant'}")

            # Dropdown to select analyte
            selected_analyte = st.selectbox(
                "Select Analyte for Violin Plot",
                options=df_long['Analyte'].unique()
            )

            # Filter data for selected analyte
            df_filtered = df_long[df_long['Analyte'] == selected_analyte]

            # Violin plot
            st.subheader(f"🎻 Violin Plot for {selected_analyte}")
            color_by = 'LotNo' if 'LotNo' in df_qc.columns else 'Analyser'
            fig = px.violin(
                df_filtered,
                x="Material",
                y="Value",
                color=color_by,
                box=True,
                points="all",
                facet_col="Analyte",
                category_orders={"Material": sorted(df_filtered["Material"].unique())},
                title=f"Distribution by QC Level for {selected_analyte}"
            )
            st.plotly_chart(fig, use_container_width=True)

            # Download ANOVA table
            csv_buffer = BytesIO()
            anova_table.to_csv(csv_buffer)
            st.download_button(
                "⬇ Download ANOVA Table",
                data=csv_buffer.getvalue(),
                file_name="nested_anova_results.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"Error during ANOVA: {e}")