import streamlit as st
import pandas as pd
import plotly.express as px
from io import BytesIO
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from utils import apply_app_styling

def run():
    st.title("😵‍💫 Two-Way Crossed ANOVA")

    with st.expander("📘 What is Two-Way Crossed ANOVA?"):
        st.markdown(""" 
        When we have two factors with at least two levels and one or more observations at each level, we define this as a two-way layout. If we assume that we have _K_ observations at each combination of _I_ levels of Factor A, and _J_ levels of Factor B, we can mode the two-way layout as:
        """)
        st.latex(r'''y_{ijk} = \mu + \alpha_i + \beta_j + \alpha\beta_{ij} + \epsilon_{ijk}''')
        st.markdown("""
            - **Factors**: 
              - Material (QC Level), 
              - Analyser (crossed with Material), 
              - Analyte, 
              - Optional: LotNo
            - **Null Hypothesis**: No effect from any factor or interaction
            - **Alternative**: At least one factor or interaction has an effect
        """)

    with st.expander("📘 Instructions"):
        st.markdown(""" 
        1. Upload a CSV file with:
            - `Material` (e.g., QC1, QC2)
            - `Analyser`
            - `Sample ID`
            - One or more numeric analyte columns
            - Optional: `LotNo`
        2. The app reshapes the data and performs crossed ANOVA using all factors.
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
        id_vars = ['Material', 'Analyser', 'Sample ID']
        if 'LotNo' in df_qc.columns:
            id_vars.append('LotNo')
        df_long = df_qc.melt(id_vars=id_vars, value_vars=analyte_cols,
                             var_name='Analyte', value_name='Value').dropna()

        st.subheader("📊 Long Format Data")
        st.dataframe(df_long.head())

        # Check enough levels for ANOVA
        if df_long['Material'].nunique() < 2:
            st.warning("Not enough QC levels for ANOVA.")
            return

        # Construct crossed ANOVA formula
        # Material and Analyser are crossed, with an interaction term
        formula_parts = ['C(Material)', 'C(Analyser)', 'C(Material):C(Analyser)', 'C(Analyte)']
        if 'LotNo' in df_qc.columns:
            formula_parts.append('C(LotNo)')
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

            # Violin plot
            st.subheader("🎻 Violin Plot")
            color_by = 'LotNo' if 'LotNo' in df_qc.columns else 'Analyser'
            fig = px.violin(
                df_long,
                x="Material",
                y="Value",
                color=color_by,
                box=True,
                points="all",
                facet_col="Analyte",
                category_orders={"Material": sorted(df_long["Material"].unique())},
                title="Distribution by QC Level and Analyte"
            )
            st.plotly_chart(fig, use_container_width=True)

            # Download ANOVA table
            csv_buffer = BytesIO()
            anova_table.to_csv(csv_buffer)
            st.download_button(
                "⬇ Download ANOVA Table",
                data=csv_buffer.getvalue(),
                file_name="crossed_anova_results.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"Error during ANOVA: {e}")
