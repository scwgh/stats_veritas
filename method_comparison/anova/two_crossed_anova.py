import streamlit as st
import pandas as pd
import plotly.express as px
from io import BytesIO
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from utils import apply_app_styling
apply_app_styling()

st.header("😵‍💫 Two-Way Crossed ANOVA")

st.set_page_config(
    page_title="Two-Way Crossed ANOVA",
    page_icon="😵",
    layout="wide",
    initial_sidebar_state="expanded",
)


with st.expander("📘 What is Two-Way Crossed ANOVA?", expanded=True):
    st.markdown("""
    ### 🧪 Overview

    **Two-Way Crossed ANOVA** is used when you want to analyze the influence of **two independent categorical factors** on a continuous outcome, with **replicate measurements** at each combination of factor levels.

    This design is typical in laboratory settings, such as when comparing **QC results** across different **analyzers** and **QC materials**.

    It allows you to test:

    - Whether results differ across levels of **Factor A** (e.g., Material)
    - Whether results differ across levels of **Factor B** (e.g., Analyzer)
    - Whether there is an **interaction** between the two factors
    """)

    st.markdown("---")
    with st.expander("### 📐 Statistical Model"):
        st.markdown("""
        Assume:
        - \( I \) levels of **Factor A** (e.g., QC Material)
        - \( J \) levels of **Factor B** (e.g., Analyzer)
        - \( K \) replicate observations at each combination

        The two-way crossed ANOVA model with interaction is:
        """)
        
        st.latex(r'''
        y_{ijk} = \mu + \alpha_i + \beta_j + (\alpha\beta)_{ij} + \epsilon_{ijk}
        ''')

        st.markdown("Where:")
        st.latex(r"y_{ijk}: \text{ Observation from the } k\text{-th replicate at level } i \text{ of Factor A and level } j \text{ of Factor B}")
        st.latex(r"\mu: \text{ Overall mean}")
        st.latex(r"\alpha_i: \text{ Effect of the } i\text{-th level of Factor A (e.g., Material)}")
        st.latex(r"\beta_j: \text{ Effect of the } j\text{-th level of Factor B (e.g., Analyzer)}")
        st.latex(r"(\alpha\beta)_{ij}: \text{ Interaction effect between levels } i \text{ and } j")
        st.latex(r"\epsilon_{ijk}: \text{ Random error term, assumed } \sim \mathcal{N}(0, \sigma^2)")
    
    with st.expander("### 🔍 Example Factors in a Laboratory Context"):
        st.markdown("""
        - **Material** — QC Level (e.g., low/high)
        - **Analyser** — Instrument used (e.g., Roche, Abbott)
        - **Analyte** — Measured compound (e.g., glucose, sodium)
        - **Lot Number** — Optional blocking or nested factor

        These can be arranged in crossed or nested designs depending on your study goals.
        """)

    with st.expander("### 📊 Hypothesis Testing"):
        st.markdown("ANOVA evaluates each main effect and interaction using F-tests:")
        st.markdown("""
        - **Main Effect A (e.g., Material):**
            - \( H_0: \alpha_1 = \alpha_2 = \dots = \alpha_I = 0 \)  
            - No difference between levels of Material

        - **Main Effect B (e.g., Analyzer):**
            - \( H_0: \beta_1 = \beta_2 = \dots = \beta_J = 0 \)  
            - No difference between levels of Analyzer

        - **Interaction Effect (A × B):**
            - \( H_0: (\alpha\beta)_{ij} = 0 \quad \forall\ i,j \)  
            - No interaction between Material and Analyzer

        - **Alternative Hypothesis:**  
            At least one factor or interaction has a significant effect
        """)

    with st.expander("### ✅ When to Use Two-Way Crossed ANOVA"):
        st.markdown("""
        - Comparing measurement systems across multiple instruments and materials
        - Assessing consistency across methods or laboratories
        - Investigating whether QC behavior is influenced by material and platform
        - Identifying systematic or interaction-driven sources of bias
        """)

    with st.expander("### 📌 Assumptions and Considerations"):
        st.markdown("""
        - Assumes **normally distributed residuals** and **homogeneity of variances**
        - Best suited for **balanced designs** (equal replicates per group)
        - Always interpret **interaction effects first** — significant interactions can change the interpretation of main effects
    """)


with st.expander("📘 Instructions"):
    st.markdown(""" 
    1. **Prepare Your Data (CSV Format)**
        Your dataset should be in **wide format**, with at least the following columns:

        - `Material` — QC level or sample group (e.g., QC1, QC2)
        - `Analyser` — Instrument or platform ID (e.g., A1, A2)
        - `Sample ID` — Unique identifier for each measurement
        - **One or more numeric analyte columns** — (e.g., Glucose, Sodium)
        - *(Optional)* `LotNo` — For tracking lot-level variability

        **For example:**

        | Material | Analyser | Sample ID | Glucose | Sodium | LotNo |
        |----------|----------|-----------|---------|--------|--------|
        | QC1      | A1       | S001      | 4.9     | 138    | L01    |
        | QC1      | A2       | S002      | 5.0     | 137    | L01    |
        | QC2      | A1       | S003      | 10.5    | 145    | L02    |
        | ...      | ...      | ...       | ...     | ...    | ...    |

    2. **Upload Your File**
        - Go to the **"📤 Upload Your CSV File"** section
        - Upload your dataset using the file uploader
        - The app automatically:
            - Filters to rows where `Material` starts with "QC"
            - Reshapes the data into **long format** for analysis
            - Identifies numeric analyte columns for ANOVA

    3. **Review the Data**
        - View both the original and long-form data
        - Confirm the correct analytes and factors are present

    4. **Run the Analysis**
        - The app fits a two-way crossed ANOVA model described above:
        - This will generate:
            - **ANOVA summary table**
            - **Significance results (p-values)**
            - **Violin plots** for visualizing data by Material and other factors

    5. **Download Results**
        - You can download the ANOVA table as a CSV file for reporting or further analysis

    ---
    ⚠️ **Tips**
    - Ensure that `Material`, `Analyser`, and `Sample ID` columns are spelled exactly as shown
    - Only numeric analyte columns will be included in the analysis - do not include text or categorical data, and avoid use of special characters described on the 'Templates' page
    - You must have at least **two levels of Material** for ANOVA to work
    """)

with st.expander("📤 Upload Your CSV File", expanded=True):
    uploaded_file = st.file_uploader("   ", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📋 Raw Data Preview")
    st.dataframe(df.head())

    # Keep only QC data
    df_qc = df[df['Material'].astype(str).str.startswith("QC")].copy()

    if df_qc.empty:
        st.warning("No QC data found. Ensure 'Material' column contains values like 'QC1', 'QC2', etc.")

    required_columns = {'Material', 'Analyser', 'Sample ID'}
    if not required_columns.issubset(df_qc.columns):
        st.error("Missing one or more required columns: 'Material', 'Analyser', 'Sample ID'")

    # Identify analyte columns
    exclude_cols = {'Material', 'Analyser', 'Sample ID', 'LotNo'}
    analyte_cols = [col for col in df_qc.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df_qc[col])]

    if not analyte_cols:
        st.error("No numeric analyte columns found.")

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
