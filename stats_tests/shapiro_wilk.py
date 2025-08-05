import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import shapiro
import matplotlib.pyplot as plt
import seaborn as sns
from utils import apply_app_styling, units_list, show_footer

apply_app_styling()

st.set_page_config(
    page_title="Shapiro-Wilk Normality Test",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.header("📊 Shapiro-Wilk Normality Test (by Material and Analyte)")

with st.expander("📘 What is the Shapiro-Wilk Test?", expanded=True):
    st.markdown("""
    The **Shapiro-Wilk Test** determines whether a sample comes from a normally distributed population.

    **Key Points:**
    - **Null Hypothesis (H₀):** Data is normally distributed
    - **Alternative Hypothesis (H₁):** Data is not normally distributed
    - **Best for:** Small to moderate sample sizes (n ≤ 5000)
    - **Interpretation:** p < 0.05 suggests data is NOT normally distributed

    **Why Test for Normality?**
    - Required assumption for many statistical tests (t-tests, Z-tests, ANOVA)
    - Helps choose appropriate statistical methods
    - Validates modeling assumptions
    """)

with st.expander("📘 Instructions"):
    st.markdown("""
    1. Upload a CSV with columns: `Analyser`, `Material`, and analyte results (from Column #6 onward).
    2. Select the material, analyte, and analyzer to test for normality.
    3. View the test results and optional histogram/Q-Q plot for visual assessment.
    4. Use results to validate assumptions for other statistical tests.
    """)

# --- File Upload ---
with st.expander("📤 Upload Your CSV File", expanded=True):
    uploaded_file = st.file_uploader("   ", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Data Preview")
    st.dataframe(df)

    required_cols = {"Analyser", "Material"}
    if not required_cols.issubset(df.columns):
        st.error("CSV must contain 'Analyser' and 'Material' columns.")
        st.stop()

    analyte_options = df.columns[5:]
    if len(analyte_options) == 0:
        st.warning("No analyte columns found (expected from column index 6 onward).")
        st.stop()

    # --- Selection Controls ---
    col1, col2 = st.columns(2)
    
    with col1:
        material_options = df["Material"].dropna().unique()
        material = st.selectbox("Select Material:", material_options)

    with col2:
        analyte = st.selectbox("Select Analyte:", analyte_options)

    filtered_df = df[df["Material"] == material]
    if filtered_df.empty:
        st.warning("No data for selected material.")
        st.stop()

    analyser_options = filtered_df["Analyser"].dropna().unique()
    analyser = st.selectbox("Select Analyser:", analyser_options)

    # --- Get Data ---
    data = filtered_df[filtered_df["Analyser"] == analyser][analyte].dropna()

    if len(data) < 3:
        st.warning("At least 3 data points are required for the Shapiro-Wilk test.")
        st.stop()

    if len(data) > 5000:
        st.warning("⚠️ Sample size > 5000. Shapiro-Wilk test may not be reliable for very large samples.")

    # --- Display Data Summary ---
    st.subheader("📈 Data Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Sample Size", len(data))
    with col2:
        st.metric("Mean", f"{np.mean(data):.4f}")
    with col3:
        st.metric("Std Dev", f"{np.std(data, ddof=1):.4f}")
    with col4:
        st.metric("Range", f"{np.max(data) - np.min(data):.4f}")

    # --- Options ---
    show_plots = st.checkbox("Show visual normality assessment plots", value=True)
    
    if st.button("Run Shapiro-Wilk Test"):
        try:
            # Perform Shapiro-Wilk test
            stat, p_value = shapiro(data)
            
            st.success("✅ Shapiro-Wilk Test Completed")
            
            # --- Results Display ---
            st.subheader("📊 Test Results")
            
            result_col1, result_col2 = st.columns(2)
            with result_col1:
                st.write(f"**Material:** {material}")
                st.write(f"**Analyte:** {analyte}")
                st.write(f"**Analyser:** {analyser}")
                st.write(f"**Sample Size:** {len(data)}")
            
            with result_col2:
                st.write(f"**W-Statistic:** {stat:.6f}")
                st.write(f"**P-Value:** {p_value:.6f}")
            
            # --- Interpretation ---
            st.subheader("🔍 Interpretation")
            
            alpha = 0.05
            if p_value < alpha:
                st.error(f"🚫 **Reject Null Hypothesis** (p = {p_value:.6f} < {alpha})")
                st.error("📉 **Data is NOT normally distributed**")
                st.info("💡 Consider using non-parametric tests or data transformation.")
            else:
                st.success(f"✅ **Fail to Reject Null Hypothesis** (p = {p_value:.6f} ≥ {alpha})")
                st.success("📈 **Data appears to be normally distributed**")
                st.info("💡 Normality assumption is satisfied for parametric tests.")

            # --- Visual Assessment ---
            if show_plots:
                st.subheader("📊 Visual Normality Assessment")
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                
                # Histogram with normal curve overlay
                ax1.hist(data, bins=min(30, len(data)//3), density=True, alpha=0.7, color='skyblue', edgecolor='black')
                
                # Overlay normal distribution
                x_norm = np.linspace(data.min(), data.max(), 100)
                y_norm = ((1/np.sqrt(2*np.pi*np.var(data))) * 
                         np.exp(-0.5*((x_norm - np.mean(data))**2)/np.var(data)))
                ax1.plot(x_norm, y_norm, 'r-', linewidth=2, label='Normal Distribution')
                ax1.set_xlabel(analyte)
                ax1.set_ylabel('Density')
                ax1.set_title('Histogram vs Normal Distribution')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Q-Q Plot
                from scipy import stats
                stats.probplot(data, dist="norm", plot=ax2)
                ax2.set_title('Q-Q Plot (Normal)')
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                st.info("""
                **Visual Interpretation:**
                - **Histogram:** Should roughly follow the red normal curve
                - **Q-Q Plot:** Points should roughly follow the diagonal line for normal data
                """)

            # --- Additional Statistics ---
            with st.expander("📈 Additional Descriptive Statistics"):
                from scipy.stats import skew, kurtosis
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"**Skewness:** {skew(data):.4f}")
                    if abs(skew(data)) < 0.5:
                        st.write("✅ Low skewness")
                    elif abs(skew(data)) < 1:
                        st.write("⚠️ Moderate skewness")
                    else:
                        st.write("🚫 High skewness")
                
                with col2:
                    kurt = kurtosis(data)
                    st.write(f"**Kurtosis:** {kurt:.4f}")
                    if abs(kurt) < 0.5:
                        st.write("✅ Normal kurtosis")
                    elif abs(kurt) < 1:
                        st.write("⚠️ Moderate kurtosis")
                    else:
                        st.write("🚫 High kurtosis")
                
                with col3:
                    st.write(f"**Median:** {np.median(data):.4f}")
                    st.write(f"**IQR:** {np.percentile(data, 75) - np.percentile(data, 25):.4f}")

        except Exception as e:
            st.error(f"Error performing Shapiro-Wilk test: {e}")
            
    # --- Test All Analyzers Option ---
    if len(analyser_options) > 1:
        st.subheader("🔄 Test All Analyzers")
        if st.button("Run Test for All Analyzers"):
            results = []
            for analyzer in analyser_options:
                analyzer_data = filtered_df[filtered_df["Analyser"] == analyzer][analyte].dropna()
                if len(analyzer_data) >= 3:
                    stat, p_val = shapiro(analyzer_data)
                    results.append({
                        'Analyzer': analyzer,
                        'Sample Size': len(analyzer_data),
                        'W-Statistic': stat,
                        'P-Value': p_val,
                        'Normal?': 'Yes' if p_val >= 0.05 else 'No'
                    })
            
            if results:
                results_df = pd.DataFrame(results)
                st.dataframe(results_df.style.format({
                    'W-Statistic': '{:.6f}',
                    'P-Value': '{:.6f}'
                }))
