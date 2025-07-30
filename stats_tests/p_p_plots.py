import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def pp_plot(data, ax, title="P–P Plot"):
    """Generate a P–P plot comparing sample data to a normal distribution."""
    sorted_data = np.sort(data)
    n = len(sorted_data)
    
    # Empirical cumulative probabilities
    emp_cdf = np.arange(1, n + 1) / n

    # Theoretical CDF values from a normal distribution
    mu, sigma = np.mean(sorted_data), np.std(sorted_data, ddof=1)
    theo_cdf = stats.norm.cdf(sorted_data, loc=mu, scale=sigma)

    ax.plot(theo_cdf, emp_cdf, 'o', label='Observed vs Expected')
    ax.plot([0, 1], [0, 1], 'r--', label='Ideal Fit')
    ax.set_title(title)
    ax.set_xlabel("Theoretical CDF")
    ax.set_ylabel("Empirical CDF")
    ax.legend()
    ax.grid(True)

def run():
    st.header("📈 P–P Plot (Probability–Probability Plot)")

    with st.expander("📘 What is a P–P Plot?", expanded=False):
        st.write("""
            A **P–P plot** compares the cumulative distribution of your data against a theoretical distribution (e.g. normal).
            If the data follows the distribution well, points will lie along the 45° diagonal line.

            - Helps check **normality** visually.
            - Complements formal tests like **K-S**, **Shapiro-Wilk**, etc.
        """)

    with st.expander("📘 Instructions"):
        st.markdown("""
            1. Upload a CSV file where:
               - `Material` is in column 4 (index 3).
               - Analyte columns start from column 6 (index 5) onward.
            2. P–P plots will be created per analyte within each `Material`.
        """)

    with st.expander("📤 Upload Your CSV File", expanded=True):
        uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            material_col = df.columns[3]
            analyte_cols = df.columns[5:]
            df[material_col] = df[material_col].astype(str)

            st.subheader("📋 Raw Data Preview")
            st.dataframe(df.head())

            if st.button("Generate P–P Plots"):
                for analyte in analyte_cols:
                    st.markdown(f"### 🔬 Analyte: **{analyte}**")

                    for material, group_df in df.groupby(material_col):
                        data = group_df[analyte].dropna()

                        if len(data) < 5:
                            st.warning(f"Not enough data for {analyte} in Material {material}.")
                            continue

                        fig, ax = plt.subplots()
                        pp_plot(data, ax, title=f"{analyte} - {material}")
                        st.pyplot(fig)

        except Exception as e:
            st.error(f"⚠️ Error loading data: {e}")
