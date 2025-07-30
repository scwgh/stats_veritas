import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import kstest, norm
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import kstest, norm
import plotly.graph_objects as go
def plot_ks_test(data: pd.Series):
    
    data_sorted = np.sort(data)
    n = len(data)

    ecdf_y = np.arange(1, n + 1) / n
    mu, sigma = np.mean(data), np.std(data, ddof=1)
    cdf_y = norm.cdf(data_sorted, loc=mu, scale=sigma)

    d_statistic = np.max(np.abs(ecdf_y - cdf_y))
    d_index = np.argmax(np.abs(ecdf_y - cdf_y))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data_sorted, y=ecdf_y, mode='lines+markers',
                             name='Empirical CDF', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=data_sorted, y=cdf_y, mode='lines',
                             name='Normal CDF', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=[data_sorted[d_index], data_sorted[d_index]],
                             y=[ecdf_y[d_index], cdf_y[d_index]],
                             mode='lines', name='D Statistic',
                             line=dict(color='red', dash='dash')))

    fig.update_layout(title="K-S Test: Empirical vs Normal CDF",
                      xaxis_title="Standardized Data",
                      yaxis_title="Cumulative Probability",
                      template="plotly_white",
                      height=400)
    return fig

def run():
    st.header("📈 Kolmogorov–Smirnov (K-S) Test")

    with st.expander("📘 What is the Kolmogorov–Smirnov Test?", expanded=False):
        st.write("""
            The Kolmogorov-Smirnov (K-S) test is based on the empirical distribution function and can be used to decide if a sample comes from a popualtion with a given distribution.
                 \n It is defined as:""")
        st.latex(r'''\epsilon _n(x) = \frac{1}{n} \sum_{i=1}^{n} I(X_i \leq x)''')
        st.write("""
                            
            \n But what is so interesting about that, you say?
            \n The K-S test has two extremely attractive features which are useful in interpreting population datasets. 
                 \n First, the test statistic itself does not depend on the cumulative distribution. This means that the K-S test can be used to compare any two distributions (for example, disease- and non-disease state populations), not just the normal distribution.
            \n And second, is an *exact* test (by contrast, the Chi-squared test depends on an adequate sample size for the approximations to be valid). This means that the K-S test is valid for even small sample sizes.
            \n The K-S test is defined by:
            \n - **Null hypothesis (H₀)**: The data follows the specified distribution.
            \n - **Alternative hypothesis (H₁)**: The data does not follow the specified distribution.
            \n - **p-value < 0.05** indicates a significant difference from the reference distribution.              
            """)
        st.latex(r'''D_n = \max_{1 \leq i \leq n} \left| \epsilon_n(x_i) - F(x_i) \right|''')
        st.markdown("""where *F* is the cumulative distribution function of the reference distribution (e.g., normal) and *D_n* is the K-S statistic.
                    The graph shown here is a plot of the empirical cumulative distribution function (CDF) of the data against the CDF of a normal distribution. The K-S test statistic is the maximum vertical distance between these two curves.""")
        
        # Example Plot
        np.random.seed(42)
        example_data = np.random.normal(loc=0, scale=1, size=100)
        fig = plot_ks_test(example_data)
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("📘 Instructions"):
        st.markdown("""
            1. Upload a CSV file where:
               - `Material` is in column 4 (index 3).
               - Analyte columns start from column 6 (index 5) onward.
            2. Then, select an analyte and a material to test.
        """)

    with st.expander("📤 Upload Your CSV File", expanded=True):
        uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            material_col = df.columns[3]
            analyte_cols = df.columns[5:]
            df[material_col] = df[material_col].astype(str)

            st.subheader("📖 Data Preview")
            st.dataframe(df.head())

            selected_analyte = st.selectbox("🔬 Select an Analyte", analyte_cols)
            selected_material = st.selectbox("🧪 Select a Material", df[material_col].unique())

            if st.button("Run K-S Test"):
                group_df = df[df[material_col] == selected_material]
                data = group_df[selected_analyte].dropna()

                st.markdown(f"### 🔬 Analyte: **{selected_analyte}**")
                st.write(f"**Material:** {selected_material}")

                if len(data) < 5:
                    st.warning("Not enough data (n < 5).")
                else:
                    standardized = (data - data.mean()) / data.std(ddof=1)
                    stat, p_value = kstest(standardized, 'norm')

                    st.write(f"- K-S Statistic: `{stat:.4f}`")
                    st.write(f"- p-value: `{p_value:.4f}`")

                    if p_value < 0.05:
                        st.error("❌ Data significantly differs from a normal distribution.")
                    else:
                        st.success("✅ Data does not significantly differ from a normal distribution.")

                    fig = plot_ks_test(standardized)
                    st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"⚠️ Error loading data: {e}")

# --- Optional Reference Section ---
    with st.expander("📚 References"):
        st.markdown("""
        **Chakravarti, I.M., Laha, R.G., and Roy, J. (1967)**, *Handbook of Methods Applied Statistics, Volume I*, John Wiley and Sons, Hoboken. pp 392-394.
        """)