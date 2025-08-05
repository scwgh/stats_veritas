# q_q_plots.py

import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import plotly.graph_objects as go
from utils import apply_app_styling, units_list, show_footer

st.set_page_config(
    page_title="Q-Q Plot",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_app_styling()

# Title and description
st.header("📈 Q-Q Plot Generator")
with st.expander("📘 What is a Q-Q Plot?", expanded=True):
    st.markdown("""
    Quantile-Quantile (Q-Q) plots are used to visually assess whether a dataset follows a specified theoretical distribution—typically a normal distribution.  
    This plot compares the quantiles of your data against the quantiles of a standard normal distribution.  
    """)

with st.expander("📤 Upload Your CSV File", expanded=True):
    uploaded_file = st.file_uploader("   ", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📋 Data Preview")
    st.dataframe(df.head())

    # Numeric columns only
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

    if numeric_columns:
        selected_column = st.selectbox("📊 Select a numeric column for Q-Q plot", numeric_columns)

        # Drop missing values
        data = df[selected_column].dropna()

        # Calculate theoretical quantiles and ordered values
        (osm, osr), (slope, intercept, r) = stats.probplot(data, dist="norm")

        # Plotly Q-Q plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=osm, y=osr,
            mode='markers',
            name='Sample Data',
            marker=dict(color='#00b4d8')
        ))

        fig.add_trace(go.Scatter(
            x=osm, y=slope * np.array(osm) + intercept,
            mode='lines',
            name='Theoretical Line',
            line=dict(color='red', dash='dash')
        ))

        fig.update_layout(
            title=f"Q-Q Plot for {selected_column}",
            xaxis_title="Theoretical Quantiles",
            yaxis_title="Sample Quantiles",
            template="simple_white",
            height=600
        )

        st.plotly_chart(fig, use_container_width=True)

        # Interpretation
        st.markdown("### 🧠 Interpretation Guide")
        st.markdown("""
        - If the points follow the red dashed line closely, the data is normally distributed.
        - Deviations suggest departures from normality:
            - Curved shape: skewness.
            - S-shaped or inverted S: heavy/light tails.
        """)
    else:
        st.error("No numeric columns found in the uploaded dataset.")