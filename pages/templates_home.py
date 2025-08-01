import numpy as np
import os
import pandas as pd
import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.font_manager as fm
from scipy import stats
from scipy.stats import linregress
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from scipy.stats import levene, ttest_ind
from scipy.odr import ODR, Model, RealData
import streamlit as st 
from datetime import datetime
import time     
import warnings
import plotly.graph_objects as go
from utils import apply_app_styling, show_footer

st.set_page_config(
    page_title="Validation Templates",
    page_icon="📂",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_app_styling() 

st.header("📂 Templates")

with st.expander("📂 Which template do I need?", expanded=True):
    st.markdown("""                 
    - For each module, you can upload your data in CSV format. Ensure your data is structured correctly according to the provided templates. 4 templates are provided:
        - Imprecision and Method Comparison data ( QC, EQA and Patient data)
        - Linearity and Calibration data
        - Limits (Blank, LOD and LOQ) data
        - Reference Intervals data
    \n For each module, your data must be in CSV format. Ensure your data is structured according to the provided templates. 
    \n For Imprecision, Method Comparison and Limits analysis, it is important that column names (i.e., Date, Material, Analyser, Sample ID) are consistent with the templates to ensure proper analysis. 
    \n You may insert as many analyte names as required - however, avoid introducing spaces or special characters **(e.g., @ : ; ! , . # < >)** in the column names.
    """)
    with st.expander("🎯 Imprecision and Method Comparison Template", expanded=True):
        st.markdown("This template includes may include data for QC, EQA and Patient samples. This file can then be used to assess imprecision and for method comparison.")

        with open("templates/vnv_data_template.csv", "rb") as file:
            st.download_button(
                label="⬇ Download Imprecision & Method Comparison Template",
                data=file,
                file_name="vnv_data_template.csv",
                mime="text/csv"
            )
    with st.expander("📈 Linearity and Calibration Template", expanded=True):
        st.markdown("Use this template to assess linearity and generate calibration curves for analytes across concentration ranges.")
        with open("templates/vnv_linearity_template.csv", "rb") as file:
            st.download_button(
                label="⬇ Download Linearity Template",
                data=file,
                file_name="vnv_linearity_template.csv",
                mime="text/csv"
            )

    with st.expander("🧪 Limits Template", expanded=True):
        st.markdown("Use this template to estimate the LOQ and LOD from replicate measurements of blank and samples near the detection threshold.")
        with open("templates/vnv_limits_template.csv", "rb") as file:
            st.download_button(
                label="⬇ Download Limits Template",
                data=file,
                file_name="vnv_limits_template.csv",
                mime="text/csv"
            )

    with st.expander("🧑‍🤝‍🧑 Reference Intervals Template", expanded=True):
        st.markdown("Use this template to estimate reference intervals from healthy population data.")
        with open("templates/vnv_reference_intervals_template.csv", "rb") as file:
            st.download_button(
                label="⬇ Download Reference Intervals Template",
                data=file,
                file_name="vnv_reference_intervals_template.csv",
                mime="text/csv"
            )
        
