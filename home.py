import streamlit as st
import sqlite3
import pandas as pd
from datetime import datetime
# from signup_utils import init_db, save_signup, get_signups
import streamlit.components.v1 as components
from utils import apply_app_styling, show_footer

st.set_page_config(
    page_title="Validation and Statistical Analysis App",
    page_icon=":computer:",
    layout="wide",
    initial_sidebar_state="expanded",
)

with open("styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

apply_app_styling()

# Styling block
st.markdown("""
<style>
    body, p {
        font-size: 14px;
        line-height: 1.6;
        color: #2C3E50;
    }
    h2 {
        font-weight: 700;
        font-size: 28px;
        color: #34495E;
        margin-bottom: 0.5rem;
    }
    a {
        color: #4A90E2;
        text-decoration: none;
        transition: color 0.3s ease;
    }
    a:hover {
        color: #1c6dd0;
        text-decoration: underline;
    }
    .sidebar .css-1d391kg {
        padding-top: 1rem;
    }
    .card {
        background:#f7f9fc; 
        padding:20px; 
        border-radius:10px; 
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    footer {
        border-top: 1px solid #ddd;
        padding-top: 12px;
        font-size: 12px;
        color: #666;
        text-align: center;
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)

st.balloons()

st.markdown(
    """
    <h2 style='text-align: center; color: #2C3E50;'>Welcome to the Validation and Statistical Analysis App <span class='wave'>👋</span></h2>
    """, 
    unsafe_allow_html=True
)

with st.spinner("Loading the analysis tools... Please wait!"):
    st.success("Analysis tools loaded successfully! Let's get started with your validation analysis! 🎉")

with st.container():
    st.markdown("""
    Welcome to the **Stats Veritas: Validation and Statistics App** — your comprehensive tool designed specifically for laboratory professionals seeking to streamline and enhance the process of analytical method validation and verification.

    This app offers a **user-friendly interface** to perform a broad range of essential statistical analyses critical to ensuring data quality and compliance with regulatory standards. 

    ### About This App

    - **Simplifies Complex Statistics:** No advanced programming knowledge needed—intuitive controls guide you through each analysis step.
    - **Saves Time:** Automates calculations and generates clear visualizations and reports instantly.
    - **Improves Data Confidence:** Helps you identify measurement variability, biases, and method performance issues early.
    - **Supports Compliance:** Facilitates adherence to regulatory requirements and quality assurance protocols.
    - **Customizable & Extensible:** Modules can be expanded, and additional tests are continuously integrated based on user feedback.

    ---
    """)

    ### Upcoming Features (In Development)
    with st.expander("🛠️ Upcoming Features (In Development)", expanded=False):
        st.markdown("""
        We are actively working to include more advanced statistical tests to further empower your data analysis, including:

        - Anderson-Darling Test (normality testing)
        - Bartlett’s Test (homogeneity of variances)
        - Cochran’s Test (variance analysis)
        - Kolmogorov-Smirnov Test (distribution comparisons)
        - Kruskal-Wallis Test (non-parametric group comparisons)
        - Levene’s Test (variance equality)
        - "Choose a test" interface for easy navigation

        **Please note:** Modules flagged as **“in development”** are not yet fully validated. **Do not use these modules for critical decision-making until you see a clear confirmation message indicating they are production-ready and they have been removed from this list.**
        """)

    ### Feedback & Support
    with st.expander("💬 Feedback & Support", expanded=False):
        st.markdown("""
        We encourage users to provide feedback and suggestions to help us improve and tailor this app to your laboratory needs. Your input directly influences new features and updates.

        If you want to hear about new features, updates, or releases, please sign up for our email list at the bottom of this page. If you have any questions, suggestions, or need assistance, please use the feedback form also at the bottom of this page.

        We are committed to continuously enhancing this app to meet the evolving needs of laboratory professionals.

        Thank you for choosing our Validation and Statistics App! We look forward to supporting your laboratory's analytical excellence.
        """)

    ### Data Upload Section
    with st.expander("📂 Data Upload and Templates", expanded=False):
        st.markdown("""
        For each module, your data must be in CSV format. Ensure your data follows the provided templates available from the Templates page.

        Column names and order (**Date, Material, Analyser, Sample ID, Batch ID, Lot Number**) must be consistent to ensure proper analysis.

        You may insert as many analyte names as needed—avoid spaces or special characters (**e.g., @ : ; ! , . # < >**) in column names or data.

        The app automatically detects columns and provides options for selection.

        If you encounter issues, please consult documentation or contact support.
        
        Please note, the app does not store any uploaded data. All analyses are performed in-memory, and no data is saved to the server or database. This ensures your data privacy and security.
        """)

    ### Available Modules
    with st.expander("📖 Available Modules", expanded=False):
        st.markdown("""
        Use the sidebar to navigate the comprehensive platform offering validation and verification analyses for laboratory settings.

        ### Modules include:
        - **Imprecision Analysis**
          - Intra-Well Imprecision
          - Intra-Batch Precision
          - Inter-Batch Precision
          - Inter-Analyser Precision
        - **Linearity**
          - Linear Regression
          - Polynomial Regression
          - Logistic Regression
          - Ridge and Lasso Regression
          - Recovery
        - **Limits**
          - Limit of Blank (LOB)
          - Limit of Detection (LOD)
          - Limit of Quantitation (LOQ)
        - **Method Comparison**
          - One-Way ANOVA
          - Two-Way Crossed ANOVA
          - Two-Way Nested ANOVA
          - Bland-Altman Analysis
          - Deming Regression
          - Passing-Bablok Analysis
        - **More Statistical Tests**
          - Anderson-Darling Test — *in development*
          - Bartlett's Test — *in development*
          - Chi-Squared Test
          - Cochran's Test
          - CUSUM Test
          - F-test
          - Kolmogorov-Smirnov Test — *in development*
          - Kruskal-Wallis Test — *in development*
          - Levene's Test — *in development*
          - Mann-Whitney U Test
          - P-P Plots
          - Q-Q Plots
          - Shapiro-Wilk Test 
          - T-test
          - Total Allowable Error (TEa)
          - Z-test
        """)

# Mailing List
with st.expander("📩 Join the Mailing List"):
    components.html(
        """
        <div style="width:100%; height:1400px; overflow:hidden;">
            <iframe 
                width="100%" 
                height="1400px" 
                src="https://forms.office.com/Pages/ResponsePage.aspx?id=pAt3Bl-NYUKZepKf633PeHHvy4O17RdBju0TuYj6WfpUMURGVzRMRk1HWVcwUjRaVUhJTlVLNk9RVy4u&embed=true" 
                frameborder="0" 
                marginwidth="0" 
                marginheight="0" 
                style="border: none; max-width: 100%; overflow: hidden;" 
            </iframe>
        </div>
        """,
        height=1420,
    )

# Feedback
with st.expander(":clipboard: Submit Feedback or Request Features"):
    components.html(
        """
        <div style="width:100%; height:1400px; overflow:hidden;">
            <iframe 
                width="100%" 
                height="1400px" 
                src="https://forms.office.com/Pages/ResponsePage.aspx?id=pAt3Bl-NYUKZepKf633PeHHvy4O17RdBju0TuYj6WfpUNUpZT1YxOVA4OEZGU0pVNjI1QThKWDI4Ni4u&embed=true" 
                frameborder="0" 
                marginwidth="0" 
                marginheight="0" 
                style="border: none; max-width: 100%; overflow: hidden;" 
            </iframe>
        </div>
        """,
        height=1420,
    )

# Footer
show_footer()
