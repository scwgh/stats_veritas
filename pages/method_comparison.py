import streamlit as st
import importlib
from utils import apply_app_styling, units_list

# Page setup
st.set_page_config(
    page_title="Method Comparison",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_app_styling()
st.header("📚 Method Comparison")

with st.expander("📘 Why perform Method Comparison?", expanded=True):
    st.markdown("""
    **Method comparison** involves comparing results from different methods or instruments to evaluate their agreement or consistency.
    This is critical in ensuring the reliability and performance of laboratory tests across different platforms, equipment, or methods.
    Here are the types of tests you can explore in this module:

    - **One-Way ANOVA**: Tests whether there are significant differences between the means of three or more independent groups (e.g., QC levels).
    - **Two-Way Crossed ANOVA**: Evaluates the effect of two independent factors (e.g., Material and Analyser) on a response variable, 
      assuming all combinations of these factors are measured. Useful for assessing interactions between factors.
    - **Two-Way Nested ANOVA**: Used when one factor (e.g., Material) is nested within another (e.g., Batch), assessing the variability within 
      and between these nested factors.
    - **Multi-Way ANOVA**: Extends One- and Two-Way ANOVA to analyze the effect of multiple factors (e.g., Material, Analyser, and Batch), 
      including their interactions, providing a more comprehensive analysis.
    - **Bland-Altman Analysis**: A graphical method to assess agreement between two measurement methods by plotting the difference vs. mean. 
      This helps identify any systematic bias or proportional differences.
    - **Deming Regression**: A linear regression method that accounts for measurement error in both variables, ideal for comparing two 
      quantitative methods, especially when neither method can be considered a true reference.
    - **Passing-Bablok Regression**: A non-parametric regression used to compare methods, robust against outliers and assumptions about data 
      distribution. It provides a regression line without assuming normal distribution of errors.
""")
    
with st.expander("📈 What type of regression? Deming vs. Passing-Bablok", expanded=False):
    st.markdown("""
    ### Deming vs. Passing-Bablok Regression: Choosing the Right Tool

    Deming Regression and Passing-Bablok Regression are both powerful statistical tools used in method comparison studies, particularly in laboratory medicine and analytical chemistry. They are employed when comparing two quantitative methods that measure the same analyte, especially when you cannot assume that one method provides "true" or error-free values. While both address the limitation of ordinary least squares (OLS) regression (which assumes the independent variable is error-free), they differ in their underlying assumptions and ideal use cases.

    | Feature                 | Deming Regression                                     | Passing-Bablok Regression                             |
    | :---------------------- | :---------------------------------------------------- | :---------------------------------------------------- |
    | **Method Type** | Parametric                                            | Non-Parametric                                        |
    | **Error Assumption** | Assumes errors are normally distributed.              | Makes no assumption about error distribution.         |
    | **Outlier Sensitivity** | Sensitive to outliers; outliers can influence the results. | Robust to outliers; less affected by their presence.   |
    | **Error Variance Ratio**| Requires an estimate of the ratio of error variances ($\lambda = \sigma_y^2 / \sigma_x^2$). Often assumed to be 1 if unknown. | Does not require an estimate of the error variance ratio. |
    | **Statistical Inference**| Provides standard errors, confidence intervals, and p-values for hypothesis testing. | Provides non-parametric confidence intervals (e.g., bootstrap-based). |
    | **Mathematical Basis** | Minimizes the weighted sum of squared perpendicular distances from points to the line. | Based on medians of slopes of connecting lines between data points. |
    | **Ideal Use Case** | When measurement errors are known to be normally distributed and/or the ratio of error variances is reliably estimated. When formal parametric statistical inference is needed. | When data or errors are not normally distributed, or when outliers are a significant concern and you want a more robust estimate. When the error variance ratio is unknown. |
    | **Interpretation** | Provides slope and intercept, and their statistical significance relative to 1 and 0, respectively. | Provides slope and intercept, which are interpreted similarly to Deming, but with less emphasis on parametric p-values directly from the regression. |

    **In Summary:**

    * **Choose Deming Regression** if your measurement errors are likely normally distributed and you have a good understanding or a reasonable assumption about the relative magnitudes of errors in both methods. It provides more detailed parametric statistical output.
    * **Choose Passing-Bablok Regression** if your data is suspected to contain outliers, or if you are unsure about the distribution of your errors. It offers a more robust analysis without strict distributional assumptions.

    Often, it's beneficial to perform both analyses. If the results are similar, it provides greater confidence in the conclusion. If they differ, it signals that the data might violate the assumptions of one method, guiding you towards the more appropriate one for your specific dataset.

    These tools support analytical method validation, troubleshooting, and cross-platform harmonization.
    
    
    """)


