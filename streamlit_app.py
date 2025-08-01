import streamlit as st
import sqlite3
import pandas as pd
from datetime import datetime
# from signup_utils import init_db, save_signup, get_signups
import streamlit.components.v1 as components
from utils import apply_app_styling, show_footer

# Home 
home_page = st.Page("home.py", title="Home")
templates_page = st.Page("pages/templates_home.py", title="Templates")
# support_page = st.Page("support.py", title="Contact and Support")
# Method Comparison
test_selector_page = st.Page("method_comparison/test_selector.py", title="Choose a test")
method_comparison_page = st.Page("method_comparison/overview_mc.py", title="Overview")
one_way_anova_page = st.Page("method_comparison/anova/one_way_anova.py", title="One-Way ANOVA")
multi_way_anova_page = st.Page("method_comparison/anova/multi_way_anova.py", title="Multi-Way ANOVA")
two_crossed_anova = st.Page("method_comparison/anova/two_crossed_anova.py", title="Two-Way Crossed ANOVA")
two_nested_anova = st.Page("method_comparison/anova/two_nested_anova.py", title="Two-Way Nested ANOVA")
bland_altman_page = st.Page("method_comparison/bland_altman.py", title="Bland-Altman Analysis")
deming_page = st.Page("method_comparison/deming_regression.py", title="Deming Regression")
passing_bablok_page = st.Page("method_comparison/passing_bablok.py", title="Passing Bablok Regression")
# Linearity
linearity_page = st.Page("regression/linearity.py", title="Linearity Analysis")
linear_regression_page = st.Page("regression/linear_regression.py", title="Linear Regression")
poly_regression_page = st.Page("regression/polynomial_regression.py", title="Polynomial Regression")
# Imprecision
imprecision_page = st.Page("pages/imprecision.py", title="Imprecision Analysis")
# Statistical tests
stats_overview_page = st.Page("stats_tests/stats_overview.py", title="Overview")
anderson_darling_page = st.Page("stats_tests/anderson_darling.py", title="Anderson Darling")
bartlett_page = st.Page("stats_tests/bartlett_test.py", title="Bartlett's Test")
chi_squared_page = st.Page("stats_tests/chi_squared.py", title="Chi Squared Test")
cochran_page = st.Page("stats_tests/cochran.py", title="Cochran Test")
cusum_page = st.Page("stats_tests/cusum.py", title="Cusum Test")
ftest_page = st.Page("stats_tests/f_test.py", title="F-Test")
k_smirnov_page = st.Page("stats_tests/kolmogorov_smirnov.py", title="Kolmogorov Smironov Test")
kruskal_page = st.Page("stats_tests/kruskal_wallis.py", title="Kruskal-Wallis Test")
levene_page = st.Page("stats_tests/levene_test.py", title="Levene's Test")
mann_whitney_page = st.Page("stats_tests/mann_whitney_u.py", title="Mann-Whitney U Test")
p_p_page = st.Page("stats_tests/p_p_plots.py", title="P-P Plots")
q_q_page = st.Page("stats_tests/q_q_plots.py", title="Q-Q Plots")
shapiro_wilk_page = st.Page("stats_tests/shapiro_wilk.py", title="Shapiro-Wilk Test")
t_test_page = st.Page("stats_tests/t_test.py", title="T-Test")
tea_page = st.Page("stats_tests/tea.py", title="Total Allowable Error")
z_test_page = st.Page("stats_tests/z_test.py", title="Z-Test")
# Limits
limits_page = st.Page("pages/limits.py", title="Limits")
# Reference Intervals
reference_intervals_page = st.Page("pages/reference_intervals.py", title="Reference Interval Calculator")

# Groups
pg = st.navigation({
    "Main": [home_page, templates_page],
    "Imprecision": [imprecision_page],
    "Limits": [limits_page],
    "Linearity": [linearity_page, poly_regression_page, linear_regression_page],
    "Method Comparison": [method_comparison_page, 
                          test_selector_page,
                          bland_altman_page, deming_page, 
                          passing_bablok_page, one_way_anova_page, 
                          two_crossed_anova, two_nested_anova, multi_way_anova_page],
    "Reference Interval Calculator": [reference_intervals_page],
    "Statistical Tests": [stats_overview_page,  
                          anderson_darling_page, 
                          bartlett_page, chi_squared_page, cochran_page, 
                          cusum_page, ftest_page, k_smirnov_page, kruskal_page, 
                          levene_page, mann_whitney_page, p_p_page, q_q_page, 
                          shapiro_wilk_page, t_test_page, tea_page, z_test_page]
    
}, position="top")

pg.run()

