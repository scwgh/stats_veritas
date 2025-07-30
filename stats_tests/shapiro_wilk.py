import streamlit as st
import pandas as pd
import scipy.stats as stats

def run():
    st.title("\U0001F4CA Shapiro-Wilk Normality Test")

    st.markdown("""
    The **Shapiro-Wilk test**, first proposed in 1965, calculates a _W_ statistic which tests whether a random sample comes from a normal distribution. Small values for _W_ are suggestive of departmture from nortmality, and percentage points for the _W_ statistic. The _W_ statistic is calculated as follows:""")
    st.latex(r'''W = \frac{\left( \sum_{i=1}^n a_i x_{(i)} \right)^2}{\sum_{i=1}^n (x_i - \bar{x})^2}''')

    st.markdown(""" where _x(i)_ are the ordered sample values adn x(1) is the smallest. The $\alpha$ i are constants generated from the means, variances and covariances of the order statistics of a sample of size _n_ from a normal distribution.
                \n  """)

