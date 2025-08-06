import streamlit as st
from utils import apply_app_styling

apply_app_styling()

st.header("🏠 Overview: Statistical Tests")

st.markdown(
    """
    Welcome to the **Statistical Tests** section of the Validation and Verification App.

    This module provides access to **common statistical tools** used in method validation and quality assurance. 
    You can assess variance, central tendency, frequency distributions, and error tolerances across various tests.

    ---
    #### ❓ What statistical tests are available?
    - **Anderson-Darling Test**: Assess whether a sample comes from a specific distribution (commonly normal).
    - **Bartlett's Test**: Test for equality of variances across multiple groups (assumes normality).
    - **Chi-Squared Test**: Analyze categorical data using contingency tables for independence or goodness-of-fit.
    - **Cochran's Test**: Detect outliers in variance across multiple groups, often used in inter-laboratory studies.
    - **CUSUM Test**: Cumulative Sum Control Chart for detecting small shifts in process mean over time.
    - **F-test**: Compare variances between two independent samples to assess precision.
    - **Kolmogorov-Smirnov Test**: Compare a sample to a reference distribution, or compare two samples.
    - **Kruskal-Wallis Test**: Non-parametric alternative to ANOVA for comparing medians of multiple groups.
    - **Levene's Test**: Assess the equality of variances across groups, robust to non-normal distributions.
    - **Mann-Whitney U Test**: Non-parametric test to compare distributions between two independent samples.
    - **P-P Plots**: Probability-Probability plots for visual assessment of distributional fit.
    - **Q-Q Plots**: Quantile-Quantile plots for checking normality or comparing two distributions.
    - **Shapiro-Wilk Test**: Sensitive test for normality, suitable for small to moderate sample sizes.
    - **T-test**: Compare means between two groups (independent or paired); assumes normality.
    - **Total Allowable Error (TEa)**: Assess observed error against defined clinical performance limits.
    - **Z-test**: Evaluate whether a sample mean significantly differs from a known population mean (n > 30).
    ---
    #### 📂 What sort of file do I need? 
    Each test requires a CSV file formatted to match the test's input expectations. Make sure:
    - Headers are clearly labeled without any special characters (e.g., ! @ # $ % ^ & * ( ) + = [ ] | \ : ; " ' < > , ? / ` ~)
    - Numeric columns are clean (no special characters, text, or missing values). Please note, blank cells are permitted but it is useful to keep datasets as organised and clean as possible. 

    📌 **Tip**: Go to the Templates page to download the appropriate template for statistical testing.

    ---
    """
)


import streamlit as st
import streamlit.components.v1 as components

def show_test_selector():
    st.title("🔍 Statistical Test Selector")
    st.markdown("Answer a few questions to find the right statistical test for your analysis")
    
    # Initialize session state
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 'start'
    if 'selected_test' not in st.session_state:
        st.session_state.selected_test = None
    if 'path_history' not in st.session_state:
        st.session_state.path_history = []

    # Reset button
    if st.button("🔄 Start Over", type="secondary"):
        st.session_state.current_step = 'start'
        st.session_state.selected_test = None
        st.session_state.path_history = []
        st.rerun()

    # Show current path
    if st.session_state.path_history:
        st.markdown("**Your Path:**")
        path_text = " → ".join(st.session_state.path_history)
        st.markdown(f"*{path_text}*")
        st.markdown("---")

    # Decision tree logic
    if st.session_state.current_step == 'start':
        show_start_step()
    elif st.session_state.current_step == 'data_type':
        show_data_type_step()
    elif st.session_state.current_step == 'continuous_analysis':
        show_continuous_analysis_step()
    elif st.session_state.current_step == 'categorical_analysis':
        show_categorical_analysis_step()
    elif st.session_state.current_step == 'normality_check':
        show_normality_step()
    elif st.session_state.current_step == 'variance_check':
        show_variance_step()
    elif st.session_state.current_step == 'comparison_type':
        show_comparison_type_step()
    elif st.session_state.current_step == 'method_comparison':
        show_method_comparison_step()
    elif st.session_state.current_step == 'final_recommendation':
        show_final_recommendation()

def show_start_step():
    st.subheader("What type of analysis do you want to perform?")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Compare Groups/Methods", use_container_width=True):
            st.session_state.current_step = 'data_type'
            st.session_state.path_history.append("Compare Groups/Methods")
            st.rerun()
    
    with col2:
        if st.button("📈 Test Assumptions", use_container_width=True):
            st.session_state.current_step = 'normality_check'
            st.session_state.path_history.append("Test Assumptions")
            st.rerun()
    
    with col3:
        if st.button("🔗 Analyze Relationships", use_container_width=True):
            st.session_state.current_step = 'continuous_analysis'
            st.session_state.path_history.append("Analyze Relationships")
            st.rerun()

def show_data_type_step():
    st.subheader("What type of data are you working with?")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Continuous Data\n(measurements, scores)", use_container_width=True):
            st.session_state.current_step = 'continuous_analysis'
            st.session_state.path_history.append("Continuous Data")
            st.rerun()
    
    with col2:
        if st.button("📋 Categorical Data\n(categories, counts)", use_container_width=True):
            st.session_state.current_step = 'categorical_analysis'
            st.session_state.path_history.append("Categorical Data")
            st.rerun()

def show_continuous_analysis_step():
    st.subheader("What do you want to analyze?")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("⚖️ Compare 2 Groups", use_container_width=True):
            st.session_state.current_step = 'comparison_type'
            st.session_state.path_history.append("Compare 2 Groups")
            st.rerun()
    
    with col2:
        if st.button("📊 Compare 3+ Groups", use_container_width=True):
            st.session_state.current_step = 'variance_check'
            st.session_state.path_history.append("Compare 3+ Groups")
            st.rerun()
    
    with col3:
        if st.button("🔬 Method Comparison", use_container_width=True):
            st.session_state.current_step = 'method_comparison'
            st.session_state.path_history.append("Method Comparison")
            st.rerun()

def show_categorical_analysis_step():
    st.subheader("What type of categorical analysis?")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🎯 Independence Test\n(Chi-squared)", use_container_width=True):
            recommend_test("Chi Squared Test", 
                         "Tests independence between categorical variables",
                         "chi_squared")
    
    with col2:
        if st.button("📈 Goodness of Fit\n(Chi-squared)", use_container_width=True):
            recommend_test("Chi Squared Test", 
                         "Tests if data follows expected distribution",
                         "chi_squared")

def show_normality_step():
    st.subheader("Which normality test do you need?")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Shapiro-Wilk\n(Small samples)", use_container_width=True):
            recommend_test("Shapiro-Wilk Test", 
                         "Best for small samples (n < 50)",
                         "shapiro_wilk")
    
    with col2:
        if st.button("📈 Anderson-Darling\n(Medium samples)", use_container_width=True):
            recommend_test("Anderson Darling", 
                         "Good for medium to large samples",
                         "anderson_darling")
    
    with col3:
        if st.button("📋 Q-Q Plots\n(Visual assessment)", use_container_width=True):
            recommend_test("Q-Q Plots", 
                         "Visual assessment of normality",
                         "q_q")

def show_variance_check():
    st.subheader("Do your groups have equal variances?")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("✅ Yes (Equal Variances)", use_container_width=True):
            recommend_test("One-Way ANOVA", 
                         "Compares means of 3+ groups with equal variances",
                         "one_way_anova")
    
    with col2:
        if st.button("❌ No (Unequal Variances)", use_container_width=True):
            recommend_test("Kruskal-Wallis Test", 
                         "Non-parametric alternative for unequal variances",
                         "kruskal")
    
    with col3:
        if st.button("🤔 Need to Test Variances", use_container_width=True):
            recommend_test("Levene's Test", 
                         "Tests equality of variances across groups",
                         "levene_test")

def show_comparison_type_step():
    st.subheader("What type of two-group comparison?")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("👥 Independent Groups\n(different subjects)", use_container_width=True):
            st.session_state.current_step = 'independent_groups'
            st.session_state.path_history.append("Independent Groups")
            show_independent_groups()
    
    with col2:
        if st.button("🔄 Paired Groups\n(same subjects)", use_container_width=True):
            st.session_state.current_step = 'paired_groups'
            st.session_state.path_history.append("Paired Groups")
            show_paired_groups()

def show_independent_groups():
    st.subheader("Are your data normally distributed?")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("✅ Yes (Normal)", use_container_width=True):
            recommend_test("T-Test (Independent)", 
                         "Compares means of two independent groups",
                         "t_test")
    
    with col2:
        if st.button("❌ No (Non-normal)", use_container_width=True):
            recommend_test("Mann-Whitney U Test", 
                         "Non-parametric test for two independent groups",
                         "mann_whitney")

def show_paired_groups():
    st.subheader("Are the differences normally distributed?")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("✅ Yes (Normal)", use_container_width=True):
            recommend_test("T-Test (Paired)", 
                         "Compares paired observations",
                         "t_test")
    
    with col2:
        if st.button("❌ No (Non-normal)", use_container_width=True):
            recommend_test("Mann-Whitney U Test", 
                         "Non-parametric alternative for paired data",
                         "mann_whitney")

def show_method_comparison_step():
    st.subheader("What type of method comparison?")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Agreement Analysis\n(Bland-Altman)", use_container_width=True):
            recommend_test("Bland-Altman Analysis", 
                         "Assesses agreement between two methods",
                         "bland_altman")
    
    with col2:
        if st.button("📈 Regression Analysis\n(Deming/Passing-Bablok)", use_container_width=True):
            st.session_state.current_step = 'regression_choice'
            st.session_state.path_history.append("Regression Analysis")
            show_regression_choice()
    
    with col3:
        if st.button("🔬 Variance Analysis\n(ANOVA)", use_container_width=True):
            recommend_test("One-Way ANOVA", 
                         "Compares multiple methods simultaneously",
                         "one_way_anova")

def show_regression_choice():
    st.subheader("Which regression method?")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Deming Regression\n(Both variables have error)", use_container_width=True):
            recommend_test("Deming Regression", 
                         "Accounts for measurement error in both variables",
                         "deming")
    
    with col2:
        if st.button("📈 Passing-Bablok\n(Non-parametric)", use_container_width=True):
            recommend_test("Passing Bablok Regression", 
                         "Non-parametric regression for method comparison",
                         "passing_bablok")

def recommend_test(test_name, description, page_key):
    st.session_state.selected_test = {
        'name': test_name,
        'description': description,
        'page_key': page_key
    }
    st.session_state.current_step = 'final_recommendation'
    st.rerun()

def show_final_recommendation():
    test = st.session_state.selected_test
    
    st.success("🎯 Recommended Test")
    
    st.markdown(f"### {test['name']}")
    st.markdown(f"**Description:** {test['description']}")
    
    # Create navigation button
    st.markdown("---")
    st.markdown("**Ready to perform this test?**")
    
    # This would need to be integrated with your navigation system
    st.info(f"Navigate to: Statistical Tests → {test['name']}")
    
    # Show related tests
    show_related_tests(test['page_key'])

def show_related_tests(current_test):
    st.markdown("### 🔗 Related Tests You Might Need")
    
    related_tests = {
        'shapiro_wilk': ['Anderson Darling', 'Q-Q Plots'],
        't_test': ['Mann-Whitney U Test', 'Levene\'s Test'],
        'one_way_anova': ['Kruskal-Wallis Test', 'Levene\'s Test', 'Bartlett\'s Test'],
        'chi_squared': ['T-Test', 'Mann-Whitney U Test'],
        'bland_altman': ['Deming Regression', 'Passing Bablok Regression'],
        'deming': ['Bland-Altman Analysis', 'Linear Regression'],
        'passing_bablok': ['Bland-Altman Analysis', 'Deming Regression']
    }
    
    if current_test in related_tests:
        for test in related_tests[current_test]:
            st.markdown(f"• {test}")

# Add interactive flowchart visualization
def show_interactive_flowchart():
    st.markdown("### 📊 Interactive Decision Flowchart")
    
    flowchart_html = """
    <div style="text-align: center; padding: 20px; background-color: #f0f2f6; border-radius: 10px; margin: 20px 0;">
        <div style="font-size: 18px; font-weight: bold; margin-bottom: 20px;">Statistical Test Decision Tree</div>
        <div style="display: flex; justify-content: center; align-items: center; flex-wrap: wrap; gap: 20px;">
            <div style="background: #ffffff; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); min-width: 120px;">
                <div style="font-weight: bold; color: #1f77b4;">Start Here</div>
                <div style="font-size: 12px; margin-top: 5px;">Choose your analysis type</div>
            </div>
            <div style="font-size: 24px; color: #666;">→</div>
            <div style="background: #ffffff; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); min-width: 120px;">
                <div style="font-weight: bold; color: #ff7f0e;">Data Type</div>
                <div style="font-size: 12px; margin-top: 5px;">Continuous or Categorical</div>
            </div>
            <div style="font-size: 24px; color: #666;">→</div>
            <div style="background: #ffffff; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); min-width: 120px;">
                <div style="font-weight: bold; color: #2ca02c;">Test Selection</div>
                <div style="font-size: 12px; margin-top: 5px;">Perfect test for your needs</div>
            </div>
        </div>
    </div>
    """
    
    components.html(flowchart_html, height=200)

# Main function to call
def main():
    show_interactive_flowchart()
    show_test_selector()

if __name__ == "__main__":
    main()