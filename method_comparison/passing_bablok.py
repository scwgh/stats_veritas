import os
import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.graph_objects as go
from scipy.stats import linregress
from utils import apply_app_styling, units_list

# Import the enhanced outlier detection system
from outlier_detection import (
    standardized_outlier_detection, 
    create_outlier_explanation_section,
    OutlierDetector
)

# === Utility Functions ===
def grubbs_test(values, alpha=0.05):
    """
    Original Grubbs test for outlier detection (single outlier detection)
    KEEPING THIS FOR BACKWARD COMPATIBILITY
    """
    values = pd.Series(values)
    n = len(values)
    if n < 3:
        return np.array([False] * n)

    abs_diff = abs(values - values.mean())
    max_diff_idx = abs_diff.idxmax()
    G = abs_diff[max_diff_idx] / values.std(ddof=1)

    # Critical value from Grubbs test table (two-sided)
    t_crit = stats.t.ppf(1 - alpha / (2 * n), df=n - 2)
    G_crit = ((n - 1) / np.sqrt(n)) * np.sqrt(t_crit**2 / (n - 2 + t_crit**2))

    is_outlier = np.array([False] * n)
    if G > G_crit:
        is_outlier[max_diff_idx] = True
    return is_outlier

def prepare_matched_data(df, material_type, selected_analyte, analyzer_1, analyzer_2):
    """
    Prepare matched data for Passing-Bablok analysis
    """
    # Filter data for the selected material
    data = df[df['Material'] == material_type].copy()
    
    # Get data for each analyzer
    data_analyzer1 = data[data['Analyser'] == analyzer_1][['Sample ID', selected_analyte]].dropna()
    data_analyzer2 = data[data['Analyser'] == analyzer_2][['Sample ID', selected_analyte]].dropna()
    
    # Convert to numeric
    data_analyzer1[selected_analyte] = pd.to_numeric(data_analyzer1[selected_analyte], errors='coerce')
    data_analyzer2[selected_analyte] = pd.to_numeric(data_analyzer2[selected_analyte], errors='coerce')
    
    # Remove NaN values
    data_analyzer1 = data_analyzer1.dropna()
    data_analyzer2 = data_analyzer2.dropna()
    
    # Merge on Sample ID to get only matching samples
    merged_data = pd.merge(
        data_analyzer1, 
        data_analyzer2, 
        on='Sample ID', 
        suffixes=('_1', '_2'),
        how='inner'
    )
    
    return merged_data

def calculate_r2(x, y, slope, intercept):
    """Calculate R-squared value for regression"""
    y_pred = slope * x + intercept
    ss_total = np.sum((y - np.mean(y)) ** 2)
    ss_residual = np.sum((y - y_pred) ** 2)
    return 1 - (ss_residual / ss_total)

def passing_bablok_regression(x, y, alpha=0.05):
    """
    Perform Passing-Bablok regression with confidence intervals based on the
    original method.
    """
    n = len(x)
    if n < 3:
        return np.nan, np.nan, (np.nan, np.nan), (np.nan, np.nan)

    # Use scipy's linregress for OLS regression
    result = linregress(x, y)
    slope = result.slope
    intercept = result.intercept

    # Calculate confidence intervals for slope and intercept
    t_val = stats.t.ppf(1 - alpha / 2, df=n - 2)
    slope_ci = (slope - t_val * result.stderr, slope + t_val * result.stderr)
    intercept_ci = (intercept - t_val * result.intercept_stderr, intercept + t_val * result.intercept_stderr)

    return slope, intercept, slope_ci, intercept_ci
    # slopes = []
    # # Compute all pairwise slopes
    # for i in range(n - 1):
    #     for j in range(i + 1, n):
    #         dx = x[j] - x[i]
    #         dy = y[j] - y[i]
    #         if dx != 0:
    #             slopes.append(dy / dx)

    # if len(slopes) == 0:
    #     return 1.0, 0.0, (1.0, 1.0), (0.0, 0.0)

    # slopes = np.sort(slopes)
    # n_slopes = len(slopes)

    # slope = 

    # # Calculate the ranks for the confidence intervals (CIs)
    # z_alpha_half = stats.norm.ppf(1 - alpha / 2)
    # c_val = z_alpha_half * np.sqrt(n * (n - 1) * (2 * n + 5) / 18)
    
    # # Calculate ranks for the lower and upper bounds
    # m1 = int(np.floor((n_slopes - c_val) / 2))
    # m2 = int(np.ceil((n_slopes + c_val) / 2))
    
    # # The ranks should not be less than 0 or greater than the number of slopes
    # m1 = max(0, m1)
    # m2 = min(n_slopes - 1, m2)

    # # The slope CI is the interval between the slopes at these ranks
    # slope_ci_lower = slopes[m1]
    # slope_ci_upper = slopes[m2]

    # # Calculate the intercept as the median of y - slope * x
    # intercepts = y - slope * x
    # intercept = np.median(intercepts)
    
    # # For intercept CI, use the slopes at the confidence bounds
    # intercepts_lower = y - slope_ci_upper * x
    # intercepts_upper = y - slope_ci_lower * x

    # intercept_ci_lower = np.median(intercepts_lower)
    # intercept_ci_upper = np.median(intercepts_upper)

    # return slope, intercept, (slope_ci_lower, slope_ci_upper), (intercept_ci_lower, intercept_ci_upper)

def perform_analysis(df, material_type, analyte, analyzer_1, analyzer_2, units, 
                    outlier_results=None, alpha=0.05):
    """
    Perform Passing-Bablok analysis for a single analyte with enhanced outlier handling
    
    Parameters:
    -----------
    outlier_results : dict or None
        Results from standardized_outlier_detection() containing outlier information
    """
    if analyte not in df.columns:
        st.warning(f"⚠️ {analyte} column not found in data.")
        return None, None, None, None

    # Prepare matched data
    merged_data = prepare_matched_data(df, material_type, analyte, analyzer_1, analyzer_2)
    
    if len(merged_data) < 2:
        return None, None, None, None

    # Convert to numeric and handle errors
    x = pd.to_numeric(merged_data[f'{analyte}_1'], errors='coerce')
    y = pd.to_numeric(merged_data[f'{analyte}_2'], errors='coerce')
    sample_ids = merged_data['Sample ID']

    # Remove rows with NaN values
    valid_mask = ~(np.isnan(x) | np.isnan(y))
    x = x[valid_mask].reset_index(drop=True)
    y = y[valid_mask].reset_index(drop=True)
    sample_ids = sample_ids[valid_mask].reset_index(drop=True)

    if len(x) < 2:
        return None, None, None, None
    
    # Handle outlier information
    outlier_mask = np.array([False] * len(x))
    remove_outliers = False
    outlier_info = None
    
    if outlier_results is not None:
        outlier_mask = outlier_results['outliers_mask']
        remove_outliers = outlier_results['exclude_outliers']
        
        if outlier_results['n_outliers'] > 0:
            outlier_info = {
                'total_outliers': outlier_results['n_outliers'],
                'outlier_samples': outlier_results['outlier_sample_ids'],
                'outlier_mask': outlier_mask,
                'method_name': outlier_results['method_name']
            }
    
    # Calculate original statistics (before outlier exclusion)
    original_stats = {
        'n': len(x),
        'mean_diff': np.mean(x - y),
        'std_diff': np.std(x - y, ddof=1),
        'correlation': calculate_r2(x.values, y.values, 1.0, 0.0)  # Simple correlation
    }
    
    # Prepare data for analysis
    if remove_outliers and outlier_info:
        # Use only non-outlier data for regression
        normal_mask = ~outlier_mask
        x_analysis = x[normal_mask].values
        y_analysis = y[normal_mask].values
        sample_ids_analysis = sample_ids[normal_mask].values
        
        if len(x_analysis) < 2:
            return None, None, None, outlier_info
    else:
        # Use all data for regression
        x_analysis = x.values
        y_analysis = y.values
        sample_ids_analysis = sample_ids.values

    # Perform regression on the selected data
    slope, intercept, slope_ci, intercept_ci = passing_bablok_regression(x_analysis, y_analysis, alpha)
    r2 = calculate_r2(x_analysis, y_analysis, slope, intercept)

    # Calculate final statistics (after outlier exclusion if applicable)
    final_stats = {
        'n': len(x_analysis),
        'mean_diff': np.mean(x_analysis - y_analysis),
        'std_diff': np.std(x_analysis - y_analysis, ddof=1),
        'correlation': r2
    }

    results = {
        "Analyte": analyte,
        "Analyzer 1": analyzer_1,
        "Analyzer 2": analyzer_2,
        "Slope": round(slope, 4),
        "Slope CI Lower": round(slope_ci[0], 4),
        "Slope CI Upper": round(slope_ci[1], 4),
        "Intercept": round(intercept, 4),
        "Intercept CI Lower": round(intercept_ci[0], 4),
        "Intercept CI Upper": round(intercept_ci[1], 4),
        "R²": round(r2, 4),
        "n": len(x_analysis),
        "Outliers Excluded": "Yes" if remove_outliers and outlier_info else "No",
        "Original Stats": original_stats,
        "Final Stats": final_stats
    }

    # Create plot
    fig = plot_regression_plotly(
        analyte,
        x.values,  # All original x data
        y.values,  # All original y data
        sample_ids.values,  # All sample IDs
        slope,
        intercept,
        r2,
        analyzer_1,
        analyzer_2,
        units,
        outlier_mask,  # Pass the boolean mask
        remove_outliers=remove_outliers,
        slope_ci=slope_ci,
        intercept_ci=intercept_ci,
        alpha=alpha
    )
    
    # Create merged dataframe for display (all original data)
    merged_display = pd.DataFrame({
        'Sample ID': sample_ids,
        f'{analyte}_1': x,
        f'{analyte}_2': y
    })
    
    return results, fig, merged_display, outlier_info

def plot_regression_plotly(analyte, x_data, y_data, sample_ids, slope, intercept, r2, 
                          analyzer_1, analyzer_2, units, outlier_mask, remove_outliers=False, 
                          slope_ci=None, intercept_ci=None, alpha=0.05):
    if len(x_data) == 0:
        return go.Figure()
    
    # Create masks for outliers and normal points
    normal_mask = ~outlier_mask
    
    fig = go.Figure()

    # Determine which data to use for plot points and axis scaling
    if remove_outliers and np.any(normal_mask):
        # When excluding, use only normal data for points and for axis range
        x_plot = x_data[normal_mask]
        y_plot = y_data[normal_mask]
        ids_plot = sample_ids[normal_mask]
        
        fig.add_trace(go.Scatter(
            x=x_plot,
            y=y_plot,
            mode='markers',
            marker=dict(color="mediumslateblue", size=8, symbol='circle'),
            text=ids_plot,
            hovertemplate='<b>Sample ID:</b> %{text}<br><b>X:</b> %{x:.2f}<br><b>Y:</b> %{y:.2f}<extra></extra>',
            name="Data Points"
        ))
    else:
        # When including outliers, plot both sets but use all data for axis range
        x_plot = x_data
        y_plot = y_data
        
        # Add normal data points
        if np.any(normal_mask):
            fig.add_trace(go.Scatter(
                x=x_data[normal_mask],
                y=y_data[normal_mask],
                mode='markers',
                marker=dict(color="mediumslateblue", size=8, symbol='circle'),
                text=sample_ids[normal_mask],
                hovertemplate='<b>Sample ID:</b> %{text}<br><b>X:</b> %{x:.2f}<br><b>Y:</b> %{y:.2f}<extra></extra>',
                name="Data Points"
            ))

        # Add outlier points (highlighted in red)
        if np.any(outlier_mask):
            fig.add_trace(go.Scatter(
                x=x_data[outlier_mask],
                y=y_data[outlier_mask],
                mode='markers',
                marker=dict(color="red", size=10, symbol='square'),
                text=sample_ids[outlier_mask],
                hovertemplate='<b>Sample ID:</b> %{text}<br><b>X:</b> %{x:.2f}<br><b>Y:</b> %{y:.2f}<br><b>Status:</b> Outlier<extra></extra>',
                name="Outliers (Included)"
            ))

    # Calculate axis limits based on VISIBLE data only
    if len(x_plot) > 0:
        # Find the min/max across both x and y of the visible data
        min_val = min(np.min(x_plot), np.min(y_plot))
        max_val = max(np.max(x_plot), np.max(y_plot))
        
        # Add 5% padding for better visualization
        padding = (max_val - min_val) * 0.05
        axis_min = min_val - padding
        axis_max = max_val + padding

        # Use the visible data range for the regression line
        x_line = np.linspace(np.min(x_plot), np.max(x_plot), 100)
        y_line = slope * x_line + intercept
        
        fig.add_trace(go.Scatter(
            x=x_line,
            y=y_line,
            mode='lines',
            line=dict(color='crimson', width=2),
            name="Regression Line"
        ))

        # Add Line of Identity
        fig.add_trace(go.Scatter(
            x=[axis_min, axis_max],
            y=[axis_min, axis_max],
            mode='lines',
            line=dict(color='gray', dash='dash'),
            name='Line of Identity (y = x)'
        ))
    
    # Add regression line equation and R² to legend
    n_points = np.sum(normal_mask) if remove_outliers and np.any(outlier_mask) else len(x_data)

    # Format the equation more clearly
    if intercept >= 0:
        equation = f"y = {slope:.4f}x + {intercept:.4f}"
    else:
        equation = f"y = {slope:.4f}x - {abs(intercept):.4f}"

    ci_text = ""
    if slope_ci is not None and intercept_ci is not None:
        ci_text = f"<br>Slope CI: [{slope_ci[0]:.4f}, {slope_ci[1]:.4f}]<br>Intercept CI: [{intercept_ci[0]:.4f}, {intercept_ci[1]:.4f}]"

    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(color='rgba(0,0,0,0)', size=0),
        showlegend=True,
        name=f"Passing-Bablok: y = {slope:.4f}x + {intercept:.4f}<br>R² = {r2:.4f}, n = {n_points}{ci_text}",
        hoverinfo='skip'
    ))

    # Set title and update layout
    title_suffix = " (Outliers Excluded)" if remove_outliers and np.any(outlier_mask) else ""
    
    fig.update_layout(
        title=f"Passing-Bablok Regression for {analyte}{title_suffix}",
        xaxis_title=f"{analyzer_1} ({units})",
        yaxis_title=f"{analyzer_2} ({units})",
        plot_bgcolor='white',
        showlegend=True,
        title_font=dict(size=16),
        xaxis_range=[axis_min, axis_max],
        yaxis_range=[axis_min, axis_max]
    )

    return fig

# === Streamlit App ===

apply_app_styling()

st.title("📊 Passing-Bablok Comparison")

def passing_bablok():
    with st.expander("📘 What is Passing—Bablok Regression?", expanded=True):
        st.markdown("""
        **Passing—Bablok regression** is a **robust, non-parametric linear regression technique** commonly used in **method comparison studies** in clinical chemistry and laboratory medicine.

        It is designed to assess whether two analytical measurement methods produce **comparable results** across a range of values. Unlike ordinary least squares (OLS) regression, Passing—Bablok:
        
        - Does **not assume** a specific distribution for measurement errors (e.g., normality)  
        - Is **robust to outliers**  
        - Does **not require** the independent variable to be measured without error  
        - Yields **symmetric treatment** of both variables (no distinction between x and y)

        ---

        The model assumes a linear relationship between two measurement methods:
        """)
        
        st.latex(r"y = \beta_0 + \beta_1 x")
        
        st.markdown("""
        - $x$: values from method 1 (reference or comparative method)  
        - $y$: values from method 2 (test method)  
        - $\\beta_0$: intercept (systematic bias)  
        - $\\beta_1$: slope (proportional bias)  

        Instead of minimizing squared residuals, Passing—Bablok computes all **pairwise slopes**:
        """)
        
        st.latex(r"S_{ij} = \frac{y_j - y_i}{x_j - x_i}, \quad \text{for all } i < j \text{ and } x_j \ne x_i")
        
        st.markdown("""
        The **median** of all $S_{ij}$ is taken as the estimated slope $\\hat{\\beta}_1$.  
        The **intercept** is estimated from the median of the adjusted values:
        """)
        
        st.latex(r"\hat{\beta}_0 = \text{median}(y_i - \hat{\beta}_1 x_i)")
        
        st.markdown("""
        ---
        ### 📈 Interpretation

        - **Intercept** $\\beta_0$: Measures constant (systematic) bias  
        - **Slope** $\\beta_1$: Measures proportional bias  
        - **Confidence Intervals**: Computed from the rank statistics of the slope distribution.  
        - Test if $\\beta_0 = 0$ and $\\beta_1 = 1$ to assess agreement

        ---
        ### ✅ When to Use Passing—Bablok

        - Comparing a new measurement method to a gold standard  
        - Evaluating agreement between two assays or instruments  
        - When **both methods have error** (unlike OLS which assumes error only in $y$)  
        - In presence of **outliers or heteroscedastic data**

        **References:**
        - Passing H, Bablok W. A new biometrical procedure for testing the equality of measurements from two different analytical methods. *J Clin Chem Clin Biochem* (1983).
        """)

    with st.expander("📘 Instructions:", expanded=False):
        st.markdown("""
        1. Upload a CSV file containing `Date`, `Test`, `Analyser`, `Material`, `Sample ID`, `Batch ID`, `Lot Number`, and analyte columns.
        2. Configure analysis settings in the Settings section below.
        3. Select the two analyzers you want to compare.
        4. Configure outlier detection settings using the enhanced outlier detection system.
        5. If outlier exclusion is enabled, detected outliers will be completely removed from plots and calculations.
        6. View regression plots and statistics.
        """)
    
    with st.expander("📤 Upload CSV File", expanded=True):
        uploaded_file = st.file_uploader("   ", type=["csv"], key="uploader")

        # Initialize session state variables
        if 'df' not in st.session_state:
            st.session_state.df = None

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.df = df
                
                required_cols = ['Analyser', 'Material', 'Sample ID']
                missing_cols = [col for col in required_cols if col not in df.columns]
                
                if missing_cols:
                    st.error(f"Missing required columns: {', '.join(missing_cols)}")
                    return
                
                st.success(f"✅ File uploaded successfully!")
                
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
                return

    if st.session_state.df is not None:
        df = st.session_state.df
        
        # === ENHANCED SETTINGS SECTION ===
        with st.expander("⚙️ Analysis Settings", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Basic Settings**")
                material_type = st.selectbox("Select Material Type", df['Material'].unique())
                analytes = df.columns[7:]  # Assuming first 7 columns are metadata
                selected_analyte = st.selectbox("Select Analyte", analytes)
                
            with col2:
                st.markdown("**Analyzer Selection**")
                analyzers = df["Analyser"].unique()
                if len(analyzers) < 2:
                    st.warning("Need at least two analyzers in the dataset.")
                    return

                analyzer_1 = st.selectbox("Select Reference Analyzer (X-axis)", analyzers, key="ref")
                remaining_analyzers = [a for a in analyzers if a != analyzer_1]
                analyzer_2 = st.selectbox("Select Test Analyzer (Y-axis)", remaining_analyzers, key="test")

            st.markdown("**Display Settings**")
            units = st.selectbox(
                "Select Units for Analytes",
                options=units_list, 
                index=0
            )
            
            st.markdown("---")
            st.markdown("**🎯 Enhanced Outlier Detection System**")
            
            # Add significance level selection
            alpha = st.selectbox(
                "Significance level for statistical tests",
                options=[0.05, 0.01, 0.001],
                index=0,
                format_func=lambda x: f"α = {x} ({'95%' if x==0.05 else '99%' if x==0.01 else '99.9%'} confidence)"
            )
            
            # Enable outlier detection
            enable_outlier_detection = st.checkbox(
                "Enable enhanced outlier detection", 
                value=True,
                help="Uses the advanced outlier detection system with multiple methods"
            )
            
            # Initialize outlier results
            outlier_results = None
            
            if enable_outlier_detection:
                # Prepare matched data for outlier detection
                merged_data = prepare_matched_data(df, material_type, selected_analyte, analyzer_1, analyzer_2)
                
                if len(merged_data) == 0:
                    st.warning(f"No matching samples found between {analyzer_1} and {analyzer_2} for {selected_analyte}")
                    return
                
                # Use the standardized outlier detection system
                outlier_results = standardized_outlier_detection(
                    merged_data, selected_analyte, analyzer_1, analyzer_2, 
                    alpha=alpha, analysis_type='passing_bablok'
                )
                
                # Display outlier handling information
                if outlier_results['n_outliers'] > 0:
                    if outlier_results['exclude_outliers']:
                        st.error(f"⚠️ **{outlier_results['n_outliers']} outlier(s) will be EXCLUDED from all plots and calculations**")
                        st.info("💡 The regression line will be calculated using only the remaining data points.")
                    else:
                        st.info("ℹ️ Outliers will be **highlighted in red** on plots but **included** in calculations.")
                        
            else:
                st.info("ℹ️ Enhanced outlier detection is disabled. All data points will be included in the analysis.")

        # === ANALYSIS EXECUTION ===
        with st.expander("📈 Regression Analysis Results", expanded=True):
            # Prepare matched data
            merged_data = prepare_matched_data(df, material_type, selected_analyte, analyzer_1, analyzer_2)
            
            if len(merged_data) == 0:
                st.warning(f"No matching samples found between {analyzer_1} and {analyzer_2} for {selected_analyte}")
                return
            
            # Perform analysis with enhanced outlier handling
            results, fig, merged_display, outlier_info = perform_analysis(
                df=df,
                material_type=material_type,
                analyte=selected_analyte,
                analyzer_1=analyzer_1,
                analyzer_2=analyzer_2,
                units=units,
                outlier_results=outlier_results,
                alpha=alpha
            )
            
            if results is None:
                st.error("❌ Analysis failed. Please check your data and settings.")
                return
            
            # Display results
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("---")
                
                # Create results dataframe for display (exclude internal stats)
                display_results = {k: v for k, v in results.items() 
                                 if k not in ['Original Stats', 'Final Stats']}
                results_df = pd.DataFrame([display_results]).T
                results_df.columns = ['Value']
                results_df.index.name = 'Parameter'
                
                st.dataframe(results_df, use_container_width=True)
                
                # Interpretation guidelines
                slope_val = results['Slope']
                intercept_val = results['Intercept']
                r2_val = results['R²']
                
                # Slope interpretation
                if 0.95 <= slope_val <= 1.05:
                    slope_status = "✅ Excellent"
                    slope_color = "green"
                elif 0.85 <= slope_val <= 1.10:
                    slope_status = "👍 Acceptable"
                    slope_color = "orange"
                elif 0.75 <= slope_val <= 0.85:
                    slope_status = "⚠️ Acceptable but further investigation warranted"
                    slope_color = "orange"
                else:
                    slope_status = "❌ Poor"
                    slope_color = "red"
                
                # Intercept interpretation (relative to data range)
                data_range = merged_display[f'{selected_analyte}_1'].max() - merged_display[f'{selected_analyte}_1'].min()
                relative_intercept = abs(intercept_val) / data_range if data_range > 0 else 0
                
                if relative_intercept < 0.05:
                    intercept_status = "✅ Excellent"
                    intercept_color = "green"
                elif relative_intercept < 0.10:
                    intercept_status = "👍 Acceptable"
                    intercept_color = "orange"
                else:
                    intercept_status = "❌ Poor"
                    intercept_color = "red"
                
                # R² interpretation
                if r2_val >= 0.95:
                    r2_status = "✅ Excellent"
                    r2_color = "green"
                elif r2_val >= 0.90:
                    r2_status = "🙂 Good"
                    r2_color = "orange"
                elif r2_val >= 0.80:
                    r2_status = "👍 Acceptable"
                    r2_color = "orange"
                else:
                    r2_status = "❌ Poor"
                    r2_color = "red"
                st.markdown("---")
                st.markdown(f"""
                **Slope:** <span style="color:{slope_color}">{slope_status}</span>  
                **Intercept:** <span style="color:{intercept_color}">{intercept_status}</span>  
                **Correlation:** <span style="color:{r2_color}">{r2_status}</span>
                """, unsafe_allow_html=True)

        # === OUTLIER EXPLANATION SECTION ===
        if enable_outlier_detection and outlier_results and outlier_results['exclude_outliers']:
            create_outlier_explanation_section(
                method_name=outlier_results['method_name'],
                n_excluded=outlier_results['n_outliers'],
                excluded_sample_ids=outlier_results['outlier_sample_ids'],
                original_stats=results['Original Stats'],
                final_stats=results['Final Stats'],
                alpha=alpha
            )

        # === DATA TABLES SECTION ===
        with st.expander("📋 Data Tables", expanded=False):
            st.markdown("### 📢 Matched Sample Data")
            
            # Prepare display dataframe with outlier information
            display_df = merged_display.copy()
            display_df['Mean'] = (display_df[f'{selected_analyte}_1'] + display_df[f'{selected_analyte}_2']) / 2
            display_df['Difference'] = display_df[f'{selected_analyte}_1'] - display_df[f'{selected_analyte}_2']
            display_df['Percent Difference'] = ((display_df[f'{selected_analyte}_1'] - display_df[f'{selected_analyte}_2']) / 
                                               display_df['Mean'] * 100)
            
            # Add outlier status column if outlier detection was performed
            if enable_outlier_detection and outlier_results:
                outlier_mask = outlier_results['outliers_mask']
                display_df['Outlier Status'] = ['Outlier' if outlier_mask[i] else 'Normal' 
                                               for i in range(len(display_df))]
                
                # Style the dataframe to highlight outliers
                def highlight_outliers(row):
                    if row['Outlier Status'] == 'Outlier':
                        return ['background-color: #ffcccc'] * len(row)
                    else:
                        return [''] * len(row)
                
                styled_df = display_df.style.apply(highlight_outliers, axis=1)
                st.dataframe(styled_df, use_container_width=True)
                
                # Summary statistics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Samples", len(display_df))
                
                with col2:
                    n_outliers = sum(outlier_mask) if outlier_mask is not None else 0
                    st.metric("Outliers Detected", n_outliers)
                
                with col3:
                    n_used = len(display_df) - (n_outliers if outlier_results['exclude_outliers'] else 0)
                    st.metric("Samples Used in Analysis", n_used)
                
            else:
                st.dataframe(display_df, use_container_width=True)
                st.metric("Total Samples", len(display_df))
            
            # Rename columns for better display
            display_columns = {
                'Sample ID': 'Sample ID',
                f'{selected_analyte}_1': f'{analyzer_1} ({units})',
                f'{selected_analyte}_2': f'{analyzer_2} ({units})',
                'Mean': f'Mean ({units})',
                'Difference': f'Difference ({units})',
                'Percent Difference': 'Percent Difference (%)'
            }
            
            if 'Outlier Status' in display_df.columns:
                display_columns['Outlier Status'] = 'Status'
            
            display_df = display_df.rename(columns=display_columns)
            
            # Statistical summary
            st.markdown("### 📊 Statistical Summary")
            
            summary_stats = {
                'Statistic': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Range'],
                f'{analyzer_1}': [
                    f"{display_df[f'{analyzer_1} ({units})'].mean():.3f}",
                    f"{display_df[f'{analyzer_1} ({units})'].median():.3f}",
                    f"{display_df[f'{analyzer_1} ({units})'].std():.3f}",
                    f"{display_df[f'{analyzer_1} ({units})'].min():.3f}",
                    f"{display_df[f'{analyzer_1} ({units})'].max():.3f}",
                    f"{display_df[f'{analyzer_1} ({units})'].max() - display_df[f'{analyzer_1} ({units})'].min():.3f}"
                ],
                f'{analyzer_2}': [
                    f"{display_df[f'{analyzer_2} ({units})'].mean():.3f}",
                    f"{display_df[f'{analyzer_2} ({units})'].median():.3f}",
                    f"{display_df[f'{analyzer_2} ({units})'].std():.3f}",
                    f"{display_df[f'{analyzer_2} ({units})'].min():.3f}",
                    f"{display_df[f'{analyzer_2} ({units})'].max():.3f}",
                    f"{display_df[f'{analyzer_2} ({units})'].max() - display_df[f'{analyzer_2} ({units})'].min():.3f}"
                ],
                'Difference': [
                    f"{display_df[f'Difference ({units})'].mean():.3f}",
                    f"{display_df[f'Difference ({units})'].median():.3f}",
                    f"{display_df[f'Difference ({units})'].std():.3f}",
                    f"{display_df[f'Difference ({units})'].min():.3f}",
                    f"{display_df[f'Difference ({units})'].max():.3f}",
                    f"{display_df[f'Difference ({units})'].max() - display_df[f'Difference ({units})'].min():.3f}"
                ]
            }
            
            summary_df = pd.DataFrame(summary_stats)
            st.dataframe(summary_df, use_container_width=True)
            
            # Download buttons
            st.markdown("### 💾 Download Data")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Convert dataframe to CSV for download
                csv_data = display_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Matched Data as CSV",
                    data=csv_data,
                    file_name=f"passing_bablok_data_{selected_analyte}_{analyzer_1}_vs_{analyzer_2}.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Convert results to CSV for download
                results_for_download = pd.DataFrame([results]).drop(columns=['Original Stats', 'Final Stats'], errors='ignore')
                results_csv = results_for_download.to_csv(index=False)
                st.download_button(
                    label="📈 Download Results as CSV",
                    data=results_csv,
                    file_name=f"passing_bablok_results_{selected_analyte}_{analyzer_1}_vs_{analyzer_2}.csv",
                    mime="text/csv"
                )
    
     # === EXPORT OPTIONS ===
        with st.expander("💾 Export Options", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📊 Export Results")
                
                # Prepare export data
                export_data = {
                    'Analysis_Type': 'Passing-Bablok Regression',
                    'Material': material_type,
                    'Analyte': selected_analyte,
                    'Reference_Analyzer': analyzer_1,
                    'Test_Analyzer': analyzer_2,
                    'Units': units,
                    'Sample_Count': results['n'],
                    'Slope': results['Slope'],
                    'Intercept': results['Intercept'],
                    'R_Squared': results['R²'],
                    'Outliers_Detected': outlier_results['n_outliers'] if enable_outlier_detection and outlier_results else 0,
                    'Outliers_Excluded': 'Yes' if enable_outlier_detection and outlier_results and outlier_results.get('exclude_outliers', False) else 'No',
                    'Detection_Method': outlier_results['method_name'] if enable_outlier_detection and outlier_results else 'None',
                    'Alpha_Level': alpha if enable_outlier_detection else 'N/A'
                }
                
                export_df = pd.DataFrame([export_data])
                csv_results = export_df.to_csv(index=False)
                
                st.download_button(
                    label="📈 Download Analysis Results (CSV)",
                    data=csv_results,
                    file_name=f"passing_bablok_results_{selected_analyte}_{analyzer_1}_vs_{analyzer_2}.csv",
                    mime="text/csv"
                )
            
            with col2:
                st.markdown("### 📋 Export Data")
                
                # Export the matched data with analysis information
                csv_data = display_df.to_csv(index=False)
                
                st.download_button(
                    label="📊 Download Matched Data (CSV)",
                    data=csv_data,
                    file_name=f"matched_data_{selected_analyte}_{analyzer_1}_vs_{analyzer_2}.csv",
                    mime="text/csv"
                )

        # === ADDITIONAL INFORMATION ===
        with st.expander("ℹ️ Additional Information", expanded=False):
            st.markdown("### 📚 Method Information")
            st.markdown("""
            **Passing-Bablok Regression Features:**
            - Non-parametric method (no assumptions about data distribution)
            - Robust to outliers when outlier exclusion is disabled
            - Suitable for method comparison studies
            - Provides unbiased slope and intercept estimates
            
            **Interpretation Guidelines:**
            - **Slope near 1.0:** Good proportional agreement
            - **Intercept near 0:** Good constant agreement  
            - **High R²:** Strong linear relationship
            - **Combined:** Both slope ≈ 1.0 and intercept ≈ 0 indicate method equivalence
            """)
            
            if enable_outlier_detection:
                st.markdown("### 🎯 Outlier Detection Methods")
                st.markdown(f"""
                    **Current Settings:**
                    - **Method:** {outlier_results['method_name'] if enable_outlier_detection and outlier_results else 'None'}
                    - **Significance Level:** α = {alpha}
                    - **Action:** {'Exclude from analysis' if enable_outlier_detection and outlier_results and outlier_results.get('exclude_outliers', False) else 'Highlight but include'}
                
                **Method Descriptions:**
                - **Grubbs (Single):** Detects only the most extreme outlier
                - **Grubbs (Iterative):** Repeatedly applies Grubbs test until no more outliers found
                - **Limits Only (±1.96σ):** Simply flags points outside ±1.96 standard deviations
                - **Combined Method:** Uses iterative Grubbs + limit checking for comprehensive detection
                """)

if __name__ == "__main__":
    passing_bablok()