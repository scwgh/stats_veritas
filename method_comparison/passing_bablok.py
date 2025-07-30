import os
import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.graph_objects as go
from scipy.stats import linregress
from utils import apply_app_styling, units_list

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

def grubbs_test_iterative(values, alpha=0.05, max_iterations=10):
    """
    Perform iterative Grubbs test for outlier detection.
    This will continue removing outliers until no more are found or max_iterations is reached.
    """
    values = pd.Series(values).copy()
    outlier_indices = []
    original_indices = values.index.tolist()
    
    for iteration in range(max_iterations):
        n = len(values)
        if n < 3:  # Need at least 3 points for Grubbs test
            break
            
        # Calculate Grubbs statistic
        mean_val = values.mean()
        std_val = values.std(ddof=1)
        
        if std_val == 0:  # All values are the same
            break
            
        abs_diff = abs(values - mean_val)
        max_diff_idx = abs_diff.idxmax()
        G = abs_diff[max_diff_idx] / std_val
        
        # Critical value from Grubbs test table (two-sided)
        t_crit = stats.t.ppf(1 - alpha / (2 * n), df=n - 2)
        G_crit = ((n - 1) / np.sqrt(n)) * np.sqrt(t_crit**2 / (n - 2 + t_crit**2))
        
        if G > G_crit:
            # Found an outlier
            outlier_indices.append(max_diff_idx)
            values = values.drop(max_diff_idx)
        else:
            # No more outliers found
            break
    
    # Create boolean mask for original data
    is_outlier = np.array([idx in outlier_indices for idx in original_indices])
    return is_outlier

def identify_outliers_beyond_limits(differences, mean_diff, std_diff, alpha=0.05, method='grubbs_iterative'):
    """
    Identify outliers using multiple methods and return the most comprehensive result.
    
    Parameters:
    - differences: array of difference values
    - mean_diff: mean of differences
    - std_diff: standard deviation of differences
    - alpha: significance level for Grubbs test
    - method: 'grubbs_iterative', 'grubbs_single', 'limits_only', or 'combined'
    """
    n = len(differences)
    
    if method == 'grubbs_iterative':
        return grubbs_test_iterative(differences, alpha)
    
    elif method == 'grubbs_single':
        return grubbs_test(differences, alpha)
    
    elif method == 'limits_only':
        # Simply identify points outside ±1.96 SD limits
        loa_upper = mean_diff + 1.96 * std_diff
        loa_lower = mean_diff - 1.96 * std_diff
        return (differences > loa_upper) | (differences < loa_lower)
    
    elif method == 'combined':
        # Combine Grubbs test with limit-based detection
        grubbs_outliers = grubbs_test_iterative(differences, alpha)
        
        # Recalculate limits after removing Grubbs outliers
        if grubbs_outliers.any():
            clean_diffs = differences[~grubbs_outliers]
            if len(clean_diffs) > 0:
                clean_mean = np.mean(clean_diffs)
                clean_std = np.std(clean_diffs, ddof=1)
                clean_upper = clean_mean + 1.96 * clean_std
                clean_lower = clean_mean - 1.96 * clean_std
                
                # Check if any remaining points are outside the new limits
                limit_outliers = (differences > clean_upper) | (differences < clean_lower)
                return grubbs_outliers | limit_outliers
        
        return grubbs_outliers
    
    else:
        raise ValueError("Method must be 'grubbs_iterative', 'grubbs_single', 'limits_only', or 'combined'")

def enhanced_outlier_analysis(diffs, x_vals, y_vals, alpha=0.05):
    """
    Perform comprehensive outlier analysis and return detailed information.
    """
    n = len(diffs)
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs, ddof=1)
    
    # Calculate percentage differences for each point
    means = (x_vals + y_vals) / 2
    percent_diffs = np.abs(diffs / means) * 100
    
    # Apply different outlier detection methods
    methods = {
        'Grubbs (Single)': identify_outliers_beyond_limits(diffs, mean_diff, std_diff, alpha, 'grubbs_single'),
        'Grubbs (Iterative)': identify_outliers_beyond_limits(diffs, mean_diff, std_diff, alpha, 'grubbs_iterative'),
        'Limits Only (±1.96σ)': identify_outliers_beyond_limits(diffs, mean_diff, std_diff, alpha, 'limits_only'),
        'Large % Difference (>50%)': percent_diffs > 50,  # Flag points with >50% difference
        'Combined Method': identify_outliers_beyond_limits(diffs, mean_diff, std_diff, alpha, 'combined')
    }
    
    # Create a comprehensive method that combines statistical outliers with large percentage differences
    comprehensive_outliers = methods['Grubbs (Iterative)'] | methods['Large % Difference (>50%)']
    methods['Comprehensive (Grubbs + %Diff)'] = comprehensive_outliers
    
    # Calculate limits of agreement
    loa_upper = mean_diff + 1.96 * std_diff
    loa_lower = mean_diff - 1.96 * std_diff
    
    results = {
        'methods': methods,
        'limits': {'upper': loa_upper, 'lower': loa_lower, 'mean': mean_diff, 'std': std_diff},
        'n_total': n,
        'percent_diffs': percent_diffs
    }
    
    return results

def perform_outlier_detection_with_options(merged_data, selected_analyte, analyzer_1, analyzer_2, alpha):
    """
    Enhanced outlier detection with multiple method options for the user.
    """
    # Extract values and calculate differences
    vals1 = merged_data[f'{selected_analyte}_1']
    vals2 = merged_data[f'{selected_analyte}_2']
    diffs_initial = vals1 - vals2
    
    # Perform comprehensive outlier analysis
    outlier_analysis = enhanced_outlier_analysis(diffs_initial.values, vals1.values, vals2.values, alpha=alpha)
    
    # Display results for each method
    st.markdown("**Outlier Detection Method Comparison:**")
    
    method_results = {}
    for method_name, is_outlier in outlier_analysis['methods'].items():
        n_outliers = sum(is_outlier)
        method_results[method_name] = {
            'outliers': is_outlier,
            'count': n_outliers,
            'indices': np.where(is_outlier)[0] if n_outliers > 0 else []
        }
    
    # Create comparison table
    comparison_data = []
    for method_name, result in method_results.items():
        comparison_data.append({
            'Method': method_name,
            'Outliers Found': result['count'],
            'Sample IDs': ', '.join(merged_data['Sample ID'].iloc[result['indices']].astype(str).tolist()) if result['count'] > 0 else 'None'
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    # Let user choose method
    method_choice = st.selectbox(
        "Select outlier detection method:",
        options=list(outlier_analysis['methods'].keys()),
        index=5,  # Default to 'Comprehensive' method
        help="""
        - **Grubbs (Single)**: Detects only the most extreme outlier
        - **Grubbs (Iterative)**: Repeatedly applies Grubbs test until no more outliers found
        - **Limits Only (±1.96σ)**: Simply flags points outside ±1.96 standard deviations
        - **Large % Difference (>50%)**: Flags points with >50% relative difference
        - **Combined Method**: Uses iterative Grubbs + limit checking
        - **Comprehensive**: Combines Grubbs iterative + large percentage differences (RECOMMENDED)
        """
    )
    
    selected_outliers = outlier_analysis['methods'][method_choice]
    
    # Always get outlier details (even if empty) to avoid NameError
    outlier_indices = np.where(selected_outliers)[0]
    
    if len(outlier_indices) == 0:
        st.success(f"✅ No outliers detected using {method_choice}.")
        return selected_outliers, method_choice
    else:
        # Get outlier details
        outlier_sample_ids = merged_data['Sample ID'].iloc[outlier_indices].tolist()
        outlier_vals1 = vals1.iloc[outlier_indices].tolist()
        outlier_vals2 = vals2.iloc[outlier_indices].tolist()
        outlier_diffs = diffs_initial.iloc[outlier_indices].tolist()
        
        st.error(f"⚠️ {len(outlier_indices)} outlier(s) detected using {method_choice}:")
        
        # Create outlier details table
        outlier_details = []
        for i, idx in enumerate(outlier_indices):
            # Calculate how far outside limits this point is
            mean_diff = np.mean(diffs_initial)
            std_diff = np.std(diffs_initial, ddof=1)
            z_score = abs(outlier_diffs[i] - mean_diff) / std_diff
            
            # Calculate percentage difference
            mean_val = (outlier_vals1[i] + outlier_vals2[i]) / 2
            percent_diff = abs(outlier_diffs[i] / mean_val) * 100 if mean_val != 0 else 0
            
            outlier_details.append({
                'Sample ID': outlier_sample_ids[i],
                f'{analyzer_1}': round(outlier_vals1[i], 3),
                f'{analyzer_2}': round(outlier_vals2[i], 3),
                'Difference': round(outlier_diffs[i], 3),
                'Mean': round(mean_val, 3),
                '% Difference': round(percent_diff, 1),
                'Z-Score': round(z_score, 2),
                'Outside Limits': '✓' if abs(z_score) > 1.96 else '✗'
            })
        
        outlier_df = pd.DataFrame(outlier_details)
        st.dataframe(outlier_df, use_container_width=True, hide_index=True)
        
        return selected_outliers, method_choice

def prepare_matched_data(df, material_type, selected_analyte, analyzer_1, analyzer_2):
    """
    Prepare matched data for Bland-Altman analysis
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

def passing_bablok_regression(x, y):
    """Perform Passing-Bablok regression"""
    n = len(x)

    slopes = []

    # Compute all pairwise slopes
    for i in range(n - 1):
        for j in range(i + 1, n):
            dx = x[j] - x[i]
            dy = y[j] - y[i]
            if dx != 0:
                slopes.append(dy / dx)

    if len(slopes) == 0:
        return 1.0, 0.0  # Default values if no slopes can be calculated

    slopes = np.sort(slopes)
    n_slopes = len(slopes)

    # Get median slope
    if n_slopes % 2 == 1:
        slope = slopes[n_slopes // 2]
    else:
        slope = 0.5 * (slopes[n_slopes // 2 - 1] + slopes[n_slopes // 2])

    # Calculate intercept as median of y - slope * x
    intercepts = y - slope * x
    intercept = np.median(intercepts)

    return slope, intercept

def detect_outliers_grubbs(data, alpha=0.05):
    """Detect outliers using Grubbs test - returns boolean mask"""
    try:
        outlier_mask = grubbs_test_iterative(data, alpha=alpha)
        return outlier_mask
    except Exception as e:
        st.warning(f"Grubbs test failed: {str(e)}")
        return np.array([False] * len(data))

def perform_analysis(df, material_type, analyte, analyzer_1, analyzer_2, units, outlier_mask=None, remove_outliers=False, alpha=0.05):
    """Perform Passing-Bablok analysis for a single analyte with outlier handling"""
    if analyte not in df.columns:
        st.warning(f"⚠ {analyte} column not found in data.")
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
    
    # Use provided outlier mask or detect outliers
    if outlier_mask is None:
        # Detect outliers using Grubbs test on both x and y data
        outliers_x = detect_outliers_grubbs(x.values, alpha)
        outliers_y = detect_outliers_grubbs(y.values, alpha)
        outlier_mask = outliers_x | outliers_y
    
    outlier_indices = np.where(outlier_mask)[0]
    
    outlier_info = None
    if len(outlier_indices) > 0:
        outlier_info = {
            'total_outliers': len(outlier_indices),
            'outlier_samples': sample_ids.iloc[outlier_indices].tolist(),
            'outlier_mask': outlier_mask
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
    slope, intercept = passing_bablok_regression(x_analysis, y_analysis)
    r2 = calculate_r2(x_analysis, y_analysis, slope, intercept)

    results = {
        "Analyte": analyte,
        "Analyzer 1": analyzer_1,
        "Analyzer 2": analyzer_2,
        "Slope": round(slope, 4),
        "Intercept": round(intercept, 4),
        "R²": round(r2, 4),
        "n": len(x_analysis),
        "Outliers Excluded": "Yes" if remove_outliers and outlier_info else "No"
    }

    # Create plot - pass the exclusion flag to determine what data to show
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
        remove_outliers=remove_outliers
    )
    
    # Create merged dataframe for display (all original data)
    merged_display = pd.DataFrame({
        'Sample ID': sample_ids,
        f'{analyte}_1': x,
        f'{analyte}_2': y
    })
    
    return results, fig, merged_display, outlier_info

def plot_regression_plotly(analyte, x_data, y_data, sample_ids, slope, intercept, r2, analyzer_1, analyzer_2, units, outlier_mask, remove_outliers=False):
    """Create Plotly regression plot with proper outlier handling and axis adjustment"""
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

    # --- FIX: Calculate axis limits based on VISIBLE data only ---
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

        # --- FIX: Update Line of Identity to use the new, adjusted axis limits ---
        fig.add_trace(go.Scatter(
            x=[axis_min, axis_max],
            y=[axis_min, axis_max],
            mode='lines',
            line=dict(color='gray', dash='dash'),
            name='Line of Identity (y = x)'
        ))
    
    # Add equation and R² to legend
    n_points = np.sum(normal_mask) if remove_outliers and np.any(outlier_mask) else len(x_data)
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(color='rgba(0,0,0,0)', size=0),
        showlegend=True,
        name=f"Passing-Bablok: y = {slope:.4f}x + {intercept:.4f}<br>R² = {r2:.4f}, n = {n_points}",
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
        # --- FIX: Explicitly set the axis ranges to match the visible data ---
        xaxis_range=[axis_min, axis_max],
        yaxis_range=[axis_min, axis_max]
    )

    return fig

# === Streamlit App ===

apply_app_styling()

st.title("📊 Passing-Bablok Comparison")

def passing_bablok():
    with st.expander("📘 What is Passing Bablok Regression?", expanded=False):
        st.markdown("""
        **Passing Bablok regression** is a **non-parametric method comparison technique** that is robust to outliers and does not assume a specific error distribution.
        
        - Uses median-based calculations instead of least squares
        - Suitable for method comparison studies
        - Does not assume normal distribution of errors
        - Provides slope and intercept estimates with confidence intervals
        """)

    with st.expander("📘 Instructions:", expanded=False):
        st.markdown("""
        1. Upload a CSV file containing `Date`, `Test`, `Analyser`, `Material`, `Sample ID`, `Batch ID`, `Lot Number`, and analyte columns.
        2. Configure analysis settings in the Settings section below.
        3. Select the two analyzers you want to compare.
        4. Configure outlier detection settings and choose whether to exclude outliers from analysis.
        5. If outlier exclusion is enabled, detected outliers will be completely removed from plots and calculations.
        6. View regression plots and statistics.
        """)
    with st.expander("📘 How Outliers Are Identified", expanded=False):
        st.markdown("""
        - **Grubbs (Single)**  
        Detects the **single most extreme** outlier using the original Grubbs test at the selected significance level (α). Only one point is flagged even if others may appear abnormal.

        - **Grubbs (Iterative)**  
        Applies the Grubbs test **repeatedly**, removing the most extreme outlier on each iteration, until **no more significant outliers** are detected or the iteration limit is reached. This is useful when multiple outliers may be present.

        - **Limits Only (±1.96σ)**  
        Flags any data points that lie **outside ±1.96 standard deviations** from the mean of the differences between analyzers. This corresponds roughly to a 95% confidence interval assuming normality.

        - **Large % Difference (>50%)**  
        Flags points where the **relative difference** between analyzers exceeds **50%**, regardless of whether the absolute values are outliers.

        - **Combined Method**  
        First removes outliers using **Grubbs (Iterative)**. Then recalculates the mean and standard deviation using the cleaned data and flags any **remaining points beyond ±1.96σ** from the new mean.

        - **Comprehensive (Grubbs + %Diff)**  
        Flags points that are outliers **according to Grubbs (Iterative)** **or** that show a **>50% relative difference**. This method aims to capture both statistically extreme values and clinically significant disagreements.

        ---
        ✅ By default, the **Comprehensive** method is selected and **recommended** for robust analysis.

        ⚙️ If outlier exclusion is enabled, flagged data points are **completely excluded** from regression and correlation calculations and **not shown** in regression plots.

        🔴 If exclusion is disabled, outliers are **highlighted in red** but included in calculations.

        ℹ️ You can change the method and toggle exclusion settings under the "Analysis Settings" section.
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
            st.markdown("**🎯 Outlier Detection & Exclusion Settings**")
            
            # Add significance level selection for Grubbs test
            col3, col4 = st.columns(2)
            with col3:
                alpha = st.selectbox(
                    "Significance level for statistical tests",
                    options=[0.05, 0.01, 0.001],
                    index=0,
                    format_func=lambda x: f"α = {x} ({'95%' if x==0.05 else '99%' if x==0.01 else '99.9%'} confidence)"
                )
            
            with col4:
                # Enable outlier detection preview
                enable_outlier_detection = st.checkbox(
                    "Enable outlier detection", 
                    value=True,
                    help="Uncheck to skip outlier detection and use all data points"
                )
            
            # Initialize variables
            selected_outliers = np.array([])
            method_choice = "None"
            exclude_outliers = False
            
            if enable_outlier_detection:
                # Prepare matched data for outlier detection preview
                merged_data = prepare_matched_data(df, material_type, selected_analyte, analyzer_1, analyzer_2)
                
                if len(merged_data) == 0:
                    st.warning(f"No matching samples found between {analyzer_1} and {analyzer_2} for {selected_analyte}")
                    return
                
                # Enhanced outlier detection with multiple methods
                selected_outliers, method_choice = perform_outlier_detection_with_options(
                    merged_data, selected_analyte, analyzer_1, analyzer_2, alpha
                )
                
                # Outlier exclusion settings
                if selected_outliers.any():
                    st.markdown("**Outlier Handling Options**")
                    exclude_outliers = st.checkbox(
                        "🚫 **Exclude detected outliers from analysis**", 
                        value=False,
                        help="When enabled, outliers will be completely removed from plots and calculations"
                    )
                    
                    if exclude_outliers:
                        outlier_sample_ids = merged_data['Sample ID'].iloc[selected_outliers].tolist()
                        st.error(f"⚠️ **{sum(selected_outliers)} outlier(s) will be EXCLUDED from all plots and calculations:**")
                        st.info("💡 The regression line will be calculated using only the remaining data points.")
                    else:
                        st.info("ℹ️ Outliers will be **highlighted in red** on plots but **included** in calculations.")
            else:
                st.info("ℹ️ Outlier detection is disabled. All data points will be included in the analysis.")

        # === ANALYSIS EXECUTION ===
        with st.expander("📈 Regression Analysis Results", expanded=True):
            # Prepare matched data
            merged_data = prepare_matched_data(df, material_type, selected_analyte, analyzer_1, analyzer_2)
            
            if len(merged_data) == 0:
                st.warning(f"No matching samples found between {analyzer_1} and {analyzer_2} for {selected_analyte}")
                return
            
            # st.info(f"Found {len(merged_data)} matching samples between analyzers.")
            
            # Perform analysis
            results, fig, merged_display, outlier_info = perform_analysis(
                df=df,
                material_type=material_type,
                analyte=selected_analyte,
                analyzer_1=analyzer_1,
                analyzer_2=analyzer_2,
                units=units,
                outlier_mask=selected_outliers if enable_outlier_detection else None,
                remove_outliers=exclude_outliers,
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
                
                # Create results dataframe for display
                results_df = pd.DataFrame([results]).T
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
                    slope_status = "😐 Acceptable"
                    slope_color = "orange"
                elif 0.75 <= slope_val <= 0.85:
                    slope_status = "⚠️ Acceptable but further investigation warran"
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
                    intercept_status = "⚠️ Acceptable"
                    intercept_color = "orange"
                else:
                    intercept_status = "❌ Poor"
                    intercept_color = "red"
                
                # R² interpretation
                if r2_val >= 0.95:
                    r2_status = "✅ Excellent"
                    r2_color = "green"
                elif r2_val >= 0.90:
                    r2_status = "⚠️ Good"
                    r2_color = "orange"
                elif r2_val >= 0.80:
                    r2_status = "⚠️ Acceptable"
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
                
                # Method agreement summary
                # st.markdown("### 🎯 Method Agreement Summary")
                # if slope_status == "✅ Excellent" and intercept_status == "✅ Excellent" and r2_status == "✅ Excellent":
                #     st.success("🎉 **Excellent agreement** between methods!")
                # elif "❌ Poor" in [slope_status, intercept_status, r2_status]:
                #     st.error("⚠️ **Poor agreement** - methods may not be interchangeable.")
                # else:
                #     st.warning("📊 **Moderate agreement** - review clinical acceptability.")

        # === DATA TABLES SECTION ===
        with st.expander("📋 Data Tables", expanded=False):
            st.markdown("### 🔢 Matched Sample Data")
            
            # Prepare display dataframe with outlier information
            display_df = merged_display.copy()
            display_df['Mean'] = (display_df[f'{selected_analyte}_1'] + display_df[f'{selected_analyte}_2']) / 2
            display_df['Difference'] = display_df[f'{selected_analyte}_1'] - display_df[f'{selected_analyte}_2']
            display_df['% Difference'] = (display_df['Difference'] / display_df['Mean'] * 100).round(2)
            
            # Add outlier status column if outlier detection is enabled
            if enable_outlier_detection and len(selected_outliers) > 0:
                display_df['Outlier Status'] = ['🔴 Outlier' if is_outlier else '✅ Normal' 
                                               for is_outlier in selected_outliers]
                
                # Add exclusion status if outliers are being excluded
                if exclude_outliers:
                    display_df['Analysis Status'] = ['❌ Excluded' if is_outlier else '✅ Included' 
                                                   for is_outlier in selected_outliers]
            
            # Reorder columns for better display
            column_order = ['Sample ID', f'{selected_analyte}_1', f'{selected_analyte}_2', 
                          'Mean', 'Difference', '% Difference']
            
            if enable_outlier_detection and len(selected_outliers) > 0:
                column_order.append('Outlier Status')
                if exclude_outliers:
                    column_order.append('Analysis Status')
            
            display_df = display_df[column_order]
            
            # Rename columns for clarity
            display_df = display_df.rename(columns={
                f'{selected_analyte}_1': f'{analyzer_1}',
                f'{selected_analyte}_2': f'{analyzer_2}'
            })
            
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            # Summary statistics
            st.markdown("### 📈 Summary Statistics")
            summary_stats = pd.DataFrame({
                'Statistic': ['Count', 'Mean', 'Std Dev', 'Min', 'Max', 'Median'],
                f'{analyzer_1}': [
                    len(display_df),
                    display_df[analyzer_1].mean().round(4),
                    display_df[analyzer_1].std().round(4),
                    display_df[analyzer_1].min().round(4),
                    display_df[analyzer_1].max().round(4),
                    display_df[analyzer_1].median().round(4)
                ],
                f'{analyzer_2}': [
                    len(display_df),
                    display_df[analyzer_2].mean().round(4),
                    display_df[analyzer_2].std().round(4),
                    display_df[analyzer_2].min().round(4),
                    display_df[analyzer_2].max().round(4),
                    display_df[analyzer_2].median().round(4)
                ]
            })
            
            st.dataframe(summary_stats, use_container_width=True, hide_index=True)

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
                    'Outliers_Detected': sum(selected_outliers) if enable_outlier_detection else 0,
                    'Outliers_Excluded': 'Yes' if exclude_outliers else 'No',
                    'Detection_Method': method_choice if enable_outlier_detection else 'None',
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
                - **Method:** {method_choice}
                - **Significance Level:** α = {alpha}
                - **Action:** {'Exclude from analysis' if exclude_outliers else 'Highlight but include'}
                
                **Method Descriptions:**
                - **Grubbs (Single):** Detects only the most extreme outlier
                - **Grubbs (Iterative):** Repeatedly applies Grubbs test until no more outliers found
                - **Limits Only (±1.96σ):** Simply flags points outside ±1.96 standard deviations
                - **Combined Method:** Uses iterative Grubbs + limit checking for comprehensive detection
                """)

# Call the function
def run():
    passing_bablok()

if __name__ == "__main__":
    run()