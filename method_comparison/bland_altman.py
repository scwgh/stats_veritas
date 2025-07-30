import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.graph_objects as go
from utils import apply_app_styling, units_list
import io

# Set up the page styling
apply_app_styling()

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

def enhanced_outlier_analysis(diffs, alpha=0.05):
    """
    Perform comprehensive outlier analysis and return detailed information.
    """
    n = len(diffs)
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs, ddof=1)
    
    # Apply different outlier detection methods
    methods = {
        'Grubbs (Single)': identify_outliers_beyond_limits(diffs, mean_diff, std_diff, alpha, 'grubbs_single'),
        'Grubbs (Iterative)': identify_outliers_beyond_limits(diffs, mean_diff, std_diff, alpha, 'grubbs_iterative'),
        'Limits Only (±1.96σ)': identify_outliers_beyond_limits(diffs, mean_diff, std_diff, alpha, 'limits_only'),
        'Combined Method': identify_outliers_beyond_limits(diffs, mean_diff, std_diff, alpha, 'combined')
    }
    
    # Calculate limits of agreement
    loa_upper = mean_diff + 1.96 * std_diff
    loa_lower = mean_diff - 1.96 * std_diff
    
    results = {
        'methods': methods,
        'limits': {'upper': loa_upper, 'lower': loa_lower, 'mean': mean_diff, 'std': std_diff},
        'n_total': n
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
    outlier_analysis = enhanced_outlier_analysis(diffs_initial.values, alpha=alpha)
    
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
        index=2,  # Default to 'Limits Only' which is more straightforward
        help="""
        - **Grubbs (Single)**: Detects only the most extreme outlier
        - **Grubbs (Iterative)**: Repeatedly applies Grubbs test until no more outliers found
        - **Limits Only (±1.96σ)**: Simply flags points outside ±1.96 standard deviations
        - **Combined Method**: Uses iterative Grubbs + limit checking for comprehensive detection
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
            
            outlier_details.append({
                'Sample ID': outlier_sample_ids[i],
                f'{analyzer_1}': round(outlier_vals1[i], 3),
                f'{analyzer_2}': round(outlier_vals2[i], 3),
                'Difference': round(outlier_diffs[i], 3),
                'Mean': round((outlier_vals1[i] + outlier_vals2[i]) / 2, 3),
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

def run():
    st.title("🥼 Bland-Altman Analysis")

    with st.expander("📘 What is Bland-Altman Analysis?"):
        st.markdown("""
        Bland-Altman analysis is a method for assessing the **agreement between two measurement methods** by plotting the **difference** against the **mean** of the two methods for each sample.
        \n Given a set of paired measurements (_x_ and _y_) where _i_ = 1 to _n_, the Bland-Altman plots calculate as:
        """)
        st.latex(r'\text{y axis} = {y}_i - {x}_i')
        st.latex(r'\text{x axis} = \frac{{y}_i + {x}_i}{2}')
        st.markdown("""
        This analysis is used to evaluate if two analyzers provide results that are **consistently close** enough for clinical or research purposes.
        \n The reference line for the mean gives an indication of the bias between the two methods. 
        \n The limits of agreement help assess whether the differences between two methods are practically significant. If the differences follow an approximately normal distribution, around 95% of the differences should fall within these limits. If the limits of agreement are considered clinically insignificant, the two measurement methods may be regarded as equivalent for practical purposes. However, especially with small sample sizes, these limits may not be reliable. In such cases, the confidence limits for the limits of agreement can provide an indication of the uncertainty. While these confidence limits are only approximate, they should be sufficient for most applications.
        \n Any results which are identified as outliers using the **Grubbs test** will be marked with a red square (🟥). 
        \n To exclude outliers from analysis for a given analyte, select the checkbox in the Analysis Settings section.""")

    with st.expander("📘 Instructions:"):
        st.markdown("""
        1. **Upload your CSV file** containing multi-analyte results.
        2. Your file must include these columns: `Material`, `Analyser`, `Sample ID`, `Batch ID`, `Lot Number` and at least one analyte.
        3. Configure your analysis settings in the "Analysis Settings" section below.
        4. Click **"Run Bland-Altman Analysis"** to generate plots and statistics for each analyte.
        """)

    with st.expander("📤 Upload Your CSV File", expanded=True):
        st.markdown("Upload a CSV containing your analyte data. Ensure it includes the following columns: `Material`, `Analyser`, and `Sample ID`.")
        uploaded_file = st.file_uploader("Choose a file to get started", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
    
        required_cols = ['Analyser', 'Material', 'Sample ID']
        if not all(col in df.columns for col in required_cols):
            st.error(f"Missing required columns: {', '.join(required_cols)}")
        else:
            # All settings in one expander
            with st.expander("⚙️ Analysis Settings", expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    material_type = st.selectbox("Select Material Type", df['Material'].unique())
                    analytes = df.columns[7:]
                    selected_analyte = st.selectbox("Select Analyte", analytes)
                    
                with col2:
                    analyzers = df["Analyser"].unique()
                    if len(analyzers) < 2:
                        st.warning("Need at least two analyzers in the dataset.")
                        return

                    analyzer_1 = st.selectbox("Select Reference Analyzer (Analyser 1)", analyzers, key="ref")
                    remaining_analyzers = [a for a in analyzers if a != analyzer_1]
                    analyzer_2 = st.selectbox("Select Test Analyzer (Analyser 2)", remaining_analyzers, key="test")

                units = st.selectbox(
                    "Select Units for Analytes",
                    options=units_list, 
                    index=0
                )
                
                # Enhanced Outlier Detection Settings
                st.markdown("**Enhanced Outlier Detection Settings**")
                
                # Add significance level selection for Grubbs test
                alpha = st.selectbox(
                    "Select significance level for statistical tests",
                    options=[0.05, 0.01, 0.001],
                    index=0,
                    format_func=lambda x: f"α = {x}"
                )
                
                # Prepare matched data for outlier detection preview
                merged_data = prepare_matched_data(df, material_type, selected_analyte, analyzer_1, analyzer_2)
                
                if len(merged_data) == 0:
                    st.warning(f"No matching samples found between {analyzer_1} and {analyzer_2} for {selected_analyte}")
                    return
                
                # Enhanced outlier detection with multiple methods
                selected_outliers, method_choice = perform_outlier_detection_with_options(
                    merged_data, selected_analyte, analyzer_1, analyzer_2, alpha
                )
                
                exclude_outliers = False
                if selected_outliers.any():
                    exclude_outliers = st.checkbox("Exclude detected outliers from analysis", value=False)
                    if exclude_outliers:
                        outlier_sample_ids = merged_data['Sample ID'].iloc[selected_outliers].tolist()
                        st.warning(f"⚠️ {sum(selected_outliers)} outlier(s) will be excluded from analysis using {method_choice}: {', '.join(map(str, outlier_sample_ids))}")

            # Run analysis button
            if st.button("🔬 Run Bland-Altman Analysis", type="primary"):
                bland_altman_analysis(df, material_type, selected_analyte, analyzer_1, analyzer_2, units, exclude_outliers, alpha, selected_outliers, method_choice)

def bland_altman_analysis(df, material_type, selected_analyte, analyzer_1, analyzer_2, units, exclude_outliers, alpha, selected_outliers=None, method_choice=None):
    """
    Perform Bland-Altman analysis and create plots
    """
    # Prepare matched data
    merged_data = prepare_matched_data(df, material_type, selected_analyte, analyzer_1, analyzer_2)
    
    if len(merged_data) == 0:
        st.warning(f"No matching samples found between {analyzer_1} and {analyzer_2} for {selected_analyte}")
        return
    
    # Extract values and calculate differences
    vals1 = merged_data[f'{selected_analyte}_1'].values  # Convert to numpy array
    vals2 = merged_data[f'{selected_analyte}_2'].values  # Convert to numpy array
    sample_ids = merged_data['Sample ID'].values  # Convert to numpy array
    
    # Calculate differences and means
    diffs = vals1 - vals2
    means = (vals1 + vals2) / 2
    percent_diffs = (diffs / np.where(means == 0, np.nan, means)) * 100
    
    # Use the pre-calculated outliers from the settings section
    if selected_outliers is None:
        # Fallback to original method if not provided
        selected_outliers = grubbs_test(diffs, alpha=alpha)
        method_choice = "Grubbs (Single)"
    
    is_outlier = selected_outliers
    
    # Apply outlier exclusion logic
    if exclude_outliers and is_outlier.any():
        # Create mask for filtering out outliers
        normal_mask = ~is_outlier
        
        # Filter out outliers from all data (both for analysis and plotting)
        vals1_final = vals1[normal_mask]
        vals2_final = vals2[normal_mask]
        diffs_final = diffs[normal_mask]
        means_final = means[normal_mask]
        percent_diffs_final = percent_diffs[normal_mask]
        sample_ids_final = sample_ids[normal_mask]
        
        analysis_note = f"Analysis performed on {len(vals1_final)} samples (excluded {sum(is_outlier)} outliers using {method_choice})"
        title_suffix = f" (Outliers Excluded - {method_choice})"
        
        # For reference, store original data for information purposes
        excluded_samples = sample_ids[is_outlier].tolist()
        
    else:
        # Use all data (outliers included in both analysis and plotting)
        vals1_final = vals1
        vals2_final = vals2
        diffs_final = diffs
        means_final = means
        percent_diffs_final = percent_diffs
        sample_ids_final = sample_ids
        
        analysis_note = f"Analysis performed on {len(vals1_final)} samples"
        title_suffix = ""
        
        # If outliers detected but not excluded, we'll show them highlighted
        if is_outlier.any():
            title_suffix = f" (Outliers Highlighted - {method_choice})"
    
    st.info(analysis_note)
    
    # Calculate statistics using final data (FIXED: Now using the correctly filtered data)
    N = len(vals1_final)
    mean_diff = np.mean(diffs_final)
    std_diff = np.std(diffs_final, ddof=1)
    loa_upper = mean_diff + 1.96 * std_diff
    loa_lower = mean_diff - 1.96 * std_diff
    
    # Confidence intervals for limits of agreement
    se = std_diff / np.sqrt(N)
    ci_range = 1.96 * se
    ci_upper_upper = loa_upper + ci_range
    ci_upper_lower = loa_upper - ci_range
    ci_lower_upper = loa_lower + ci_range
    ci_lower_lower = loa_lower - ci_range
    
    # Statistical tests using final data (FIXED: Now using the correctly filtered data)
    t_stat, p_val = stats.ttest_rel(vals1_final, vals2_final)
    slope, intercept, r_value, p_val_reg, _ = stats.linregress(vals1_final, vals2_final)
    
    # Calculate percentage statistics using final data (FIXED: Now using the correctly filtered data)
    mean_percent_diff = np.nanmean(percent_diffs_final)
    std_percent_diff = np.nanstd(percent_diffs_final, ddof=1)
    loa_upper_percent = mean_percent_diff + 1.96 * std_percent_diff
    loa_lower_percent = mean_percent_diff - 1.96 * std_percent_diff
    
    # Determine plot ranges based on final data (so axes update when outliers excluded)
    x_range = [means_final.min(), means_final.max()]
    x_range_reg = np.linspace(min(vals1_final.min(), vals2_final.min()), 
                             max(vals1_final.max(), vals2_final.max()), 100)
    y_fit = intercept + slope * x_range_reg
    
    # --- Plot 1: Numerical Differences ---
    fig1 = go.Figure()
    
    # Determine how to show points based on outlier settings
    if not exclude_outliers and is_outlier.any():
        # Show outliers highlighted (not excluded)
        normal_mask = ~is_outlier
        
        # Normal points
        if normal_mask.any():
            fig1.add_trace(go.Scatter(
                x=means[normal_mask],
                y=diffs[normal_mask],
                mode='markers',
                marker=dict(color='mediumblue', symbol='circle', size=8),
                name=f'Normal (N = {sum(normal_mask)})',
                hovertemplate='<b>Sample ID: %{text}</b><br>Mean: %{x:.3f}<br>Diff: %{y:.3f}<extra></extra>',
                text=sample_ids[normal_mask]
            ))
        
        # Outliers (highlighted but included)
        fig1.add_trace(go.Scatter(
            x=means[is_outlier],
            y=diffs[is_outlier],
            mode='markers',
            marker=dict(color='red', symbol='square', size=8),
            name=f'Outlier (N = {sum(is_outlier)})',
            hovertemplate='<b>Sample ID: %{text}</b><br>Mean: %{x:.3f}<br>Diff: %{y:.3f}<extra></extra>',
            text=sample_ids[is_outlier]
        ))
    else:
        # Show only final data (either no outliers or outliers excluded)
        fig1.add_trace(go.Scatter(
            x=means_final,
            y=diffs_final,
            mode='markers',
            marker=dict(color='mediumblue', symbol='circle', size=8),
            name=f'N = {N}',
            hovertemplate='<b>Sample ID: %{text}</b><br>Mean: %{x:.3f}<br>Diff: %{y:.3f}<extra></extra>',
            text=sample_ids_final
        ))
    
    # Add reference lines (based on final data)
    fig1.add_trace(go.Scatter(
        x=x_range,
        y=[mean_diff, mean_diff],
        mode='lines',
        line=dict(color='darkslateblue', dash='solid'),
        name=f"Mean Diff: {mean_diff:.3f}"
    ))
    fig1.add_trace(go.Scatter(
        x=x_range,
        y=[loa_upper, loa_upper],
        mode='lines',
        line=dict(color='slateblue', dash='dash'),
        name=f"+1.96 SD: {loa_upper:.3f}"
    ))
    fig1.add_trace(go.Scatter(
        x=x_range,
        y=[loa_lower, loa_lower],
        mode='lines',
        line=dict(color='slateblue', dash='dash'),
        name=f"-1.96 SD: {loa_lower:.3f}"
    ))
    
    fig1.update_layout(
        title=f"{selected_analyte} - Bland-Altman Plot (Numerical Difference){title_suffix}",
        xaxis_title=f"Mean of {analyzer_1} and {analyzer_2} ({units})",
        yaxis_title=f"Difference ({analyzer_1} - {analyzer_2}) ({units})",
        template="plotly_white"
    )
    st.plotly_chart(fig1, use_container_width=True)
    
    # --- Plot 2: Percentage Differences ---
    fig2 = go.Figure()
    
    # Same logic for percentage plot
    if not exclude_outliers and is_outlier.any():
        # Show outliers highlighted (not excluded)
        normal_mask = ~is_outlier
        
        # Normal points
        if normal_mask.any():
            fig2.add_trace(go.Scatter(
                x=means[normal_mask],
                y=percent_diffs[normal_mask],
                mode='markers',
                marker=dict(color='mediumblue', symbol='circle', size=8),
                name=f'Normal (N = {sum(normal_mask)})',
                hovertemplate='<b>Sample ID: %{text}</b><br>Mean: %{x:.3f}<br>% Diff: %{y:.1f}%<extra></extra>',
                text=sample_ids[normal_mask]
            ))
        
        # Outliers (highlighted but included)
        fig2.add_trace(go.Scatter(
            x=means[is_outlier],
            y=percent_diffs[is_outlier],
            mode='markers',
            marker=dict(color='red', symbol='square', size=8),
            name=f'Outlier (N = {sum(is_outlier)})',
            hovertemplate='<b>Sample ID: %{text}</b><br>Mean: %{x:.3f}<br>% Diff: %{y:.1f}%<extra></extra>',
            text=sample_ids[is_outlier]
        ))
    else:
        # Show only final data
        fig2.add_trace(go.Scatter(
            x=means_final,
            y=percent_diffs_final,
            mode='markers',
            marker=dict(color='mediumblue', symbol='circle', size=8),
            name=f'N = {N}',
            hovertemplate='<b>Sample ID: %{text}</b><br>Mean: %{x:.3f}<br>% Diff: %{y:.1f}%<extra></extra>',
            text=sample_ids_final
        ))
    
    # Add reference lines (based on final data)
    fig2.add_trace(go.Scatter(
        x=x_range,
        y=[mean_percent_diff, mean_percent_diff],
        mode='lines',
        line=dict(color='darkslateblue', dash='solid'),
        name=f"Mean % Diff: {mean_percent_diff:.2f}%"
    ))
    fig2.add_trace(go.Scatter(
        x=x_range,
        y=[loa_upper_percent, loa_upper_percent],
        mode='lines',
        line=dict(color='slateblue', dash='dash'),
        name=f"+1.96 SD: {loa_upper_percent:.2f}%"
    ))
    fig2.add_trace(go.Scatter(
        x=x_range,
        y=[loa_lower_percent, loa_lower_percent],
        mode='lines',
        line=dict(color='slateblue', dash='dash'),
        name=f"-1.96 SD: {loa_lower_percent:.2f}%"
    ))
    
    fig2.update_layout(
        title=f"{selected_analyte} - Bland-Altman Plot (Percentage Difference){title_suffix}",
        xaxis_title=f"Mean of {analyzer_1} and {analyzer_2} ({units})",
        yaxis_title=f"Percentage Difference (%)",
        template="plotly_white"
    )
    st.plotly_chart(fig2, use_container_width=True)
    
# --- Plot 3: Regression Plot ---
    fig3 = go.Figure()
    
    # Same logic for regression plot
    if not exclude_outliers and is_outlier.any():
        # Show outliers highlighted (not excluded)
        normal_mask = ~is_outlier
        
        # Normal points
        if normal_mask.any():
            fig3.add_trace(go.Scatter(
                x=vals1[normal_mask],
                y=vals2[normal_mask],
                mode='markers',
                marker=dict(color='mediumblue', symbol='circle', size=8),
                name=f'Normal (N = {sum(normal_mask)})',
                text=sample_ids[normal_mask],
                hovertemplate='<b>Sample ID: %{text}</b><br>%{x:.3f} vs %{y:.3f}<extra></extra>'
            ))
        
        # Outliers (highlighted but included)
        fig3.add_trace(go.Scatter(
            x=vals1[is_outlier],
            y=vals2[is_outlier],
            mode='markers',
            marker=dict(color='red', symbol='square', size=8),
            name=f'Outlier (N = {sum(is_outlier)})',
            text=sample_ids[is_outlier],
            hovertemplate='<b>Sample ID: %{text}</b><br>%{x:.3f} vs %{y:.3f}<extra></extra>'
        ))
    else:
        # Show only final data
        fig3.add_trace(go.Scatter(
            x=vals1_final,
            y=vals2_final,
            mode='markers',
            marker=dict(color='mediumblue', symbol='circle', size=8),
            name=f'N = {N}',
            text=sample_ids_final,
            hovertemplate='<b>Sample ID: %{text}</b><br>%{x:.3f} vs %{y:.3f}<extra></extra>'
        ))
    
    # Calculate prediction intervals (wider than confidence intervals)
    # Use final data for CI calculations
    x_mean = np.mean(vals1_final)
    sxx = np.sum((vals1_final - x_mean) ** 2)
    mse = np.sum((vals2_final - (intercept + slope * vals1_final)) ** 2) / (N - 2)  # Mean squared error
    
    # Calculate standard error for prediction intervals (includes individual point variability)
    se_pred = np.sqrt(mse * (1 + 1/N + (x_range_reg - x_mean)**2 / sxx))
    
    # 95% prediction interval (t-distribution with N-2 degrees of freedom)
    t_val = stats.t.ppf(0.975, df=N-2)  # 95% PI
    ci_upper = y_fit + t_val * se_pred
    ci_lower = y_fit - t_val * se_pred
    
    # Add confidence interval as shaded area
    fig3.add_trace(go.Scatter(
        x=np.concatenate([x_range_reg, x_range_reg[::-1]]),
        y=np.concatenate([ci_upper, ci_lower[::-1]]),
        fill='toself',
        fillcolor='rgba(220, 20, 60, 0.2)',  # Light crimson with transparency
        line=dict(color='rgba(255,255,255,0)'),  # Invisible line
        name='95% Prediction Interval',
        hoverinfo='skip',
        showlegend=True
    ))
    
    # Add regression line (based on final data)
    fig3.add_trace(go.Scatter(
        x=x_range_reg,
        y=y_fit,
        mode='lines',
        line=dict(color='crimson', dash='solid', width=2),
        name=f'Regression Line<br>y = {slope:.3f}x + {intercept:.3f}<br>R² = {r_value**2:.3f}'
    ))
    
    # Add line of identity
    fig3.add_trace(go.Scatter(
        x=x_range_reg,
        y=x_range_reg,
        mode='lines',
        line=dict(color='gray', dash='dot', width=1),
        name='Line of Identity (y = x)'
    ))
    
    fig3.update_layout(
        title=f"{selected_analyte} - Regression Plot with Prediction Intervals{title_suffix}",
        xaxis_title=f"{analyzer_1} ({units})",
        yaxis_title=f"{analyzer_2} ({units})",
        template="plotly_white"
    )
    st.plotly_chart(fig3, use_container_width=True)
    
    # --- Plot 4: Bar Plot of Differences ---
    diffs_df = pd.DataFrame({
        'Sample ID': sample_ids_final,
        'Difference': diffs_final
    })
    diffs_df_sorted = diffs_df.sort_values('Difference').reset_index(drop=True)
    
    fig4 = go.Figure()
    fig4.add_trace(go.Bar(
        x=diffs_df_sorted.index,
        y=diffs_df_sorted['Difference'],
        marker_color='indianred',
        text=diffs_df_sorted['Sample ID'],
        hovertemplate='<b>Sample ID: %{text}</b><br>Difference: %{y:.3f}<extra></extra>',
        name=f'N = {N}'
    ))
    
    # Add reference lines (based on final data)
    fig4.add_hline(y=mean_diff, line=dict(color='black', dash='solid'), 
                   annotation_text=f'Mean Diff: {mean_diff:.3f}', annotation_position='top left')
    fig4.add_hline(y=loa_upper, line=dict(color='slateblue', dash='dash'), 
                   annotation_text=f'+1.96 SD: {loa_upper:.3f}', annotation_position='top right')
    fig4.add_hline(y=loa_lower, line=dict(color='slateblue', dash='dash'), 
                   annotation_text=f'-1.96 SD: {loa_lower:.3f}', annotation_position='bottom right')
    
    fig4.update_layout(
        title=f"{selected_analyte} - Bar Plot of Differences{title_suffix}",
        xaxis_title="Sample Index (Sorted by Difference)",
        yaxis_title=f"Difference ({analyzer_1} - {analyzer_2}) ({units})",
        template="plotly_white"
    )
    st.plotly_chart(fig4, use_container_width=True)
    
    # Add outlier information if present
    if is_outlier.any():
        st.markdown("### 🔍 Outlier Information")
        outlier_status = "Excluded from analysis and plots" if exclude_outliers else "Highlighted in plots but included in analysis"
        st.info(f"**{sum(is_outlier)} outlier(s) detected** using Grubbs test (α = {alpha}). Status: {outlier_status}")
        
        if exclude_outliers:
            # Enhanced explanation section for excluded outliers
            with st.expander("📖 Why Were These Outliers Excluded?", expanded=True):
                st.markdown(f"""
                **{sum(is_outlier)} sample(s) were excluded from this analysis** based on the Grubbs test for outliers at significance level α = {alpha}.
                
                #### What is the Grubbs Test?
                The Grubbs test (also known as the extreme studentized deviate test) is a statistical method used to detect outliers in a univariate dataset. It tests whether the most extreme value in the dataset is significantly different from the rest of the data.
                
                #### Why Exclude Outliers in Bland-Altman Analysis?
                - **Outliers can skew results**: Extreme values can artificially widen the limits of agreement, making two methods appear less comparable than they actually are
                - **May represent measurement errors**: Outliers could indicate transcription errors, instrument malfunctions, or sample handling issues
                - **Improve clinical relevance**: Excluding clear outliers can provide a more realistic assessment of typical method agreement
                - **Statistical robustness**: Limits of agreement are more reliable when calculated from normally distributed differences
                
                #### What Does This Mean for Your Analysis?
                - The **limits of agreement** are now calculated based on {N} samples instead of {len(vals1)} samples
                - Statistical measures (mean difference, standard deviation, correlation) reflect the **typical performance** between methods
                - Results are more representative of **routine analytical conditions**
                - The excluded samples should be **investigated separately** to determine the cause of the extreme differences
                
                #### Excluded Sample Details:
                """)
                
                # Show excluded samples with detailed information
                excluded_indices = np.where(is_outlier)[0]
                excluded_details = []
                for idx in excluded_indices:
                    sample_id = sample_ids[idx]
                    val1 = vals1[idx]
                    val2 = vals2[idx]
                    diff = diffs[idx]
                    mean_val = means[idx]
                    
                    # Calculate how many standard deviations from mean
                    z_score = abs(diff - np.mean(diffs)) / np.std(diffs, ddof=1)
                    
                    excluded_details.append({
                        'Sample ID': sample_id,
                        f'{analyzer_1}': round(val1, 3),
                        f'{analyzer_2}': round(val2, 3),
                        'Difference': round(diff, 3),
                        'Mean': round(mean_val, 3),
                        'Z-Score': round(z_score, 2),
                        'Deviation': f"{z_score:.1f}σ from mean"
                    })
                
                excluded_df = pd.DataFrame(excluded_details)
                st.dataframe(excluded_df, use_container_width=True, hide_index=True)
                
                st.markdown(f"""
                #### Recommendations:
                
                1. **Investigate excluded samples**: Review the measurement process for samples {', '.join(map(str, excluded_samples))}
                2. **Check for errors**: Verify data entry, sample handling, and instrument calibration
                3. **Consider clinical context**: Determine if extreme differences have clinical significance
                4. **Document decisions**: Record why outliers were excluded for audit purposes
                5. **Consider replication**: If possible, re-analyze excluded samples to confirm results
                
                #### Statistical Impact:
                - **Before exclusion**: N = {len(vals1)}, Mean difference = {np.mean(diffs):.3f}, SD = {np.std(diffs, ddof=1):.3f}
                - **After exclusion**: N = {N}, Mean difference = {mean_diff:.3f}, SD = {std_diff:.3f}
                - **Change in precision**: {((np.std(diffs, ddof=1) - std_diff) / np.std(diffs, ddof=1) * 100):.1f}% reduction in standard deviation
                
                ---
                *Note: Outlier exclusion should be based on scientific rationale, not just statistical convenience. Always document and justify exclusion decisions.*
                """)
            
            st.warning(f"⚠️ Analysis performed with {sum(is_outlier)} excluded sample(s): {', '.join(map(str, excluded_samples))}")
        else:
            # Brief explanation when outliers are highlighted but not excluded
            st.info(f"""
            **Outliers are highlighted in red squares (🟥) but remain included in the analysis.**
            
            These {sum(is_outlier)} sample(s) show extreme differences that may warrant investigation:
            {', '.join(map(str, sample_ids[is_outlier]))}
            
            Consider checking the "Exclude outliers from analysis" option if these represent measurement errors or non-representative conditions.
            """)
    else:
        st.success("✅ No outliers detected using Grubbs test.")

    # --- Summary Statistics ---
    st.markdown("### 📊 Statistical Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Number of Samples", N)
        st.metric("Mean Difference", f"{mean_diff:.3f} {units}")
        st.metric("SD of Differences", f"{std_diff:.3f} {units}")
    
    with col2:
        st.metric("Lower LoA", f"{loa_lower:.3f} {units}")
        st.metric("Upper LoA", f"{loa_upper:.3f} {units}")
        st.metric("p-value (paired t-test)", f"{p_val:.3f}")
    
    with col3:
        st.metric("Correlation (R²)", f"{r_value**2:.3f}")
        st.metric("Slope", f"{slope:.3f}")
        st.metric("Intercept", f"{intercept:.3f}")
    
    # Add outlier information if present
    if is_outlier.any():
        st.markdown("### 🔍 Outlier Information")
        outlier_status = "Excluded from analysis and plots" if exclude_outliers else "Highlighted in plots but included in analysis"
        st.info(f"**{sum(is_outlier)} outlier(s) detected** using Grubbs test (α = {alpha}). Status: {outlier_status}")
        
        if exclude_outliers:
            st.warning(f"Excluded samples: {', '.join(map(str, excluded_samples))}")
    
    # --- Full Summary Table: All Materials × All Analytes ---
    st.markdown("### 📋 Complete Analysis Summary")
    
    summary_table = []
    
    for material in df['Material'].unique():
        analytes = df.columns[7:]
        
        for analyte in analytes:
            try:
                # Get analyzers for this material
                material_data = df[df['Material'] == material]
                analyzers = material_data['Analyser'].unique()
                
                if len(analyzers) < 2:
                    continue
                
                # Use first two analyzers
                analyzer1, analyzer2 = analyzers[:2]
                
                # Prepare matched data
                matched_data = prepare_matched_data(df, material, analyte, analyzer1, analyzer2)
                
                if len(matched_data) == 0:
                    continue
                
                # Calculate statistics
                vals1 = matched_data[f'{analyte}_1']
                vals2 = matched_data[f'{analyte}_2']
                diffs = vals1 - vals2
                
                # Apply Grubbs test for outlier detection
                is_outlier_summary = identify_outliers_beyond_limits(diffs.values, np.mean(diffs), np.std(diffs, ddof=1), alpha, 'limits_only')
                n_outliers = sum(is_outlier_summary)
                
                # Calculate stats with and without outliers
                if exclude_outliers and is_outlier_summary.any():
                    normal_mask = ~is_outlier_summary
                    vals1_clean = vals1.values[normal_mask]
                    vals2_clean = vals2.values[normal_mask]
                    diffs_clean = vals1_clean - vals2_clean
                    
                    mean_diff = np.mean(diffs_clean)
                    std_diff = np.std(diffs_clean, ddof=1)
                    n_samples = len(diffs_clean)
                    _, p_val = stats.ttest_rel(vals1_clean, vals2_clean)
                else:
                    mean_diff = np.mean(diffs)
                    std_diff = np.std(diffs, ddof=1)
                    n_samples = len(diffs)
                    _, p_val = stats.ttest_rel(vals1, vals2)
                
                loa_upper = mean_diff + 1.96 * std_diff
                loa_lower = mean_diff - 1.96 * std_diff
                
                # Calculate correlation
                if exclude_outliers and is_outlier_summary.any():
                    slope, intercept, r_value, _, _ = stats.linregress(vals1_clean, vals2_clean)
                else:
                    slope, intercept, r_value, _, _ = stats.linregress(vals1, vals2)
                
                summary_table.append({
                    'Material': material,
                    'Analyte': analyte,
                    'Analyzer 1': analyzer1,
                    'Analyzer 2': analyzer2,
                    'N Samples': n_samples,
                    'N Outliers': n_outliers,
                    'Mean Difference': round(mean_diff, 3),
                    'SD of Differences': round(std_diff, 3),
                    'LoA Lower': round(loa_lower, 3),
                    'LoA Upper': round(loa_upper, 3),
                    'R²': round(r_value**2, 3),
                    'p-value': round(p_val, 3)
                })
                
            except Exception as e:
                st.warning(f"⚠️ Could not process {material} - {analyte}: {str(e)}")
                continue
    
    if summary_table:
        summary_df = pd.DataFrame(summary_table)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        # Add download button for summary table
        csv_buffer = io.StringIO()
        summary_df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()
        
        st.download_button(
            label="📥 Download Summary Table as CSV",
            data=csv_data,
            file_name=f"bland_altman_summary_{material_type}_{selected_analyte}.csv",
            mime="text/csv"
        )
        
        # Display key insights
        st.markdown("### 🔍 Key Insights")
        
        # Calculate some summary statistics across all combinations
        total_combinations = len(summary_df)
        significant_differences = len(summary_df[summary_df['p-value'] < 0.05])
        high_correlation = len(summary_df[summary_df['R²'] > 0.9])
        total_outliers = summary_df['N Outliers'].sum()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Combinations", total_combinations)
        
        with col2:
            st.metric("Significant Differences", f"{significant_differences}/{total_combinations}")
        
        with col3:
            st.metric("High Correlation (R² > 0.9)", f"{high_correlation}/{total_combinations}")
        
        with col4:
            st.metric("Total Outliers Detected", total_outliers)
        
        # Show materials/analytes with concerning results
        concerning_results = summary_df[
            (summary_df['p-value'] < 0.05) | 
            (summary_df['R²'] < 0.8) | 
            (summary_df['N Outliers'] > 0)
        ]
        
        if len(concerning_results) > 0:
            st.markdown("#### ⚠️ Attention Required")
            st.markdown("*The following combinations show significant differences, low correlation, or outliers:*")
            
            # Create a more focused display
            concerning_display = concerning_results[['Material', 'Analyte', 'Analyzer 1', 'Analyzer 2', 
                                                   'p-value', 'R²', 'N Outliers']].copy()
            
            # Add interpretation column
            def interpret_concern(row):
                concerns = []
                if row['R²'] < 0.8:
                    concerns.append("Correlation score <0.8 - further investigation suggested.")
                if row['N Outliers'] > 0:
                    concerns.append(f"{row['N Outliers']} outlier(s)")
                return "; ".join(concerns)
            
            concerning_display['Concerns'] = concerning_results.apply(interpret_concern, axis=1)
            st.dataframe(concerning_display, use_container_width=True, hide_index=True)
        else:
            st.success("✅ All analyzer combinations show good agreement!")
    
    else:
        st.warning("No valid combinations found for analysis. Please check your data format and ensure there are matching samples between analyzers.")

if __name__ == "__main__":
    run()