import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.odr import ODR, RealData, Model
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils import apply_app_styling, units_list

apply_app_styling()

# === Utility Functions ===
def grubbs_test(values, alpha=0.05):
    """
    Original Grubbs test for outlier detection (single outlier detection)
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
    """
    values = pd.Series(values).copy()
    outlier_indices = []
    original_indices = values.index.tolist()
    
    for iteration in range(max_iterations):
        n = len(values)
        if n < 3:
            break
            
        mean_val = values.mean()
        std_val = values.std(ddof=1)
        
        if std_val == 0:
            break
            
        abs_diff = abs(values - mean_val)
        max_diff_idx = abs_diff.idxmax()
        G = abs_diff[max_diff_idx] / std_val
        
        t_crit = stats.t.ppf(1 - alpha / (2 * n), df=n - 2)
        G_crit = ((n - 1) / np.sqrt(n)) * np.sqrt(t_crit**2 / (n - 2 + t_crit**2))
        
        if G > G_crit:
            outlier_indices.append(max_diff_idx)
            values = values.drop(max_diff_idx)
        else:
            break
    
    is_outlier = np.array([idx in outlier_indices for idx in original_indices])
    return is_outlier

def identify_outliers(differences, alpha=0.05, method='grubbs_iterative', x_vals=None, y_vals=None):
    """
    Identify outliers using multiple methods and return a boolean array.
    """
    if method == 'grubbs_iterative':
        return grubbs_test_iterative(differences, alpha)
    
    elif method == 'grubbs_single':
        return grubbs_test(differences, alpha)
    
    elif method == 'limits_only':
        mean_diff = np.mean(differences)
        std_diff = np.std(differences, ddof=1)
        loa_upper = mean_diff + 1.96 * std_diff
        loa_lower = mean_diff - 1.96 * std_diff
        return (differences > loa_upper) | (differences < loa_lower)
    
    elif method == 'large_percent_diff':
        if x_vals is None or y_vals is None:
            raise ValueError("x_vals and y_vals are required for 'large_percent_diff' method.")
        means = (x_vals + y_vals) / 2
        percent_diffs = np.abs(differences / means) * 100
        return percent_diffs > 50
    
    elif method == 'comprehensive':
        grubbs_iterative_outliers = grubbs_test_iterative(differences, alpha)
        
        if x_vals is None or y_vals is None:
            st.warning("x_vals and y_vals are required for 'Comprehensive' method. Using Grubbs only.")
            return grubbs_iterative_outliers
            
        means = (x_vals + y_vals) / 2
        # Avoid division by zero
        safe_means = np.where(means == 0, 1e-10, means)
        percent_diffs = np.abs(differences / safe_means) * 100
        large_percent_outliers = percent_diffs > 50
        
        return grubbs_iterative_outliers | large_percent_outliers

    else:
        raise ValueError("Invalid outlier detection method.")

def get_outlier_analysis_results(merged_data, selected_analyte, analyzer_1, analyzer_2, alpha, chosen_method):
    """
    Performs outlier detection and returns outlier flags and details.
    """
    vals1 = merged_data[f'{selected_analyte}_1']
    vals2 = merged_data[f'{selected_analyte}_2']
    diffs = vals1 - vals2

    is_outlier = identify_outliers(diffs.values, alpha, chosen_method, x_vals=vals1.values, y_vals=vals2.values)
    
    outlier_details_df = pd.DataFrame()
    if np.sum(is_outlier) > 0:
        outlier_indices = np.where(is_outlier)[0]
        outlier_sample_ids = merged_data['Sample ID'].iloc[outlier_indices].tolist()
        
        outlier_details = []
        for i, idx in enumerate(outlier_indices):
            val1 = vals1.iloc[idx]
            val2 = vals2.iloc[idx]
            diff = diffs.iloc[idx]
            mean_val = (val1 + val2) / 2
            percent_diff = abs(diff / mean_val) * 100 if mean_val != 0 else 0
            
            initial_mean_diff = np.mean(diffs)
            initial_std_diff = np.std(diffs, ddof=1)
            z_score = abs(diff - initial_mean_diff) / initial_std_diff if initial_std_diff != 0 else np.nan

            outlier_details.append({
                'Sample ID': outlier_sample_ids[i],
                f'{analyzer_1}': round(val1, 3),
                f'{analyzer_2}': round(val2, 3),
                'Difference': round(diff, 3),
                'Mean': round(mean_val, 3),
                '% Difference': round(percent_diff, 1),
                'Z-Score': round(z_score, 2) if not np.isnan(z_score) else "N/A"
            })
        
        outlier_details_df = pd.DataFrame(outlier_details)
        
    return is_outlier, outlier_details_df

def perform_deming_regression(x, y, delta=1.0):
    """
    Perform Deming regression and return results.
    """
    def linear(B, x): 
        return B[0] * x + B[1]
    
    model = Model(linear)
    odr_data = RealData(x, y, sx=np.std(x)*np.sqrt(delta), sy=np.std(y))
    odr = ODR(odr_data, model, beta0=[1, 0])
    output = odr.run()
                
    slope, intercept = output.beta
    se_slope, se_intercept = output.sd_beta
    
    # Calculate R-squared
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    # Get covariance matrix for confidence intervals
    cov_matrix = output.cov_beta
    
    return slope, intercept, se_slope, se_intercept, r_squared, cov_matrix

def calculate_confidence_intervals(x_range, slope, intercept, cov_matrix, x_data, confidence_level=0.95):
    """
    Calculate confidence intervals for the regression line at each point.
    """
    n = len(x_data)
    alpha = 1 - confidence_level
    t_val = stats.t.ppf(1 - alpha/2, n - 2)
    
    # Variance of predicted y for each x value
    x_mean = np.mean(x_data)
    
    # For each point on the regression line
    y_pred = slope * x_range + intercept
    
    # Calculate standard error of prediction at each x
    se_pred = np.zeros_like(x_range)
    
    for i, x_val in enumerate(x_range):
        # Jacobian matrix [x, 1] for the linear model y = slope*x + intercept
        jacobian = np.array([x_val, 1])
        
        # Variance of prediction: J^T * Cov * J
        var_pred = np.dot(jacobian, np.dot(cov_matrix, jacobian))
        se_pred[i] = np.sqrt(var_pred)
    
    # Confidence intervals
    ci_lower = y_pred - t_val * se_pred
    ci_upper = y_pred + t_val * se_pred
    
    return ci_lower, ci_upper

def run():
    apply_app_styling()

    st.title("⚖️ Deming Regression Analysis")
    
    with st.expander("📘 What is Deming regression?"):
        st.markdown("""
        **Deming regression** is used when **both X and Y variables have measurement error**,
        which is common in method comparison studies.
        - **Slope**: Indicates proportional bias (ideal value is 1).
        - **Intercept**: Indicates constant bias (ideal value is 0).
        - **R²**: Represents the strength of the linear relationship between the two methods.
        """)

    # File upload section 
    with st.expander("📤 Upload CSV File", expanded=True):
        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                # Check required columns
                required_cols = ['Analyser', 'Material', 'Sample ID']
                missing_cols = [col for col in required_cols if col not in df.columns]
                
                if missing_cols:
                    st.error(f"❌ Missing required columns: **{', '.join(missing_cols)}**")
                    st.stop()
                
                st.success(f"✅ File uploaded successfully!")
                
                # Display data preview
                st.markdown("#### Data Preview")
                st.dataframe(df.head(), use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")
                st.stop()
        else:
            st.info("Please upload a CSV file to begin analysis.")
            st.stop()

    # Analysis Settings 
    with st.expander("⚙️ Analysis Settings", expanded=True):
        st.markdown("### ⚙️ Analysis Settings")

        analyzers = df['Analyser'].dropna().unique()
        if len(analyzers) < 2:
            st.error("❌ Need at least two different analyzers for comparison.")
            st.stop()

        materials = df['Material'].dropna().unique()
        metadata_cols = ['Date', 'Test', 'Material', 'Analyser', 'Sample ID', 'Batch ID', 'Lot Number']
        analytes = [col for col in df.columns if col not in metadata_cols]

        if not analytes:
            st.error("❌ No analyte columns found.")
            st.stop()

        col1, col2 = st.columns(2)

        with col1:
            analyzer_1 = st.selectbox("Select Analyzer 1 (X-axis)", analyzers, index=0)
            selected_material = st.selectbox("Select Material Type", materials)
            selected_analyte = st.selectbox("Select Analyte for Analysis", analytes)
            confidence_level = st.slider("Confidence Level (%)", min_value=80, max_value=99, value=95)

        with col2:
            analyzer_2 = st.selectbox("Select Analyzer 2 (Y-axis)", analyzers, index=1 if len(analyzers) > 1 else 0)
            selected_units = st.selectbox("Select Units", units_list)
            st.markdown("### 🔍 Outlier Detection")
            use_outlier_detection = st.checkbox("Enable Outlier Detection", value=True)

            if use_outlier_detection:
                selected_method_name = st.selectbox(
                    "Outlier Detection Method:",
                    ["Grubbs (Single)", "Grubbs (Iterative)", "Limits Only (±1.96σ)",
                    "Large % Difference (>50%)", "Comprehensive (Grubbs + Large % Diff)"],
                    index=4
                )
                exclude_outliers = st.checkbox(
                    "❌ Exclude detected outliers from regression",
                    value=False,
                    help="Remove outliers from both visualization and calculation"
                )
            else:
                selected_method_name = None
                exclude_outliers = False

        if analyzer_1 == analyzer_2:
            st.warning("⚠ Please select two different analyzers.")
            st.stop()

        alpha = 1 - confidence_level / 100
        outlier_methods = {
            "Grubbs (Single)": 'grubbs_single',
            "Grubbs (Iterative)": 'grubbs_iterative',
            "Limits Only (±1.96σ)": 'limits_only',
            "Large % Difference (>50%)": 'large_percent_diff',
            "Comprehensive (Grubbs + Large % Diff)": 'comprehensive'
        }
        chosen_method = outlier_methods[selected_method_name] if selected_method_name else None

    # ===== MISSING DATA PROCESSING SECTION - THIS WAS THE PROBLEM =====
    # Filter data for selected material and analyte
    filtered_df = df[df['Material'] == selected_material].copy()
    
    # Check if selected analyte has data
    if selected_analyte not in filtered_df.columns:
        st.error(f"❌ Analyte '{selected_analyte}' not found in the data.")
        st.stop()
    
    # Remove rows where the selected analyte is null
    filtered_df = filtered_df.dropna(subset=[selected_analyte])
    
    if len(filtered_df) == 0:
        st.error(f"❌ No data found for material '{selected_material}' and analyte '{selected_analyte}'.")
        st.stop()
    
    # Create pivot table to get paired data
    try:
        pivot = filtered_df.pivot_table(
            index='Sample ID', 
            columns='Analyser', 
            values=selected_analyte, 
            aggfunc='mean'
        )
        
        # Filter for the two selected analyzers
        if analyzer_1 not in pivot.columns or analyzer_2 not in pivot.columns:
            st.error(f"❌ One or both analyzers not found in data for the selected material and analyte.")
            st.stop()
        
        pivot = pivot[[analyzer_1, analyzer_2]].dropna()
        
        if len(pivot) == 0:
            st.error(f"❌ No paired data found between {analyzer_1} and {analyzer_2}.")
            st.stop()
            
    except Exception as e:
        st.error(f"❌ Error creating pivot table: {str(e)}")
        st.stop()
    
    # Extract data
    original_x = pivot[analyzer_1].values
    original_y = pivot[analyzer_2].values
    original_sample_ids = pivot.index.tolist()
    n_original = len(original_x)

    st.success(f"✅ Found {n_original} paired data points.")

    # Initialize data for regression
    x = original_x.copy()
    y = original_y.copy()
    sample_ids = original_sample_ids.copy()
    outlier_flags = np.array([False] * n_original)
    outlier_details_df = pd.DataFrame()

    # Perform outlier detection if enabled
    if use_outlier_detection:
        # Create temporary dataframe for outlier analysis
        temp_df = pd.DataFrame({
            'Sample ID': original_sample_ids,
            f'{selected_analyte}_1': original_x,
            f'{selected_analyte}_2': original_y
        })
        
        outlier_flags, outlier_details_df = get_outlier_analysis_results(
            temp_df, selected_analyte, analyzer_1, analyzer_2, alpha, chosen_method
        )
        
        if np.sum(outlier_flags) > 0:
            st.markdown("### 🚨 Outliers Detected")
            st.dataframe(outlier_details_df, use_container_width=True)
            
            if exclude_outliers:
                # Remove outliers from regression data
                non_outlier_mask = ~outlier_flags
                x = original_x[non_outlier_mask]
                y = original_y[non_outlier_mask]
                sample_ids = [original_sample_ids[i] for i in range(len(original_sample_ids)) if non_outlier_mask[i]]
                st.info(f"ℹ️ Excluded {np.sum(outlier_flags)} outliers from regression analysis.")

    # Perform Deming Regression
    st.markdown(f"### 📊 Deming Regression Results")
    
    try:
        slope, intercept, se_slope, se_intercept, r_squared, cov_matrix = perform_deming_regression(x, y)
        
        # Statistical tests
        dof = len(x) - 2
        if dof > 0 and se_slope > 0:
            t_val = stats.t.ppf(1 - alpha / 2, dof)
            
            # Test if slope significantly different from 1 (two-tailed test)
            t_stat_slope = (slope - 1) / se_slope
            p_val_slope = 2 * (1 - stats.t.cdf(abs(t_stat_slope), dof))
            
            # Test if intercept significantly different from 0 (two-tailed test)  
            t_stat_intercept = intercept / se_intercept if se_intercept > 0 else np.nan
            p_val_intercept = 2 * (1 - stats.t.cdf(abs(t_stat_intercept), dof)) if not np.isnan(t_stat_intercept) else np.nan
            
            # Confidence intervals
            ci_slope_lower = slope - t_val * se_slope
            ci_slope_upper = slope + t_val * se_slope
            ci_intercept_lower = intercept - t_val * se_intercept
            ci_intercept_upper = intercept + t_val * se_intercept
            
        else:
            p_val_slope = np.nan
            p_val_intercept = np.nan
            ci_slope_lower = np.nan
            ci_slope_upper = np.nan
            ci_intercept_lower = np.nan
            ci_intercept_upper = np.nan

        # Create plots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=['Deming Regression with Confidence Intervals', 'Residuals Plot'],
            vertical_spacing=0.25
        )

        # Plot 1: Regression plot
        fig.add_trace(
            go.Scatter(
                x=x, y=y,
                mode='markers',
                marker=dict(color='dodgerblue', size=8),
                line=dict(color='black', width=1),
                name='Data',
                text=sample_ids,
                hovertemplate='<b>%{text}</b><br>X: %{x:.3f}<br>Y: %{y:.3f}<extra></extra>'
            ),
            row=1, col=1
        )

        # Add outliers if detected and not excluded
        if use_outlier_detection and np.sum(outlier_flags) > 0 and not exclude_outliers:
            outlier_indices = np.where(outlier_flags)[0]
            fig.add_trace(
                go.Scatter(
                    x=original_x[outlier_indices],
                    y=original_y[outlier_indices],
                    mode='markers',
                    marker=dict(color='red', size=10, symbol='square'),
                    name='Outliers',
                    text=[original_sample_ids[i] for i in outlier_indices],
                    hovertemplate='<b>%{text}</b><br>X: %{x:.3f}<br>Y: %{y:.3f}<br><i>Outlier</i><extra></extra>'
                ),
                row=1, col=1
            )

        # Regression line and confidence intervals
        x_range = np.linspace(min(x.min(), y.min()), max(x.max(), y.max()), 100)
        y_reg = slope * x_range + intercept
        
        # Calculate confidence intervals for the regression line
        ci_lower, ci_upper = calculate_confidence_intervals(
            x_range, slope, intercept, cov_matrix, x, confidence_level/100
        )
        
        # Add confidence interval shaded area
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([x_range, x_range[::-1]]),
                y=np.concatenate([ci_upper, ci_lower[::-1]]),
                fill='toself',
                fillcolor='rgba(255,0,0,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name=f'{confidence_level}% CI',
                hoverinfo='skip',
                showlegend=True
            ),
            row=1, col=1
        )
        
        # Add regression line
        fig.add_trace(
            go.Scatter(
                x=x_range, y=y_reg,
                mode='lines',
                line=dict(color='red', width=2),
                name=f'y={slope:.3f}x+{intercept:.3f}, R²={r_squared:.3f}',
                hoverinfo='name'
            ),
            row=1, col=1
        )

        # Line of equality
        fig.add_trace(
            go.Scatter(
                x=x_range, y=x_range,
                mode='lines',
                line=dict(color='gray', width=1, dash='dash'),
                name='y=x',
                hoverinfo='name'
            ),
            row=1, col=1
        )

        # Plot 2: Residuals (Bland-Altman style) - using ALL original data for proper Bland-Altman
        all_means = (original_x + original_y) / 2
        all_differences = original_y - original_x
        mean_diff = np.mean(all_differences)
        std_diff = np.std(all_differences, ddof=1)
        
        # But for the non-outlier points, use the filtered data
        means = (x + y) / 2
        differences = y - x

        fig.add_trace(
            go.Scatter(
                x=means, y=differences,
                mode='markers',
                marker=dict(color='dodgerblue', size=8),
                name='Differences',
                text=sample_ids,
                hovertemplate='<b>%{text}</b><br>Mean: %{x:.3f}<br>Diff: %{y:.3f}<extra></extra>',
                showlegend=False
            ),
            row=2, col=1
        )

        # Add outliers to residuals plot if detected and not excluded
        if use_outlier_detection and np.sum(outlier_flags) > 0 and not exclude_outliers:
            outlier_indices = np.where(outlier_flags)[0]
            outlier_means = (original_x[outlier_indices] + original_y[outlier_indices]) / 2
            outlier_diffs = original_y[outlier_indices] - original_x[outlier_indices]
            
            fig.add_trace(
                go.Scatter(
                    x=outlier_means, y=outlier_diffs,
                    mode='markers',
                    marker=dict(color='red', size=10, symbol='square'),
                    name='Outlier Differences',
                    text=[original_sample_ids[i] for i in outlier_indices],
                    hovertemplate='<b>%{text}</b><br>Mean: %{x:.3f}<br>Diff: %{y:.3f}<br><i>Outlier</i><extra></extra>',
                    showlegend=False
                ),
                row=2, col=1
            )

        # Mean line and limits of agreement
        x_mean_range = np.linspace(all_means.min(), all_means.max(), 100)
        
        fig.add_trace(
            go.Scatter(
                x=x_mean_range, y=[mean_diff] * len(x_mean_range),
                mode='lines',
                line=dict(color='green', width=2),
                name=f'Mean Diff ({mean_diff:.3f})',
                showlegend=False
            ),
            row=2, col=1
        )

        loa_upper = mean_diff + 1.96 * std_diff
        loa_lower = mean_diff - 1.96 * std_diff

        for loa, name in [(loa_upper, 'Upper LoA'), (loa_lower, 'Lower LoA')]:
            fig.add_trace(
                go.Scatter(
                    x=x_mean_range, y=[loa] * len(x_mean_range),
                    mode='lines',
                    line=dict(color='red', width=1, dash='dot'),
                    name=f'{name} ({loa:.3f})',
                    showlegend=False
                ),
                row=2, col=1
            )

        # Update layout
        fig.update_xaxes(title_text=f"{analyzer_1} ({selected_units})", row=1, col=1)
        fig.update_yaxes(title_text=f"{analyzer_2} ({selected_units})", row=1, col=1)
        fig.update_xaxes(title_text=f"Mean ({selected_units})", row=2, col=1)
        fig.update_yaxes(title_text=f"Difference ({selected_units})", row=2, col=1)

        fig.update_layout(
            title=f"Deming Regression: {selected_analyte} ({analyzer_1} vs {analyzer_2})",
            height=800,
            showlegend=True
        )

        st.plotly_chart(fig, use_container_width=True)

        # Statistical Summary
        st.markdown("### 📈 Statistical Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Sample Size", f"{len(x)}" + (f"/{n_original}" if exclude_outliers else ""))
            st.metric("Slope", f"{slope:.4f}")
            st.metric("Slope SE", f"{se_slope:.4f}")
        
        with col2:
            st.metric("Intercept", f"{intercept:.4f}")
            st.metric("Intercept SE", f"{se_intercept:.4f}")
            st.metric("R²", f"{r_squared:.4f}")
        
        with col3:
            if not np.isnan(p_val_slope):
                st.metric("P-value (Slope≠1)", f"{p_val_slope:.4f}")
            if not np.isnan(p_val_intercept):
                st.metric("P-value (Intercept≠0)", f"{p_val_intercept:.4f}")
            if use_outlier_detection:
                st.metric("Outliers Detected", f"{np.sum(outlier_flags)}")
        with col4:
            if not np.isnan(ci_intercept_lower):
                st.metric("Intercept 95% CIs - Lower and Upper", f"{ci_intercept_lower:.3f}, {ci_intercept_upper:.3f}")
            if not np.isnan(ci_slope_lower): 
                st.metric("Slope 95% CIs - Lower and Upper", f"{ci_slope_lower:.3f}, {ci_slope_upper:.3f}")
            # st.markdown("### 📜 Interpretation: Slope ≈ 1 and Intercept ≈ 0 indicate good agreement.")
        
        # Bland-Altman interpretation
        st.markdown("#### Bland-Altman Analysis")
        st.write(f"- Mean difference: {mean_diff:.3f} {selected_units}")
        st.write(f"- 95% Limits of Agreement: [{loa_lower:.3f}, {loa_upper:.3f}] {selected_units}")
        st.write(f"- Standard deviation of differences: {std_diff:.3f} {selected_units}")
        
    except Exception as e:
        st.error(f"❌ Error performing Deming regression: {str(e)}")

if __name__ == "__main__":
    run()