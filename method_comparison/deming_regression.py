# deming_regression.py
import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.odr import ODR, RealData, Model
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils import apply_app_styling, units_list
from data_preparation import get_analysis_ready_data
from outlier_detection import standardized_outlier_detection, create_outlier_explanation_section

apply_app_styling()

# === Utility Functions ===
def perform_deming_regression(x, y, delta=1.0):
    """
    Perform Deming regression and return results.
    """
    def linear(B, x):
        return B[0] * x + B[1]

    model = Model(linear)
    odr_data = RealData(x, y, sx=np.std(x) * np.sqrt(delta), sy=np.std(y))
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
    t_val = stats.t.ppf(1 - alpha / 2, n - 2)

    # Variance of predicted y for each x value
    
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

    st.set_page_config(
        page_title="Deming Regression Analysis",
        page_icon="⚖️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.header("⚖️ Deming Regression Analysis")

    with st.expander("📘 What is Deming regression?", expanded=True):
        st.markdown("""
        **Deming regression** is a statistical technique used for linear regression when **both the X and Y variables are subject to measurement error**. This situation is common in laboratory method comparison studies, where two instruments or methods are used to measure the same analyte.

        Unlike ordinary least squares (OLS) regression—which assumes that all error resides in the Y variable—Deming regression accounts for **errors in both dimensions**, providing a more balanced and realistic model of the relationship.

        Key metrics:
        - **Slope**: Indicates **proportional bias**. A slope of 1 suggests that both methods increase at the same rate.
        - **Intercept**: Indicates **constant bias**. A value of 0 means there is no fixed offset between the two methods.
        - **R² (coefficient of determination)**: Reflects how well the two methods agree across the measurement range. Closer to 1 indicates stronger agreement.
        """)

    with st.expander("📘 Why use Deming regression?", expanded=True):
        st.markdown("""
        - ✅ **Accounts for measurement error in both variables**: This makes it more appropriate than OLS regression when comparing instruments or methods, both of which may introduce variability.
        - 🔬 **Suitable for analytical method validation**: It is especially useful when establishing whether two methods produce equivalent results across a range of concentrations.
        - 📊 **Statistically robust**: Reduces bias in slope and intercept estimates, leading to more reliable conclusions when assessing method agreement.
        - 🔁 **Symmetric treatment of variables**: Interchanging the X and Y variables does not change the fitted line, unlike OLS.
        """)

    with st.expander("📘 Instructions", expanded=False):
        st.markdown("""
        To perform a Deming regression analysis on your dataset:

        1. 📄 **Upload your data file** (CSV format). It must include the following columns:
            - `Analyser`: Identifier for the measurement instrument.
            - `Material`: Type of sample (e.g., QC, patient, calibrator).
            - `Sample ID`: Unique identifier for each sample.
            - One or more **analyte columns** with numeric values.

        2. ➕ **Select two analysers** to compare. One will be treated as the X-axis (reference), the other as Y-axis (test).

        3. 🧪 **Choose the material type** to focus the comparison (e.g., only QC or patient data).

        4. 📈 **Pick an analyte** for which to perform the regression.

        5. 🎯 **Set your desired confidence level** (e.g., 95%) for the regression intervals.

        6. 🚨 Optionally enable **outlier detection** to identify and mitigate extreme values that may skew the regression.

        7. ▶️ Click **Run Analysis** to execute the Deming regression and view:
            - A scatter plot with the regression line and confidence intervals.
            - Calculated slope, intercept, and R².
            - A bias table and optional Bland-Altman plot (if enabled).
        """)

    # File upload section
    with st.expander("📤 Upload CSV File", expanded=True):
        uploaded_file = st.file_uploader("   ", type=["csv"])

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

        if analyzer_1 == analyzer_2:
            st.warning("⚠ Please select two different analyzers.")
            st.stop()

        alpha = 1 - confidence_level / 100

        # Add duplicate handling option
        duplicate_handling = st.selectbox(
            "Handle Duplicate Sample IDs:",
            options=['mean', 'first', 'last'],
            index=0,
            help="""
            • mean: Average multiple measurements for the same Sample ID
            • first: Keep the first occurrence of each Sample ID
            • last: Keep the last occurrence of each Sample ID
            """
        )
    # ===== MISSING DATA PROCESSING SECTION - THIS WAS THE PROBLEM =====
    st.markdown("### 📊 Data Processing")

    try:
        original_x, original_y, original_sample_ids, n_original, merged_data_full = get_analysis_ready_data(
            df, selected_material, selected_analyte, analyzer_1, analyzer_2,
            handle_duplicates='mean', verbose=True
        )
    except ValueError as e:
        st.error(f"❌ {str(e)}")
        st.stop()
    except Exception as e:
        st.error(f"❌ Unexpected error in data preparation: {str(e)}")
        st.stop()

    st.success(f"✅ Successfully prepared {n_original} matched sample pairs for analysis")

    # ===== NEW: Standardized Outlier Detection =====
    st.markdown("### 🔍 Outlier Detection")

    outlier_results = {}
    if use_outlier_detection:
        outlier_results = standardized_outlier_detection(
            merged_data_full, selected_analyte, analyzer_1, analyzer_2, alpha=alpha, analysis_type='deming_regression'
        )

        outlier_flags = outlier_results.get('outliers_mask', np.array([False] * n_original))
        exclude_outliers = outlier_results.get('exclude_outliers', False)

        if exclude_outliers:
            non_outlier_mask = ~outlier_flags
            x = original_x[non_outlier_mask]
            y = original_y[non_outlier_mask]
            sample_ids = [original_sample_ids[i] for i in range(len(original_sample_ids)) if non_outlier_mask[i]]
        else:
            x = original_x.copy()
            y = original_y.copy()
            sample_ids = original_sample_ids.copy()

    else:
        # If detection is not enabled, use all original data
        x = original_x.copy()
        y = original_y.copy()
        sample_ids = original_sample_ids.copy()
        outlier_flags = np.array([False] * n_original)
        exclude_outliers = False
        st.info("ℹ️ Outlier detection is disabled. All data points will be included in the analysis.")

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
            x_range, slope, intercept, cov_matrix, x, confidence_level / 100
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
        fig.update_xaxes(title_text=f"Mean ({analyzer_1} & {analyzer_2}) ({selected_units})", row=2, col=1)
        fig.update_yaxes(title_text=f"Difference ({analyzer_2} - {analyzer_1}) ({selected_units})", row=2, col=1)

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

        # NEW: Call the explanation section
        if use_outlier_detection and exclude_outliers and outlier_results.get('n_outliers', 0) > 0:
            # Gather statistics before exclusion
            original_mean_diff = np.mean(original_y - original_x)
            original_std_diff = np.std(original_y - original_x, ddof=1)

            # Gather statistics after exclusion
            final_mean_diff = np.mean(y - x)
            final_std_diff = np.std(y - x, ddof=1)

            final_r_squared = r_squared

            original_stats = {
                'n': n_original,
                'mean_diff': original_mean_diff,
                'std_diff': original_std_diff,
            }

            final_stats = {
                'n': len(x),
                'mean_diff': final_mean_diff,
                'std_diff': final_std_diff,
                'correlation': final_r_squared
            }

            create_outlier_explanation_section(
                method_name=outlier_results['method_name'],
                n_excluded=outlier_results['n_outliers'],
                excluded_sample_ids=outlier_results['outlier_sample_ids'],
                original_stats=original_stats,
                final_stats=final_stats,
                alpha=alpha
            )

    except Exception as e:
        st.error(f"❌ Error performing Deming regression: {str(e)}")

if __name__ == "__main__":
    run()