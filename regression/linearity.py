import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats

# Safe import of utils
try:
    from utils import apply_app_styling, units_list
except ImportError:
    st.warning("Utils module not found. Using default settings.")
    def apply_app_styling():
        pass
    units_list = ["mg/L", "μg/L", "ng/L", "mmol/L", "μmol/L", "Units"]

# Page setup
st.set_page_config(
    page_title="Linearity",
    page_icon="📏",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_app_styling()

st.title("📏 Linearity Analysis")

# Documentation sections
with st.expander("📘 About Linearity Assessment", expanded=True):
    st.markdown("""
    **Linearity** refers to the ability of an analytical method to produce results that are directly proportional to the concentration of the analyte in the sample. 
    
    The linear range experiment requires enough of each specimen to prepare dilutions and to carry out testing with 5 or more concentrations. 
    
    A test of linearity starts with a plot of the measured values against the corresponding reference standards to see whether the points fall along a straight line. 
    
    **Calibration Curve**:
    In the context of a calibration curve, we use linearity studies to predict future measurements made with the same instrument. Applying this to calibration results, a calibration line is created using the inverse of a linear model:
    """)
    st.latex(r''' {Y} = \alpha + \beta{x} + \epsilon ''')
    st.markdown("This may be re-worked to provide the calibrated value:")
    st.latex(r'''\bar{X} = \frac{Y - \hat{\alpha}}{\hat{\beta}}''')
    st.markdown("""
    At least one measurement is needed for each standard. The linearity assumption in linear regression means that the relationship between the independent and dependent variable is a straight line. If this is met, the instrument performance is linear. Statistical control in this context implies not only that the measurements are repeatable within certain limits, but that the instrument response also remains linear.
    
    It is a critical aspect of method validation, ensuring that the method provides accurate and reliable results across the entire range of concentrations.
    """)

with st.expander("🔍 Investigating Linearity Issues", expanded=True):   
    st.markdown("""
    Calibration does not always eliminate bias. Several important factors to consider when investigating poor linearity:

    - **Poor Precision**: Instrument imprecision or day-to-day variability can lead to unreliable calibration. Precision should be assessed before selecting an instrument.
    - **Outliers**: Extreme data points, especially at calibration range endpoints, can distort the curve. Isolated outliers should be removed, and inconsistent daily results must be reviewed.
    - **Operator Bias**: Different operators may introduce systematic biases. If significant, consider retraining or creating separate calibration curves per operator.
    - **System Instability**: Instrument drift over time can invalidate calibration. Regular statistical monitoring is essential.
    - **Unseen Day-to-Day Variation**: Aggregated plots may hide daily inconsistencies. Fine-grained plots can help reveal such hidden variability.
    """)

with st.expander("📋 Instructions", expanded=False):
    st.markdown("""
    1. Upload a CSV file containing your standard curve data
    2. Ensure the file includes columns for concentration and response values
    3. Optionally include an identifier column for sample tracking
    4. Select the appropriate columns and units for analysis
    5. Review the results and download reports as needed
    """)

# Initialize variables
df = None
results_df = None
analysis_complete = False

# ================================
# 1. DATA UPLOAD AND PREVIEW
# ================================
st.header("📊 Data Upload and Preview")

uploaded_file = st.file_uploader("Upload CSV file containing standard curve data", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")
    
    with st.expander("📖 Data Preview", expanded=True):
        st.dataframe(df, use_container_width=True)
        st.info(f"Dataset contains {len(df)} rows and {len(df.columns)} columns")
else:
    st.info("Please upload a CSV file to begin analysis")

# ================================
# 2. ANALYSIS CONTROLS
# ================================
if df is not None:
    st.header("⚙️ Controls")
    
    with st.expander("🔧 Configure Analysis Parameters", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            # Analyser filter
            if "Analyser" in df.columns:
                analysers = df["Analyser"].unique().tolist()
                selected_analyser = st.selectbox("Select Analyser", options=["All"] + analysers)
                if selected_analyser != "All":
                    df = df[df["Analyser"] == selected_analyser]
            else:
                selected_analyser = "All"
            
            # Units selection
            units = st.selectbox("Select Units", options=units_list, index=0)
            
            # NEW: Date grouping checkbox
            has_date_column = "Date" in df.columns
            if has_date_column:
                group_by_date = st.checkbox("📅 Group by Date (show individual trendlines by date)", 
                                          value=False,
                                          help="When checked, data will be grouped by date and each date will have its own trendline")
            else:
                group_by_date = False
                if st.checkbox("📅 Group by Date", disabled=True):
                    st.warning("Date column not found in dataset")
        
        with col2:
            # Column selection
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_columns) < 2:
                st.error("❌ Need at least 2 numeric columns for analysis")
                st.stop()
            
            x_axis = st.selectbox("Choose your X-axis (e.g., Expected Concentration)", numeric_columns)
            
            # Auto-suggest next column for Y-axis
            x_index = numeric_columns.index(x_axis)
            default_y_index = min(x_index + 1, len(numeric_columns) - 1)
            y_axis = st.selectbox("Choose your Y-axis (e.g., Calculated Response)", numeric_columns, index=default_y_index)
            
            # Optional identifier column
            identifier_column = st.selectbox("Sample Identifier (e.g., Sample ID)", [None] + df.columns.tolist())

# ================================
# 3. LINEARITY ANALYSIS AND PLOTS
# ================================
if df is not None and len(df.select_dtypes(include=[np.number]).columns) >= 2:
    st.header("📈 Linearity Analysis")
    
    # Prepare data
    clean_df = df[[x_axis, y_axis] + ([identifier_column] if identifier_column else []) + (["Date"] if has_date_column else [])].dropna()
    
    if clean_df.empty:
        st.error("❌ No valid data found in selected columns")
        st.stop()
    
    # Perform linear regression
    x = clean_df[x_axis].to_numpy()
    y = clean_df[y_axis].to_numpy()
    
    try:
        # Calculate overall statistics (used for ungrouped analysis)
        slope, intercept = np.polyfit(x, y, 1)
        fitted_values = slope * x + intercept
        residuals = y - fitted_values
        r_squared = 1 - (np.sum(residuals**2) / np.sum((y - np.mean(y))**2))
        
        # Calculate confidence intervals per standard
        unique_standards = np.unique(x)
        ci_data = []
        points_outside_ci = np.zeros(len(x), dtype=bool)
        
        for standard in unique_standards:
            mask = x == standard
            y_standard = y[mask]
            n_standard = len(y_standard)
            
            if n_standard > 1:
                mean_y = np.mean(y_standard)
                std_y = np.std(y_standard, ddof=1)
                se_mean = std_y / np.sqrt(n_standard)
                t_val = stats.t.ppf(0.975, n_standard - 1)
                ci_upper = mean_y + t_val * se_mean
                ci_lower = mean_y - t_val * se_mean
                
                outside_mask = (y_standard < ci_lower) | (y_standard > ci_upper)
                points_outside_ci[mask] = outside_mask
                
                ci_data.append({
                    'standard': standard,
                    'mean': mean_y,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper,
                    'n_points': n_standard,
                    'points_outside': np.sum(outside_mask)
                })
            else:
                ci_data.append({
                    'standard': standard,
                    'mean': y_standard[0],
                    'ci_lower': y_standard[0],
                    'ci_upper': y_standard[0],
                    'n_points': 1,
                    'points_outside': 0
                })
        
        points_outside_count = np.sum(points_outside_ci)
        total_points = len(x)
        ci_percentage = ((total_points - points_outside_count) / total_points) * 100
        
        # Create plot
        with st.expander("📊 Standard Curve Plot", expanded=True):
            hover_text = (
                clean_df[identifier_column].astype(str) + "<br>"
                + x_axis + ": " + clean_df[x_axis].astype(str) + "<br>"
                + y_axis + ": " + clean_df[y_axis].astype(str)
            ) if identifier_column else None
            
            fig = go.Figure()
            
            # Color palette for different dates
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
            
            if group_by_date and has_date_column:
                # Group by date and create separate trendlines
                unique_dates = clean_df["Date"].unique()
                date_stats = []
                
                for i, date in enumerate(unique_dates):
                    date_mask = clean_df["Date"] == date
                    date_data = clean_df[date_mask]
                    
                    if len(date_data) >= 2:  # Need at least 2 points for a line
                        date_x = date_data[x_axis].to_numpy()
                        date_y = date_data[y_axis].to_numpy()
                        
                        # Calculate statistics for this date
                        date_slope, date_intercept = np.polyfit(date_x, date_y, 1)
                        date_fitted = date_slope * date_x + date_intercept
                        date_residuals = date_y - date_fitted
                        date_r_squared = 1 - (np.sum(date_residuals**2) / np.sum((date_y - np.mean(date_y))**2))
                        
                        date_stats.append({
                            'date': date,
                            'r_squared': date_r_squared,
                            'slope': date_slope,
                            'intercept': date_intercept,
                            'n_points': len(date_data)
                        })
                        
                        color = colors[i % len(colors)]
                        
                        # Format the linear equation for the legend
                        equation = f"y = {date_slope:.4f}x + {date_intercept:.4f}"
                        
                        # Add data points for this date
                        fig.add_trace(go.Scatter(
                            x=date_x, y=date_y,
                            mode='markers',
                            name=f'{equation} ({date})',
                            marker=dict(color=color, size=8),
                            text=hover_text[date_mask] if hover_text is not None else None,
                            hoverinfo='text' if identifier_column else 'x+y'
                        ))
                        
                        # Add fitted line for this date
                        x_range = np.linspace(date_x.min(), date_x.max(), 100)
                        y_fitted_line = date_slope * x_range + date_intercept
                        
                        fig.add_trace(go.Scatter(
                            x=x_range, y=y_fitted_line,
                            mode='lines',
                            name=f'Fit - {date} (R²={date_r_squared:.3f})',
                            line=dict(color=color, width=2, dash='solid'),
                            hoverinfo='skip'
                        ))
                
                # Add overall statistics as text annotation
                fig.add_annotation(
                    x=0.98, y=0.98,
                    xref='paper', yref='paper',
                    text=f"Overall: R² = {r_squared:.4f}, Slope = {slope:.4f}",
                    showarrow=False,
                    font=dict(size=12, color="black"),
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="gray",
                    borderwidth=1
                )
                
                plot_title = f"Standard Curve - Grouped by Date ({len(unique_dates)} dates)"
                
                # Display date statistics
                if date_stats:
                    st.subheader("📊 Date Group Statistics")
                    date_stats_df = pd.DataFrame(date_stats)
                    st.dataframe(date_stats_df.round(4), use_container_width=True)
                
            else:
                # Original ungrouped analysis
                # Add confidence intervals
                for ci in ci_data:
                    if ci['n_points'] > 1:
                        fig.add_trace(go.Scatter(
                            x=[ci['standard']],
                            y=[ci['mean']],
                            error_y=dict(
                                type='data',
                                array=[ci['ci_upper'] - ci['mean']],
                                arrayminus=[ci['mean'] - ci['ci_lower']],
                                visible=True,
                                color='rgba(255, 0, 0, 0.7)',
                                thickness=2,
                                width=3
                            ),
                            mode='markers',
                            marker=dict(color='red', size=8, symbol='diamond'),
                            name=f'95% CI (Std {ci["standard"]})',
                            showlegend=False,
                            hoverinfo='skip'
                        ))
                
                # Add legend entry for CIs
                fig.add_trace(go.Scatter(
                    x=[None], y=[None],
                    mode='markers',
                    marker=dict(color='red', size=8, symbol='diamond'),
                    name='95% CI per Standard',
                    showlegend=True,
                    hoverinfo='skip'
                ))
                
                # Add data points
                fig.add_trace(go.Scatter(
                    x=x, y=y,
                    mode='markers',
                    name='Data Points',
                    marker=dict(color='blue', size=8),
                    text=hover_text,
                    hoverinfo='text' if identifier_column else 'x+y'
                ))
                
                # Add fitted line
                x_sorted = np.sort(x)
                fitted_sorted = slope * x_sorted + intercept
                fig.add_trace(go.Scatter(
                    x=x_sorted, y=fitted_sorted,
                    mode='lines',
                    name=f"Fit: y = {slope:.4f}x + {intercept:.4f}<br>R² = {r_squared:.4f}",
                    line=dict(color='red', width=2)
                ))
                
                plot_title = "Standard Curve with Linear Fit and 95% Confidence Intervals"
            
            fig.update_layout(
                title=plot_title,
                xaxis_title=f"{x_axis} ({units})",
                yaxis_title=f"{y_axis} ({units})",
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                height=750,     
                width=1000,
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        analysis_complete = True
        
    except Exception as e:
        st.error(f"❌ Analysis failed: {str(e)}")
        st.stop()

# ================================
# 4. RESULTS PREVIEW AND SUMMARY
# ================================
if analysis_complete:
    st.header("📋 Results Summary")

    with st.expander("🧠 Interpretation", expanded=True):
        # Use columns for side-by-side display
        col1, col2 = st.columns([1, 1])

        # R² Interpretation
        with col1:
            st.subheader("📈 Linearity")
            if r_squared >= 0.99:
                st.success("**Excellent** linearity")
                st.write("The response is highly consistent with the standard concentrations (R² ≥ 0.99).")
            elif r_squared >= 0.95:
                st.info("**Good** linearity")
                st.write("Results are acceptable, but further verification may be considered (R² ≥ 0.95).")
            elif r_squared >= 0.90:
                st.warning("**Moderate** linearity")
                st.write("Further investigation may be needed for accuracy at extreme points (R² ≥ 0.90).")
            else:
                st.error("**Poor** linearity")
                st.write("Data may not be reliable for quantitative analysis (R² < 0.90).")

        # CI Interpretation
        with col2:
            st.subheader("🎯 Precision")
            if points_outside_count == 0:
                st.success("All points within 95% CI")
                st.write(f"All **{total_points}** data points fall within their respective confidence intervals — indicating excellent precision.")
            elif points_outside_count <= 0.05 * total_points:
                st.info("Acceptable precision")
                st.write(f"**{ci_percentage:.1f}%** of points are within CI. A small number of deviations are acceptable.")
            else:
                st.warning("Reduced precision")
                st.write(f"Only **{ci_percentage:.1f}%** of points fall within their confidence intervals. Consider investigating the outliers.")

    
    # Key metrics
    with st.expander("📊 Key Metrics", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("R² Value", f"{r_squared:.4f}")
            st.metric("Slope", f"{slope:.4f}")
        
        with col2:
            st.metric("Intercept", f"{intercept:.4f}")
            st.metric("Total Points", total_points)
        
        with col3:
            st.metric("CI Coverage", f"{ci_percentage:.1f}%")
            st.metric("Points Outside CI", points_outside_count)
        
        with col4:
            st.metric("Mean Residual", f"{np.mean(residuals):.4f}")
            st.metric("Residual Std", f"{np.std(residuals):.4f}")
    
    # Detailed results preview
    with st.expander("📋 Detailed Results Preview", expanded=False):
        # Deviation Assessment
        st.subheader("📊 Residuals and 95% Confidence Intervals Assessment")
        results_df = clean_df.copy()
        
        if identifier_column:
            results_df = results_df[[identifier_column, x_axis, y_axis]].copy()
        else:
            results_df = results_df[[x_axis, y_axis]].copy()
            
        results_df["Fitted Value"] = fitted_values
        results_df["Residuals"] = residuals

        # Add CI information
        ci_lower_per_point = np.zeros(len(x))
        ci_upper_per_point = np.zeros(len(x))

        for ci in ci_data:
            mask = x == ci['standard']
            ci_lower_per_point[mask] = ci['ci_lower']
            ci_upper_per_point[mask] = ci['ci_upper']

        results_df["95% CI Lower"] = ci_lower_per_point
        results_df["95% CI Upper"] = ci_upper_per_point
        results_df["Outside 95% CI"] = points_outside_ci

        show_summary = st.checkbox("🔍 Show summary instead of detailed table")

        if show_summary and identifier_column:
            # Group by Sample ID for summary
            summary_df = results_df.groupby(identifier_column).agg(
                Mean_Fitted_Value=("Fitted Value", "mean"),
                Mean_Residual=("Residuals", "mean"),
                Std_Residual=("Residuals", "std"),
                Num_Outside_CI=("Outside 95% CI", "sum"),
                Total_Points=("Outside 95% CI", "count"),
            ).reset_index()

            summary_df["% Outside CI"] = (
                summary_df["Num_Outside_CI"] / summary_df["Total_Points"] * 100
            ).round(2)

            st.dataframe(summary_df, use_container_width=True)

        else:
            # Show detailed table with Sample ID at the start
            st.dataframe(results_df, use_container_width=True)

        
        # Deviation Assessment
        st.subheader("📊 Deviation Assessment")
        
        if "Date" in df.columns and identifier_column:
            # Group data by Date and calculate statistics
            date_groups = df.groupby("Date")
            deviation_rows = []
            
            for date, group_data in date_groups:
                # Calculate R² and Slope for this date's data
                date_x = group_data[x_axis].dropna()
                date_y = group_data[y_axis].dropna()
                
                if len(date_x) >= 2 and len(date_y) >= 2:
                    # Ensure same length
                    min_len = min(len(date_x), len(date_y))
                    date_x = date_x.iloc[:min_len]
                    date_y = date_y.iloc[:min_len]
                    
                    date_slope, date_intercept = np.polyfit(date_x, date_y, 1)
                    date_fitted = date_slope * date_x + date_intercept
                    date_residuals = date_y - date_fitted
                    date_r2 = 1 - (np.sum(date_residuals**2) / np.sum((date_y - np.mean(date_y))**2))
                    
                    # Initialize row data
                    row_data = {
                        "Date": date,
                        "R²": round(date_r2, 4),
                        "Slope": round(date_slope, 4)
                    }
                    
                    # Calculate % deviation for each Sample ID
                    sample_ids = group_data[identifier_column].unique()
                    
                    for sample_id in sample_ids:
                        sample_data = group_data[group_data[identifier_column] == sample_id]
                        
                        if len(sample_data) > 0:
                            # Calculate expected vs actual for this sample
                            expected_values = sample_data[x_axis].values
                            actual_values = sample_data[y_axis].values
                            
                            if len(expected_values) > 0 and len(actual_values) > 0:
                                # Calculate mean deviation percentage
                                deviations = []
                                for exp, act in zip(expected_values, actual_values):
                                    if exp != 0:
                                        deviation = ((act - exp) / exp) * 100
                                        deviations.append(deviation)
                                
                                if deviations:
                                    mean_deviation = np.mean(deviations)
                                    row_data[f"% Deviation {sample_id}"] = round(mean_deviation, 2)
                    
                    deviation_rows.append(row_data)
            
            if deviation_rows:
                deviation_df = pd.DataFrame(deviation_rows)
                st.dataframe(deviation_df, use_container_width=True)
            else:
                st.warning("No sufficient data for deviation assessment")
                
        elif "Date" not in df.columns:
            st.warning("Date column not found in dataset. Cannot perform deviation assessment.")
        elif not identifier_column:
            st.warning("Sample identifier column not selected. Cannot perform deviation assessment.")
        else:
            st.warning("Insufficient data for deviation assessment.")

# ================================
# 5. RECOVERY ANALYSIS
# ================================
if analysis_complete:
    with st.expander("📊 Recovery Analysis", expanded=False):
        
        selectable_columns = df.columns[6:] if len(df.columns) > 6 else df.columns
        
        if len(selectable_columns) >= 2:
            col1, col2 = st.columns(2)
            
            with col1:
                expected_column = st.selectbox("Expected Concentration Column", selectable_columns)
            
            with col2:
                calculated_column = st.selectbox("Calculated Concentration Column", selectable_columns)
            
            if expected_column != calculated_column:
                recovery_df = clean_df.copy()
                recovery_df["Sample_ID"] = df["Sample ID"] if "Sample ID" in df.columns else "N/A"
                recovery_df["Expected"] = df[expected_column]
                recovery_df["Calculated"] = df[calculated_column]
                
                recovery_df["Recovery (%)"] = np.where(
                    recovery_df["Expected"] > 0, 
                    (recovery_df["Calculated"] / recovery_df["Expected"]) * 100, 
                    np.nan
                )
                
                show_summary = st.checkbox("Show Summary by Sample")
                
                if show_summary:
                    summary_df = recovery_df.groupby("Sample_ID").agg({
                        "Expected": "mean",
                        "Calculated": "mean", 
                        "Recovery (%)": "mean"
                    }).reset_index().round(2)
                    
                    st.dataframe(summary_df, use_container_width=True)
                else:
                    display_cols = ["Sample_ID", "Expected", "Calculated", "Recovery (%)"]
                    st.dataframe(recovery_df[display_cols], use_container_width=True)
        else:
            st.warning("Not enough columns available for recovery calculation")

# ================================
# 6. DOWNLOAD SECTION
# ================================
if analysis_complete:    
    st.markdown("   ")
    with st.expander("📁 Download Options", expanded=True):
        # Prepare comprehensive results
        comprehensive_results = results_df.copy()
        comprehensive_results["Analysis_Date"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        comprehensive_results["Selected_Analyser"] = selected_analyser
        comprehensive_results["Units"] = units
        comprehensive_results["R_Squared"] = r_squared
        comprehensive_results["Slope"] = slope
        comprehensive_results["Intercept"] = intercept
        
        # Summary statistics
        summary_stats = {
            "Total_Points": len(comprehensive_results),
            "R_Squared": r_squared,
            "Slope": slope,
            "Intercept": intercept,
            "Points_Outside_CI": points_outside_count,
            "CI_Coverage_Percent": ci_percentage,
            "Mean_Residual": np.mean(residuals),
            "Std_Residual": np.std(residuals),
            "Analysis_Timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.download_button(
                label="📊 Download Detailed Results",
                data=comprehensive_results.to_csv(index=False).encode('utf-8'),
                file_name=f"linearity_detailed_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                help="Complete results with fitted values and residuals"
            )
        
        with col2:
            summary_report = pd.DataFrame([summary_stats])
            st.download_button(
                label="📋 Download Summary Report",
                data=summary_report.to_csv(index=False).encode('utf-8'),
                file_name=f"linearity_summary_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                help="Summary statistics and key metrics"
            )
        
        with col3:
            # CI details
            ci_summary_df = pd.DataFrame(ci_data)
            st.download_button(
                label="📈 Download CI Details",
                data=ci_summary_df.to_csv(index=False).encode('utf-8'),
                file_name=f"linearity_ci_details_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                help="Confidence interval details per standard"
            )

# ================================
# 7. ADDITIONAL ANALYSIS TOOLS
# ================================
if analysis_complete:
    st.markdown("---")
    st.subheader("🔍 Additional Analysis Tools")
    
    # Residual Analysis
    with st.expander("🎢 Residuals", expanded=False):
               
        fig_residual = go.Figure()
        fig_residual.add_trace(go.Scatter(
            x=fitted_values,
            y=residuals,
            mode='markers',
            name='Residuals',
            marker=dict(color='blue', size=8),
            text=hover_text if identifier_column else None,
            hoverinfo='text' if identifier_column else 'x+y'
        ))
        
        fig_residual.add_hline(y=0, line_dash="dash", line_color="red", 
                              annotation_text="Zero Line")
        
        fig_residual.update_layout(
            title="Residual Plot",
            xaxis_title="Fitted Values",
            yaxis_title="Residuals",
            height=500
        )
        
        st.plotly_chart(fig_residual, use_container_width=True)
        st.info("Residuals should be randomly distributed around zero for a good linear fit.")

        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Mean", f"{np.mean(residuals):.4f}")
            st.metric("Median", f"{np.median(residuals):.4f}")
        
        with col2:
            st.metric("Std Dev", f"{np.std(residuals):.4f}")
            st.metric("Range", f"{np.max(residuals) - np.min(residuals):.4f}")
        
        with col3:
            st.metric("Min", f"{np.min(residuals):.4f}")
            st.metric("Max", f"{np.max(residuals):.4f}")
    
    # Outlier Detection
    with st.expander("🎯 Outlier Detection", expanded=False):      
        # st.markdown("Points ±3 standard deviations from mean residual are flagged as outliers. ")  
        residual_mean = np.mean(residuals)
        residual_std = np.std(residuals)
        outlier_threshold = 3 * residual_std
        
        outliers = np.abs(residuals - residual_mean) > outlier_threshold
        outlier_count = np.sum(outliers)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Outlier Threshold (±3σ)", f"{outlier_threshold:.4f}")
        
        with col2:
            st.metric("Outliers Found", outlier_count)
        
        with col3:
            st.metric("Outlier Rate", f"{(outlier_count / len(residuals)) * 100:.1f}%")
        
        if outlier_count > 0:
            st.warning(f"⚠️ {outlier_count} potential outlier(s) detected")
            
            outlier_data = comprehensive_results[outliers].copy()
            outlier_data['Outlier Score'] = np.abs(residuals[outliers] - residual_mean) / residual_std
            
            display_columns = [x_axis, y_axis, 'Fitted Value', 'Residuals', 'Outlier Score']
            if identifier_column:
                display_columns.insert(0, identifier_column)
            
            st.dataframe(outlier_data[display_columns], use_container_width=True)
        else:
            st.success("✅ No statistical outliers detected")
    
    # Method Performance Assessment
    with st.expander("🎯 Method Performance Assessment", expanded=False):
        
        cv_residuals = (np.std(residuals) / np.mean(np.abs(fitted_values))) * 100 if np.mean(np.abs(fitted_values)) > 0 else 0
        
        performance_score = "Excellent" if r_squared >= 0.99 else "Good" if r_squared >= 0.95 else "Fair" if r_squared >= 0.90 else "Poor"
        
        # col1, col2 = st.columns(2)
        
        # with col1:
        #     st.metric("Overall Performance", performance_score)
        #     st.metric("R² Value", f"{r_squared:.4f}")
        #     st.metric("Slope", f"{slope:.4f}")
        #     st.metric("Intercept", f"{intercept:.4f}")
        
        # with col2:
        #     st.metric("CI Coverage", f"{ci_percentage:.1f}%")
        #     st.metric("CV of Residuals", f"{cv_residuals:.2f}%")
        #     st.metric("Outlier Rate", f"{(outlier_count / len(residuals)) * 100:.1f}%")
        #     st.metric("Standards Tested", len(np.unique(x)))
        
        # Performance summary table
        performance_metrics = {
            "Metric": ["Overall Score", "R²", "Slope", "Intercept", "CI Coverage (%)", "CV Residuals (%)", "Outlier Rate (%)", "Standards"],
            "Value": [performance_score, f"{r_squared:.4f}", f"{slope:.4f}", f"{intercept:.4f}", 
                     f"{ci_percentage:.1f}", f"{cv_residuals:.2f}", f"{(outlier_count / len(residuals)) * 100:.1f}", len(np.unique(x))]
        }
        
        performance_df = pd.DataFrame(performance_metrics)
        st.dataframe(performance_df, use_container_width=True)

# Add custom CSS for better styling
st.markdown("""
<style>
.stMetric > div > div > div > div {
    background-color: #f0f2f6;
    border: 2px solid #e1e5e9;
    padding: 15px;
    border-radius: 5px;
    margin: 10px;
}
</style>
""", unsafe_allow_html=True)