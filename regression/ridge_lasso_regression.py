import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import r2_score

# Safe import of utils
try:
    from utils import apply_app_styling, units_list
except ImportError:
    st.warning("Utils module not found. Using default settings.")
    def apply_app_styling():
        pass
    units_list = ["mg/L", "μg/L", "ng/L", "mmol/L", "μmol/L", "Units"]

# Page setup
apply_app_styling()

st.header("📏 Ridge & Lasso Regression")
st.subheader("Regularized Regression for Linearity Analysis")

with st.expander("📘 What is Ridge and Lasso regression?", expanded=True):
    st.markdown("""
    **Ridge and Lasso regression** are extensions of linear regression that add a penalty to the model's complexity. This is particularly useful for preventing **overfitting** and handling **multicollinearity** (when predictors are highly correlated).

    - **Ridge Regression** adds an L2 penalty, which shrinks the coefficients towards zero but doesn't force them to be exactly zero. This helps stabilize the model and is effective for datasets with many correlated features.
    
    - **Lasso Regression** adds an L1 penalty, which can force some coefficients to be exactly zero. This performs feature selection and is useful for building sparser models where only the most important features are retained.

    Both models include a hyperparameter, **alpha ($\lambda$)**, which controls the strength of the regularization. A higher alpha value increases the penalty on the coefficients, leading to a simpler model.
    """)
    st.latex(r''' \text{Ridge: } \min_w ||Y - Xw||_2^2 + \lambda ||w||_2^2 ''')
    st.latex(r''' \text{Lasso: } \min_w \frac{1}{2n_{samples}} ||Y - Xw||_2^2 + \lambda ||w||_1 ''')


# ================================
# 1. DATA UPLOAD AND PREVIEW
# ================================
with st.expander("📤 Upload Your CSV File", expanded=True):
    uploaded_file = st.file_uploader("   ", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")
    
    with st.expander("📖 Data Preview", expanded=True):
        st.dataframe(df, use_container_width=True)

# ================================
# 2. ANALYSIS CONTROLS
# ================================
if uploaded_file is not None:
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
            
            # NEW: Regression Model selection (Ridge or Lasso)
            regression_model = st.selectbox(
                "Choose Regression Model", 
                options=["Ridge", "Lasso"],
                help="Ridge and Lasso add regularization to the regression line."
            )
            
            # NEW: Alpha slider for Ridge and Lasso
            alpha = st.slider(
                'Alpha (Regularization Strength, $\lambda$)',
                min_value=0.0,
                max_value=1.0,
                value=0.1,
                step=0.01,
                help="Higher alpha values increase the penalty on model coefficients. A value of 0 is equivalent to standard Linear Regression."
            )

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
            
analysis_complete = False
# ================================
# 3. LINEARITY ANALYSIS AND PLOTS
# ================================
if uploaded_file is not None and len(df.select_dtypes(include=[np.number]).columns) >= 2:
    st.header("📈 Regression Analysis")

    # Prepare data
    clean_df = df[[x_axis, y_axis] + ([identifier_column] if identifier_column else []) + (["Date"] if has_date_column else [])].dropna()

    if clean_df.empty:
        st.error("❌ No valid data found in selected columns")
        st.stop()
    
    # Perform selected regression
    x = clean_df[[x_axis]]
    y = clean_df[y_axis]
    
    try:
        # Choose the model based on user selection
        if regression_model == "Ridge":
            model = Ridge(alpha=alpha)
        else: # Lasso
            model = Lasso(alpha=alpha)
            
        model.fit(x, y)
        
        slope = model.coef_[0]
        intercept = model.intercept_
        
        fitted_values = model.predict(x)
        residuals = y - fitted_values
        r_squared = r2_score(y, fitted_values)
        
        # Calculate confidence intervals per standard (same logic as before)
        unique_standards = np.unique(x)
        ci_data = []
        points_outside_ci = np.zeros(len(x), dtype=bool)
        
        for standard in unique_standards:
            mask = x[x_axis] == standard
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
                    'mean': y_standard.iloc[0],
                    'ci_lower': y_standard.iloc[0],
                    'ci_upper': y_standard.iloc[0],
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
                unique_dates = clean_df["Date"].unique()
                date_stats = []
                
                for i, date in enumerate(unique_dates):
                    date_mask = clean_df["Date"] == date
                    date_data = clean_df[date_mask]
                    
                    if len(date_data) >= 2:  # Need at least 2 points for a line
                        date_x = date_data[x_axis].to_numpy().reshape(-1, 1)
                        date_y = date_data[y_axis].to_numpy()
                        
                        # Calculate statistics for this date using the selected model
                        if regression_model == "Ridge":
                            date_model = Ridge(alpha=alpha)
                        else: # Lasso
                            date_model = Lasso(alpha=alpha)
                            
                        date_model.fit(date_x, date_y)
                        date_slope = date_model.coef_[0]
                        date_intercept = date_model.intercept_
                        
                        date_fitted = date_model.predict(date_x)
                        date_r_squared = r2_score(date_y, date_fitted)
                        
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
                            x=date_x.flatten(), y=date_y,
                            mode='markers',
                            name=f'{equation} ({date})',
                            marker=dict(color=color, size=8),
                            text=hover_text[date_mask] if hover_text is not None else None,
                            hoverinfo='text' if identifier_column else 'x+y'
                        ))
                        
                        # Add fitted line for this date
                        x_range = np.linspace(date_x.min(), date_x.max(), 100).reshape(-1, 1)
                        y_fitted_line = date_model.predict(x_range)
                        
                        fig.add_trace(go.Scatter(
                            x=x_range.flatten(), y=y_fitted_line,
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
                
                plot_title = f"Standard Curve with {regression_model} Fit - Grouped by Date"
                
                # Display date statistics
                if date_stats:
                    st.subheader("📊 Date Group Statistics")
                    date_stats_df = pd.DataFrame(date_stats)
                    st.dataframe(date_stats_df.round(4), use_container_width=True)
                
            else:
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
                    x=x[x_axis].to_numpy(), y=y.to_numpy(),
                    mode='markers',
                    name='Data Points',
                    marker=dict(color='blue', size=8),
                    text=hover_text,
                    hoverinfo='text' if identifier_column else 'x+y'
                ))
                
                # Add fitted line
                x_range = np.linspace(x.min().item(), x.max().item(), 100).reshape(-1, 1)
                fitted_sorted = model.predict(x_range)
                fig.add_trace(go.Scatter(
                    x=x_range.flatten(), y=fitted_sorted,
                    mode='lines',
                    name=f"Fit: y = {slope:.4f}x + {intercept:.4f}<br>R² = {r_squared:.4f}",
                    line=dict(color='red', width=2)
                ))
                
                plot_title = f"Standard Curve with {regression_model} Fit (α={alpha})"
            
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
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("📈 Fit Quality")
            if r_squared >= 0.99:
                st.success("**Excellent** fit")
                st.write(f"The response is highly consistent with the standard concentrations (R² ≥ 0.99).")
            elif r_squared >= 0.95:
                st.info("**Good** fit")
                st.write(f"Results are acceptable, but further verification may be considered (R² ≥ 0.95).")
            elif r_squared >= 0.90:
                st.warning("**Moderate** fit")
                st.write(f"Further investigation may be needed for accuracy at extreme points (R² ≥ 0.90).")
            else:
                st.error("**Poor** fit")
                st.write(f"Data may not be reliable for quantitative analysis (R² < 0.90).")

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
        st.subheader("📊 Residuals and 95% Confidence Intervals Assessment")
        results_df = clean_df.copy()
        
        if identifier_column:
            results_df = results_df[[identifier_column, x_axis, y_axis]].copy()
        else:
            results_df = results_df[[x_axis, y_axis]].copy()
            
        results_df["Fitted Value"] = fitted_values
        results_df["Residuals"] = residuals

        ci_lower_per_point = np.zeros(len(x))
        ci_upper_per_point = np.zeros(len(x))

        for ci in ci_data:
            mask = x[x_axis] == ci['standard']
            ci_lower_per_point[mask] = ci['ci_lower']
            ci_upper_per_point[mask] = ci['ci_upper']

        results_df["95% CI Lower"] = ci_lower_per_point
        results_df["95% CI Upper"] = ci_upper_per_point
        results_df["Outside 95% CI"] = points_outside_ci

        show_summary = st.checkbox("🔍 Show summary instead of detailed table")

        if show_summary and identifier_column:
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
            st.dataframe(results_df, use_container_width=True)

        st.subheader("📊 Deviation Assessment")
        
        if "Date" in df.columns and identifier_column:
            date_groups = df.groupby("Date")
            deviation_rows = []
            
            for date, group_data in date_groups:
                date_x = group_data[x_axis].dropna()
                date_y = group_data[y_axis].dropna()
                
                if len(date_x) >= 2 and len(date_y) >= 2:
                    min_len = min(len(date_x), len(date_y))
                    date_x = date_x.iloc[:min_len].to_numpy().reshape(-1, 1)
                    date_y = date_y.iloc[:min_len].to_numpy()
                    
                    if regression_model == "Ridge":
                        date_model = Ridge(alpha=alpha)
                    else:
                        date_model = Lasso(alpha=alpha)
                    
                    date_model.fit(date_x, date_y)
                    date_slope = date_model.coef_[0]
                    date_intercept = date_model.intercept_
                    
                    date_fitted = date_model.predict(date_x)
                    date_r2 = r2_score(date_y, date_fitted)
                    
                    row_data = {
                        "Date": date,
                        "R²": round(date_r2, 4),
                        "Slope": round(date_slope, 4)
                    }
                    
                    sample_ids = group_data[identifier_column].unique()
                    
                    for sample_id in sample_ids:
                        sample_data = group_data[group_data[identifier_column] == sample_id]
                        
                        if len(sample_data) > 0:
                            expected_values = sample_data[x_axis].values
                            actual_values = sample_data[y_axis].values
                            
                            if len(expected_values) > 0 and len(actual_values) > 0:
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
        comprehensive_results = results_df.copy()
        comprehensive_results["Analysis_Date"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        comprehensive_results["Selected_Analyser"] = selected_analyser
        comprehensive_results["Units"] = units
        comprehensive_results["Regression_Model"] = regression_model
        comprehensive_results["Alpha"] = alpha
        comprehensive_results["R_Squared"] = r_squared
        comprehensive_results["Slope"] = slope
        comprehensive_results["Intercept"] = intercept
        
        summary_stats = {
            "Total_Points": len(comprehensive_results),
            "Regression_Model": regression_model,
            "Alpha": alpha,
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
            title=f"Residual Plot ({regression_model})",
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
    
    with st.expander("🎯 Outlier Detection", expanded=False):      
        residual_mean = np.mean(residuals)
        residual_std = np.std(residuals)
        outlier_threshold = 3 * residual_std
        
        outliers = np.abs(residuals - residual_mean) > outlier_threshold
        outlier_count = np.sum(outliers)
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"Using a **3-standard deviation** threshold ({outlier_threshold:.4f})")
            if outlier_count > 0:
                st.warning(f"Found **{outlier_count}** potential outlier(s).")
            else:
                st.success("No potential outliers detected based on this criteria.")

        with col2:
            if outlier_count > 0:
                outlier_data = clean_df[outliers]
                st.dataframe(outlier_data, use_container_width=True)
            else:
                st.write("No data to display.")

        st.info("""
            **Note on Outliers:** Outlier detection is based on the residuals being more than three standard deviations away from the mean residual. This is a common rule-of-thumb but may not be suitable for all datasets. Further investigation of these points is recommended.
        """)
