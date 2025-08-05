
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
apply_app_styling()

st.header("📈 Polynomial Regression")

# Documentation sections
with st.expander("📘 About Polynomial Regression", expanded=True):
    st.markdown("""
    **Polynomial Regression** is a form of regression analysis in which the relationship between the independent variable $x$ and the dependent variable $y$ is modeled as an $n^{th}$-degree polynomial. It's a powerful technique used when a linear model is not sufficient to capture the relationship in the data.

    The model is still considered a type of linear regression because it is linear in its coefficients, even though it fits a non-linear curve to the data. The general equation is:
    """)
    st.latex(r''' y = \beta_0 + \beta_1 x + \beta_2 x^2 + \dots + \beta_n x^n + \epsilon ''')
    st.markdown("""
    Where $n$ is the **degree** of the polynomial. A degree of 1 corresponds to a simple linear regression. For higher degrees, the model can fit more complex curves. It is a critical aspect of method validation, particularly for assays that exhibit a non-linear response across their concentration range.
    """)
    
    st.markdown("""
    ---
    
    ##### How to choose the right degree?
    Selecting the appropriate polynomial degree is crucial to avoid **overfitting** (a model that is too complex and fits noise) or **underfitting** (a model that is too simple and misses the trend). You should generally choose the lowest degree that provides a good fit without excessive complexity.
    """)

with st.expander("📋 Instructions", expanded=False):
    st.markdown("""
    1. Upload a CSV file containing your standard curve data
    2. Ensure the file includes columns for concentration and response values
    3. Optionally include an identifier column for sample tracking
    4. Select the appropriate columns and units for analysis
    5. **Adjust the polynomial degree** to find the best-fit curve
    6. Review the results and download reports as needed
    """)

# Initialize variables
df = None
results_df = None
analysis_complete = False

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
    
    st.subheader("Model Configuration")
    polynomial_degree = st.slider("Select Polynomial Degree", min_value=1, max_value=5, value=2, 
                                  help="A degree of 1 is linear regression. Increase to fit more complex curves.")


# ================================
# 3. POLYNOMIAL REGRESSION ANALYSIS AND PLOTS
# ================================
if df is not None and len(df.select_dtypes(include=[np.number]).columns) >= 2:
    st.header("📈 Polynomial Regression Analysis")
    
    # Prepare data
    clean_df = df[[x_axis, y_axis] + ([identifier_column] if identifier_column else []) + (["Date"] if has_date_column else [])].dropna()
    
    if clean_df.empty:
        st.error("❌ No valid data found in selected columns")
        st.stop()
    
    # Perform polynomial regression
    x = clean_df[x_axis].to_numpy()
    y = clean_df[y_axis].to_numpy()
    
    try:
        # Calculate overall statistics (used for ungrouped analysis)
        poly_coeffs = np.polyfit(x, y, polynomial_degree)
        poly_model = np.poly1d(poly_coeffs)
        fitted_values = poly_model(x)
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
                    
                    if len(date_data) >= polynomial_degree + 1:
                        date_x = date_data[x_axis].to_numpy()
                        date_y = date_data[y_axis].to_numpy()
                        
                        date_poly_coeffs = np.polyfit(date_x, date_y, polynomial_degree)
                        date_poly_model = np.poly1d(date_poly_coeffs)
                        date_fitted = date_poly_model(date_x)
                        date_residuals = date_y - date_fitted
                        date_r_squared = 1 - (np.sum(date_residuals**2) / np.sum((date_y - np.mean(date_y))**2))
                        
                        date_stats.append({
                            'date': date,
                            'r_squared': date_r_squared,
                            'n_points': len(date_data),
                            'coefficients': date_poly_coeffs
                        })
                        
                        color = colors[i % len(colors)]
                        
                        equation_str = " + ".join([f"{c:.2f}x^{d}" for d, c in enumerate(date_poly_coeffs[::-1])])
                        
                        fig.add_trace(go.Scatter(
                            x=date_x, y=date_y,
                            mode='markers',
                            name=f'Data ({date})',
                            marker=dict(color=color, size=8),
                            text=hover_text[date_mask] if hover_text is not None else None,
                            hoverinfo='text' if identifier_column else 'x+y'
                        ))
                        
                        # Fix: Ensure x_range is a numpy array
                        x_range = np.linspace(date_x.min(), date_x.max(), 100)
                        y_fitted_line = date_poly_model(x_range)
                        
                        fig.add_trace(go.Scatter(
                            x=x_range, y=y_fitted_line,
                            mode='lines',
                            name=f'Fit - {date} (R²={date_r_squared:.3f})',
                            line=dict(color=color, width=2, dash='solid'),
                            hoverinfo='skip'
                        ))
                
                plot_title = f"Standard Curve - Grouped by Date ({len(unique_dates)} dates)"
                
                if date_stats:
                    st.subheader("📊 Date Group Statistics")
                    date_stats_df = pd.DataFrame(date_stats)
                    date_stats_df['coefficients'] = date_stats_df['coefficients'].apply(lambda x: [round(c, 4) for c in x])
                    st.dataframe(date_stats_df.round(4), use_container_width=True)
                
            else:
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
                
                fig.add_trace(go.Scatter(
                    x=[None], y=[None],
                    mode='markers',
                    marker=dict(color='red', size=8, symbol='diamond'),
                    name='95% CI per Standard',
                    showlegend=True,
                    hoverinfo='skip'
                ))
                
                fig.add_trace(go.Scatter(
                    x=x, y=y,
                    mode='markers',
                    name='Data Points',
                    marker=dict(color='blue', size=8),
                    text=hover_text,
                    hoverinfo='text' if identifier_column else 'x+y'
                ))
                
                x_range = np.linspace(x.min(), x.max(), 500)
                fitted_sorted = poly_model(x_range)
                equation_str = " + ".join([f"{c:.4f}x^{d}" for d, c in enumerate(poly_coeffs[::-1])])
                
                fig.add_trace(go.Scatter(
                    x=x_range, y=fitted_sorted,
                    mode='lines',
                    name=f"Fit (deg {polynomial_degree}): y = {equation_str}<br>R² = {r_squared:.4f}",
                    line=dict(color='red', width=2)
                ))
                
                plot_title = f"Standard Curve with Polynomial Fit (Degree {polynomial_degree})"
            
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
            st.subheader("📈 Linearity")
            if r_squared >= 0.99:
                st.success("**Excellent** fit")
                st.write("The polynomial model provides a very strong fit to the data (R² ≥ 0.99).")
            elif r_squared >= 0.95:
                st.info("**Good** fit")
                st.write("Results are acceptable, but further verification may be considered (R² ≥ 0.95).")
            elif r_squared >= 0.90:
                st.warning("**Moderate** fit")
                st.write("Further investigation may be needed for accuracy (R² ≥ 0.90).")
            else:
                st.error("**Poor** fit")
                st.write("The model may not be reliable for quantitative analysis (R² < 0.90).")

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
    
    with st.expander("📊 Key Metrics", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("R² Value", f"{r_squared:.4f}")
            st.metric("Model Degree", polynomial_degree)
        
        with col2:
            st.metric("Total Points", total_points)
            st.metric("CI Coverage", f"{ci_percentage:.1f}%")
        
        with col3:
            st.metric("Points Outside CI", points_outside_count)
            st.metric("Mean Residual", f"{np.mean(residuals):.4f}")
        
        with col4:
            st.metric("Residual Std", f"{np.std(residuals):.4f}")
            
    st.markdown("---")
    st.subheader("Polynomial Coefficients")
    
    coefficients_df = pd.DataFrame({
        'Term': [f'x^{i}' for i in range(polynomial_degree, -1, -1)],
        'Coefficient': poly_coeffs
    }).reset_index(drop=True)
    st.dataframe(coefficients_df.round(6), use_container_width=True)

    with st.expander("📋 Detailed Results Preview", expanded=False):
        st.subheader("📊 Residuals and 95% Confidence Intervals Assessment")
        results_df = clean_df.copy()
        
        if identifier_column:
            results_df = results_df[[identifier_column, x_axis, y_axis] + (["Date"] if has_date_column else [])].copy()
        else:
            results_df = results_df[[x_axis, y_axis] + (["Date"] if has_date_column else [])].copy()
            
        results_df["Fitted Value"] = fitted_values
        results_df["Residuals"] = residuals

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
                
                if len(date_x) >= polynomial_degree + 1 and len(date_y) >= polynomial_degree + 1:
                    min_len = min(len(date_x), len(date_y))
                    date_x = date_x.iloc[:min_len]
                    date_y = date_y.iloc[:min_len]
                    
                    date_poly_coeffs = np.polyfit(date_x, date_y, polynomial_degree)
                    date_poly_model = np.poly1d(date_poly_coeffs)
                    date_fitted = date_poly_model(date_x)
                    date_residuals = date_y - date_fitted
                    date_r2 = 1 - (np.sum(date_residuals**2) / np.sum((date_y - np.mean(date_y))**2))
                    
                    row_data = {
                        "Date": date,
                        "R²": round(date_r2, 4),
                        "Coefficients": [round(c, 4) for c in date_poly_coeffs]
                    }
                    
                    deviation_rows.append(row_data)
            
            if deviation_rows:
                deviation_df = pd.DataFrame(deviation_rows)
                st.dataframe(deviation_df, use_container_width=True)
            else:
                st.warning("No sufficient data for deviation assessment. Make sure each date group has at least `degree + 1` data points.")
                
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
        
        # Check if the dataframe has enough columns for recovery analysis
        if len(df.columns) < 2:
            st.warning("Not enough columns in the dataset for recovery analysis.")
            st.stop()
        
        # Slicing the dataframe might be unsafe, better to use all columns
        selectable_columns = df.columns.tolist()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Let the user select columns from all available columns
            expected_column = st.selectbox("Expected Concentration Column", selectable_columns, key="expected_col")
        
        with col2:
            calculated_column = st.selectbox("Calculated Concentration Column", selectable_columns, key="calculated_col")
        
        if expected_column and calculated_column and expected_column != calculated_column:
            recovery_df = df.copy()
            # Ensure "Sample ID" exists before accessing it
            recovery_df["Sample_ID"] = recovery_df[identifier_column] if identifier_column else "N/A"
            recovery_df["Expected"] = recovery_df[expected_column]
            recovery_df["Calculated"] = recovery_df[calculated_column]
            
            recovery_df["Recovery (%)"] = np.where(
                recovery_df["Expected"] != 0,
                (recovery_df["Calculated"] / recovery_df["Expected"]) * 100,
                np.nan
            )
            
            show_summary = st.checkbox("Show Summary by Sample")
            
            if show_summary and identifier_column:
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
            st.warning("Please select two different columns for recovery calculation.")


# ================================
# 6. DOWNLOAD SECTION
# ================================
if analysis_complete:     
    st.markdown("  ")
    with st.expander("📁 Download Options", expanded=True):
        comprehensive_results = results_df.copy()
        comprehensive_results["Analysis_Date"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        comprehensive_results["Selected_Analyser"] = selected_analyser
        comprehensive_results["Units"] = units
        comprehensive_results["R_Squared"] = r_squared
        comprehensive_results["Model_Degree"] = polynomial_degree
        
        for i, coeff in enumerate(poly_coeffs):
            comprehensive_results[f"Coefficient_{i}"] = coeff
        
        summary_stats = {
            "Total_Points": len(comprehensive_results),
            "R_Squared": r_squared,
            "Model_Degree": polynomial_degree,
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
                file_name=f"polynomial_regression_detailed_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                help="Complete results with fitted values and residuals"
            )
        
        with col2:
            summary_report = pd.DataFrame([summary_stats])
            st.download_button(
                label="📋 Download Summary Report",
                data=summary_report.to_csv(index=False).encode('utf-8'),
                file_name=f"polynomial_regression_summary_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                help="Summary statistics and key metrics"
            )
        
        with col3:
            ci_summary_df = pd.DataFrame(ci_data)
            st.download_button(
                label="📈 Download CI Details",
                data=ci_summary_df.to_csv(index=False).encode('utf-8'),
                file_name=f"polynomial_regression_ci_details_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
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
        
        fig_residual.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Zero Line")
        
        fig_residual.update_layout(
            title="Residual Plot",
            xaxis_title="Fitted Values",
            yaxis_title="Residuals",
            height=500
        )
        
        st.plotly_chart(fig_residual, use_container_width=True)
        st.info("Residuals should be randomly distributed around zero for a good fit. Patterns (e.g., a 'U' shape) may indicate the model is not appropriate.")

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
            
            st.dataframe(outlier_data[display_columns].round(4), use_container_width=True)