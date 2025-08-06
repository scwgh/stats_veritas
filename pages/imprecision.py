import streamlit as st
import pandas as pd
import numpy as np
import os
from scipy import stats
from scipy.stats import f, ttest_ind
from scipy.stats import f_oneway
from statsmodels.stats.oneway import anova_oneway
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from itertools import combinations
from datetime import datetime
from utils import apply_app_styling, check_westgard_rules, grubbs_test, units_list

st.set_page_config(
    page_title="Imprecision",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_app_styling()

st.header("📊 Imprecision Analysis (with Westgard and Grubbs' tests)")

with st.expander("📘 What is Imprecision Analysis?", expanded=True):
    st.markdown("""
    Imprecision analysis is used to evaluate random error associated with a measurement. Measurement of laboratory analytical error falls into two main categories: \n "***systematic error***" and "***random error***". 
    \n **Systematic errors** are predictable problems influencing observations consistently in one direction, while **random errors** are more unpredictable. Systematic errors are typically demonstrated  by bias (for example, ongoing negative bias for a QC), while random errors by the imprecision measured by the coefficient of variation (CV, %) (for example, one measurement outside of the expected range with all other repeat measurements within range). 
    \n Imprecision affects the reproducibility and repeatability of results. Reproducibility is defined as the closeness of the results of successive measurements under changed conditions. Repeatability is the closeness of the results of at least twenty successive measurements under similar conditions. By contrast, bias is the average deviation from a true value with minimal contribution of imprecision while inaccuracy is the deviation of a single measurement from the true value with significant contribution by imprecision. Multiple measurements, at least twenty and preferably forty, are therefore required for calculating imprecision as well as bias.
                
    \n ****What types of imprecision are we interested in?****
    - **Intra-well imprecision**: Variation in repeated measurements within a single well or sample.
    - **Intra-batch imprecision**: Variation in repeated measurements within a single batch.
    - **Inter-batch imprecision**: Variation in measurements of the same sample across different batches run across days.
                
    \n As a general rule-of-thumb, if the nominal bias between instruments is smaller than the inter-analyser CV (%), and the p-value is not significant (i.e., > 0.05), then there is no strong evidence that the instruments differ in a meaningful or consistent way. Inter-analyser %CV you calculate is based on the standard deviation of mean values across analyzers for a given analyte-material pair. The associated p-value is derived from a one-way ANOVA, which tests whether the means from different analyzers are significantly different from each other. 

    \n > <small> ✅ **Low bias + High p-value:** No significant difference — acceptable agreement.</small>

    \n > <small> ⚠️ **Low bias + Low p-value:** Statistically significant difference, but it may not be clinically relevant — further investigation may be warranted.</small>

    \n > <small> ❌ **High bias + Low p-value:** Significant and possibly clinically relevant difference — calibration or methodological review required.</small>

    """, unsafe_allow_html=True)

with st.expander("📘 Imprecision Metrics Explained:", expanded=False):
    st.markdown("###### 📐 Imprecision Metrics Explained")
    st.markdown("**🔹 Standard Deviation (SD)**: A measure of the dispersion of individual measurements around the mean. A low standard deviation suggests values are close to the mean; while a high standard deviation indicates values are spread over a wider range. SD is often used to determine if a value is an outlier.")
    st.latex(r'''\text{SD} = \sqrt{ \frac{1}{n - 1} \sum_{i=1}^{n} (x_i - \bar{x})^2 }''')
    st.markdown("**🔹 Coefficient of Variation (CV)**: The coefficient of variation (CV) is defined as a ratio of standard deviation to mean. CV should only be used for data which has a meaningful zero and therefore can be applied as a relative comparison between two measurements. It may also be defined as the standard deviation expressed as a percentage of the mean.")
    st.latex(r'''\text{CV}(\%) = \left( \frac{\text{SD}}{\bar{x}} \right) \times 100''')
    st.markdown("**🔹 Bias**: Bias is expressed as a percentage of the mean measured value versus the expected value.")
    st.latex(r'''\text{Bias (\%)} = \left( \frac{\bar{x} - \mu}{\mu} \right) \times 100''')
    tab_westgard, tab_grubbs = st.tabs(["Westgard Rules", "Grubbs' Test"])

    with tab_westgard:
        subtab_expl, subtab_violations = st.tabs(["🤔 What are Westgard Rules?", "❌ Westgard Rules Violations"])
        
        with subtab_expl:
            st.markdown("""
            ##### 🤔 What are Westgard Rules?
            - The **Westgard Rules** are a set of statistical criteria used to monitor analytical performance and detect potential errors in quality control (QC) data. Each rule examines patterns or outliers based on the mean and standard deviation (SD) of control measurements. 
            - The app allows you to apply Westgard rules to your data and visualize any violations using a red cross (❌).
            - The rules are applied to the selected analyte and will be displayed in the plot.
            - You can choose which rules to apply using the checkboxes in the sidebar.
            """)

        with subtab_violations:
            st.markdown("""
            \n ⚠️ **Warning Rule**
            \n- **1₂s**: One control result exceeds **±2 SD** from the mean. This is considered a **warning**, not a rejection.
            \n ❌ **Rejection Rules**
            \n - **1₃s**: One control result exceeds **±3 SD**. This suggests a **random error**.
            \n - **2₂s**: Two consecutive control results exceed **±2 SD in the same direction**. Indicates **systematic error**.
            \n - **R₄s**: One control result exceeds the mean by +2 SD and the next by -2 SD (or vice versa), giving a **range > 4 SD**. Detects **random error**.
            \n - **4₁s**: Four consecutive results exceed **±1 SD in the same direction**. Indicates **systematic shift**.
            \n - **10x*:*: Ten consecutive results are on the **same side of the mean**. Flags **bias** or drift.
            \n - **7T**: Seven points show a **consistent trend**, either increasing or decreasing. Suggests **progressive change**.
            \n - **8x**: Eight consecutive points lie on the **same side of the mean**. Another test for **systematic error**.
            """)
    with tab_grubbs:
        st.markdown("""
        ###### 🟪 Grubbs’ Test for outlier identification:
        - **Grubb's test**, also known as Extreme Studentized Deviate (ESD) test, is used to identify outliers which are statistically significant from a univariate data set.
        - The primary assumption of the Grubbs' test is  that the data is normally distributed. When applying a Grubbs' test, the null hypothesis states that there are no outliers in the dataset. 
        - The G statistic is calculated as the ratio of the maximum absolute deviation from the mean to the standard deviation of the sample. If this G statistic exceeds a critical value from the Grubbs' distribution table, the point is considered a significant outlier.
        """)
        st.latex(r"""
            G = \frac{\max{\left| x_i - \bar{x} \right|}}{s}
            """)
            
        st.markdown("""
        - The value of G is compared to a critical value from the Grubbs' distribution table. If G exceeds the critical value, the point is considered a significant outlier.
        - Any outliers which are flagged using the test here will be marked with a purple square (🟪). Tick the box to exclude outliers - any outliers identified will be removed from subsequent calculations and intra-well, intra-batch, and inter-batch imprecision will be recalculated.
        """)
    # with tab_welch:
    #     st.markdown("""
    #     ##### 📝 Welch's One-Way ANOVA:
    #     - Welch's ANOVA is a statistical test used to determine if there are significant differences between the means of three or more groups that may have different variances.
    #     - The app will perform Welch's ANOVA on the selected analyte and display the results.
    #     - The results will include the F-statistic and p-value, which indicate whether there are significant differences between the means of the groups.
    #     - A p-value less than 0.05 indicates a statistically significant difference between the means of the groups.
    #     """)

# --- Instructions ---
with st.expander("📘 Instructions:", expanded=False):
    
    st.markdown("""
    This tool allows you to assess **intra-well, intra-batch, and inter-batch imprecision** across different levels of control or patient materials.

    1. **Upload your CSV file** – it should contain repeated measurements for the same sample/material across different runs or days.
    2. Your file should include the following columns:
    - `Date` – can be in short or long date format
    - `Test` (e.g., Intra_Batch_Imprecision, Inter_Batch_Imprecision) - please ensure underscores are included. 
    - `Analyser` (e.g., Waters TQ-D 1, Waters TQ-D 2)
    - `Material` (e.g., QC1, QC2, QC3)
    - `Sample ID` (e.g., 12345, 67890)
    - `Batch ID` (e.g., Test_Batch_123)
    - One or more **analyte columns** – ensure consistent naming and avoid use of special characters. You DO NOT need to include units in the column names.

    3. After upload, the app will:
    - Group data by `Material`, `QC Level`, and `Analyte`
    - Calculate intra-well, intra-batch, and inter-batch imprecision (if data is available).
    - Generate summary statistics and visualizations for each analyte

    4. **Select the analyte** you want to analyze from the dropdown menu.
    - The app will filter the data accordingly and display the results.
    - Use the toggle in the sidebar to enable or disable rule overlays, including Westgard rules and outlier identification using Grubbs` test.
    """)


def precision_studies(df, selected_analyte, rules_enabled, grubbs_outliers, exclude_westgard=False, units=units_list, data_type='QC'):
    results = []
    outlier_indices = []
    filtered_data = df.copy() 
    
    # Filter data based on selected data type
    if data_type == 'QC':
        qc_df = df[df['Material'].str.contains('QC', na=False)]
    elif data_type == 'Standard':
        qc_df = df[df['Material'].str.contains('Standard', na=False) | df['Material'].str.contains('STD', na=False)]
    else:
        # If 'Both' is selected, include all data
        qc_df = df.copy()

    # Filter for Inter_Batch_Imprecision data only for plotting
    inter_batch_data = qc_df[qc_df['Test'] == 'Inter_Batch_Imprecision'].copy()
    
    # Filter for the selected analyte and remove any rows with NaN values
    inter_batch_data = inter_batch_data.dropna(subset=[selected_analyte])
    
    # Group by Material, Sample ID, and Analyser for plotting
    plot_groups = list(inter_batch_data.groupby(['Material', 'Sample ID', 'Analyser']))
    
    # Check if we have any data to plot
    if not plot_groups:
        st.info("No Inter-Batch data available to plot for the selected analyte.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), [], filtered_data, []

    num_plots = len(plot_groups)
    num_cols = 2
    num_rows = (num_plots + num_cols - 1) // num_cols  # Ceiling division

    # Calculate appropriate vertical spacing
    if num_rows > 1:
        max_allowed_spacing = 0.9 / (num_rows - 1)
        if num_rows > 10:
            vertical_spacing = min(0.02, max_allowed_spacing)
        elif num_rows > 5:
            vertical_spacing = min(0.05, max_allowed_spacing)
        else:
            vertical_spacing = min(0.15, max_allowed_spacing)
    else:
        vertical_spacing = 0.15

    # Create subplot titles
    subplot_titles = [f"{material} - {sample_id} - {analyzer}" 
                     for (material, sample_id, analyzer), _ in plot_groups]

    # Create subplots
    fig = make_subplots(
        rows=num_rows,
        cols=num_cols,
        subplot_titles=subplot_titles,
        shared_xaxes=False,  # Each subplot has its own x-axis
        shared_yaxes=False,  # Each subplot has its own y-axis
        horizontal_spacing=0.08,
        vertical_spacing=vertical_spacing,
        specs=[[{"secondary_y": False} for _ in range(num_cols)] for _ in range(num_rows)]
    )

    # Plot each group in its own subplot
    for idx, ((material, sample_id, analyzer), group) in enumerate(plot_groups):
        row = (idx // num_cols) + 1
        col = (idx % num_cols) + 1

        # Prepare the data for this specific group
        group = group.copy()
        group['Date'] = pd.to_datetime(group['Date'], errors='coerce', dayfirst=True)
        group = group.dropna(subset=['Date', selected_analyte])

        if group.empty or len(group) < 2:
            continue

        # Sort by date for proper line connections
        group = group.sort_values('Date')

        # Calculate statistics for this group
        overall_mean = group[selected_analyte].mean()
        sd = group[selected_analyte].std()

        # Show legend only for the first subplot (top-right)
        show_legend = (idx == 0)

        # Add sample points
        fig.add_trace(go.Scatter(
            x=group['Date'], 
            y=group[selected_analyte], 
            mode='markers',
            marker=dict(color='darkblue', size=8, opacity=0.8),
            name='Sample', 
            showlegend=show_legend, 
            legendgroup='sample_points',
            customdata=np.stack((
                group['Sample ID'], 
                group['Date'].dt.strftime('%Y-%m-%d'), 
                group[selected_analyte]
            ), axis=-1),
            hovertemplate=(
                "Sample ID: %{customdata[0]}<br>" +
                "Date: %{customdata[1]}<br>" +
                f"{selected_analyte}: %{{customdata[2]:.2f}}<extra></extra>"
            )
        ), row=row, col=col)

        # Add mean line
        fig.add_trace(go.Scatter(
            x=group['Date'], 
            y=[overall_mean] * len(group),
            mode='lines', 
            line=dict(color='green', width=2),
            name=f'Mean', 
            showlegend=show_legend, 
            legendgroup='mean_line',
            hovertemplate=f"Mean: {overall_mean:.2f}<extra></extra>"
        ), row=row, col=col)

        # Add +2 SD line
        fig.add_trace(go.Scatter(
            x=group['Date'], 
            y=[overall_mean + (2*sd)] * len(group),
            mode='lines', 
            line=dict(color='orange', dash='dash', width=1.5),
            name='+2 SD', 
            showlegend=show_legend, 
            legendgroup='control_limits',
            hovertemplate=f"+2 SD: {overall_mean + (2*sd):.2f}<extra></extra>"
        ), row=row, col=col)

        # Add -2 SD line
        fig.add_trace(go.Scatter(
            x=group['Date'], 
            y=[overall_mean - (2*sd)] * len(group),
            mode='lines', 
            line=dict(color='darkorange', dash='dot', width=1.5),
            name='-2 SD', 
            showlegend=show_legend, 
            legendgroup='control_limits',
            hovertemplate=f"-2 SD: {overall_mean - (2*sd):.2f}<extra></extra>"
        ), row=row, col=col)

        # Add +3 SD line
        fig.add_trace(go.Scatter(
            x=group['Date'], 
            y=[overall_mean + (3*sd)] * len(group),
            mode='lines', 
            line=dict(color='red', dash='dot', width=1.5),
            name='+3 SD', 
            showlegend=show_legend, 
            legendgroup='control_limits',
            hovertemplate=f"+3 SD: {overall_mean + (3*sd):.2f}<extra></extra>"
        ), row=row, col=col)

        # Add -3 SD line
        fig.add_trace(go.Scatter(
            x=group['Date'], 
            y=[overall_mean - (3*sd)] * len(group),
            mode='lines', 
            line=dict(color='red', dash='dot', width=1.5),
            name='-3 SD', 
            showlegend=show_legend, 
            legendgroup='control_limits',
            hovertemplate=f"-3 SD: {overall_mean - (3*sd):.2f}<extra></extra>"
        ), row=row, col=col)

        # Apply Westgard Rules if enabled
        if any(rules_enabled.values()):
            westgard_violations = []
            rule_alerts = check_westgard_rules(group[selected_analyte].tolist(), overall_mean, sd, rules_enabled)

            for i, rule in rule_alerts:
                westgard_violations.append(i)
                fig.add_trace(go.Scatter(
                    x=[group['Date'].iloc[i]],
                    y=[group[selected_analyte].iloc[i]],
                    mode='markers',
                    marker=dict(color='crimson', size=12, symbol='x'),
                    name=f'Westgard Violation',
                    showlegend=show_legend and len(westgard_violations) == 1,  # Show legend only once
                    legendgroup='westgard_violations',
                    customdata=[[rule, group.iloc[i]['Sample ID'], group.iloc[i]['Date'].strftime('%Y-%m-%d'), group.iloc[i][selected_analyte]]],
                    hovertemplate=(
                        "Westgard Alert: %{customdata[0]}<br>" +
                        "Sample ID: %{customdata[1]}<br>" +
                        "Date: %{customdata[2]}<br>" +
                        f"{selected_analyte}: %{{customdata[3]:.2f}}<extra></extra>"
                    )
                ), row=row, col=col)

        # Apply Grubbs' Test if enabled
        if grubbs_outliers.get("perform_grubbs"):
            outlier_alerts = grubbs_test(group[selected_analyte])
            grubbs_outliers_indices = outlier_alerts["Outlier Indices"]
            # st.info(f"🟪 Grubbs' Test applied to {material} [{analyzer}]. No outliers identified.")

            for outlier_idx in grubbs_outliers_indices:
                if outlier_idx in group.index:
                    row_data = group.loc[outlier_idx]
                    fig.add_trace(go.Scatter(
                        x=[row_data['Date']],
                        y=[row_data[selected_analyte]],
                        mode='markers',
                        marker=dict(color='darkorchid', size=12, symbol='square', line=dict(width=2, color='darkorchid')),
                        name="Grubbs' Outlier",
                        showlegend=show_legend and outlier_idx == grubbs_outliers_indices[0],
                        legendgroup='grubbs_outliers',
                        customdata=[[row_data['Sample ID'], row_data['Date'].strftime('%Y-%m-%d'), row_data[selected_analyte]]],
                        hovertemplate=(
                            "Grubbs' Outlier<br>" +
                            "Sample ID: %{customdata[0]}<br>" +
                            "Date: %{customdata[1]}<br>" +
                            f"{selected_analyte}: %{{customdata[2]:.2f}}<extra></extra>"
                        ),
                    ), row=row, col=col)
            if grubbs_outliers_indices:
                outlier_details = [
                    f"{selected_analyte} = {group.loc[idx, selected_analyte]} ({group.loc[idx, 'Material']})"
                    for idx in grubbs_outliers_indices
                ]
                st.error(f"❗ Grubbs' Test applied to {material} [{analyzer}]. {len(grubbs_outliers_indices)} outlier(s): {' | '.join(outlier_details)} identified.")

                if grubbs_outliers.get("exclude_grubbs"):
                    group = group.drop(index=grubbs_outliers_indices)

        # Update axes for this subplot with auto-range
        fig.update_xaxes(
            title_text="Date",
            showticklabels=True,
            tickangle=45,
            tickformat="%d-%m-%Y",
            autorange=True,  # Let each subplot auto-scale its x-axis
            row=row, 
            col=col
        )
        
        fig.update_yaxes(
            title_text=f"{selected_analyte} ({units})",
            autorange=True,  # Let each subplot auto-scale its y-axis
            row=row, 
            col=col
        )

    # Update overall layout
    plot_height = max(400 * num_rows, 600)  # Minimum height of 600px
    plot_height = min(plot_height, 8000)   # Maximum height of 8000px
    
    fig.update_layout(
        height=plot_height,
        title_text=f"{selected_analyte} - Inter-Batch Imprecision Analysis",
        title_x=0.35,
        template='plotly_white',
        margin=dict(t=80, l=60, r=60, b=80),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,  # Move legend just outside the plot area
            bgcolor="rgba(255,255,255,0.7)",
            borderwidth=1
        )
    )

    # Force each subplot to have independent axis ranges
    for i in range(1, num_rows + 1):
        for j in range(1, num_cols + 1):
            if (i-1) * num_cols + j <= num_plots:  # Only update existing subplots
                fig.update_xaxes(
                    autorange=True,
                    fixedrange=False,
                    row=i, 
                    col=j
                )
                fig.update_yaxes(
                    autorange=True,
                    fixedrange=False,
                    row=i, 
                    col=j
                )

    st.plotly_chart(fig, use_container_width=True)

    # import base64
    # fig_html = fig.to_html(include_plotlyjs='cdn', full_html=True)
    # b64 = base64.b64encode(fig_html.encode()).decode()
    # href = f'<a href="data:text/html;base64,{b64}" target="_blank">🔗 Open Chart in New Window</a>'
    # st.markdown(href, unsafe_allow_html=True)

    # --- Imprecision Calculations ---
    excluded_indices = set()
    grubbs_excluded_indices = set()
    analyzer_means = {}

    for analyte in df.columns[7:]:
        for (material, analyzer, test, sample_id), group in qc_df.groupby(['Material', 'Analyser', 'Test', 'Sample ID']):
            group = group.copy()

            if group.empty or len(group) < 2:
                continue

            # Calculate initial mean and SD BEFORE exclusion
            initial_mean = group[analyte].mean()
            initial_sd = group[analyte].std()

            # Apply Westgard Exclusion FIRST (before Grubbs')
            if rules_enabled and exclude_westgard:
                westgard_violations = [i for i, _ in check_westgard_rules(group[analyte].tolist(), initial_mean, initial_sd, rules_enabled)]
                if westgard_violations:
                    excluded_indices.update(group.index[westgard_violations])
                    group = group.drop(index=group.index[westgard_violations])

            if group.empty or len(group) < 2:
                continue

            # Apply Grubbs' test and optionally exclude outliers
            if grubbs_outliers.get("perform_grubbs"):
                outlier_alerts = grubbs_test(group[analyte])
                outlier_indices = outlier_alerts["Outlier Indices"]
                if grubbs_outliers.get("exclude_grubbs") and outlier_indices:
                    grubbs_excluded_indices.update(outlier_indices)
                    group = group.drop(index=outlier_indices)
            
            if group.empty or len(group) < 2:
                continue 

            # Recalculate mean and SD after exclusions
            overall_mean = round(group[analyte].mean(), 2)
            sd = round(group[analyte].std(), 2)
            nobs = group[analyte].count()
            cv = round((sd / overall_mean) * 100, 2) if overall_mean != 0 else np.nan
            sem = round(sd / np.sqrt(nobs), 2) if nobs != 0 else np.nan

            results.append({
                'Test': test,
                'Analyte': analyte,
                'n': nobs,
                'Analyser': analyzer,
                'Material': material,
                'Sample ID': sample_id,
                f'Mean ({units})': overall_mean,
                'SD': sd,
                'CV (%)': cv,
                'SEM': sem
            })

            if test == "Inter_Batch_Imprecision":
                key = (analyte, material, sample_id)
                analyzer_means.setdefault(key, {})[analyzer] = overall_mean

    # -- Inter-analyser differences --
    analyser_comparison = []
    differences = []

    all_excluded_indices = excluded_indices.union(grubbs_excluded_indices)
    filtered_qc_df = qc_df[~qc_df.index.isin(all_excluded_indices)]

    filtered_qc_df = qc_df[~qc_df.index.isin(excluded_indices)]
    for (analyte, material, sample_id), means_dict in analyzer_means.items():
        analyzers = list(means_dict.keys())
        if len(analyzers) < 2:
            continue

        analyzer_values = [
            filtered_qc_df[
                (filtered_qc_df['Material'] == material) & 
                (filtered_qc_df['Sample ID'] == sample_id) &
                (filtered_qc_df['Analyser'] == analyzer)
            ][analyte].dropna()
            for analyzer in analyzers
        ]

        analyzer_values = [vals for vals in analyzer_values if len(vals) >= 2]

        f_statistic = np.nan
        f_critical = np.nan
        p_value = np.nan

        if len(analyzer_values) == 2:
            group1, group2 = analyzer_values
            var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
            n1, n2 = len(group1), len(group2)

            if var1 >= var2:
                f_statistic = var1 / var2
                df1, df2 = n1 - 1, n2 - 1
            else:
                f_statistic = var2 / var1
                df1, df2 = n2 - 1, n1 - 1

            p_value = 1 - f.cdf(f_statistic, df1, df2)
            alpha = 0.05
            f_critical = f.ppf(1 - alpha, df1, df2)

        diffs = []
        for a1, a2 in combinations(analyzers, 2):
            m1, m2 = means_dict[a1], means_dict[a2]
            if m1 and m2:
                pct_diff = abs(m1 - m2) / ((m1 + m2) / 2) * 100
                diffs.append(pct_diff)

        mean_values = [means_dict[a] for a in analyzers]
        ia_mean = np.mean(mean_values)
        ia_sd = np.std(mean_values, ddof=1)
        ia_cv = (ia_sd / ia_mean) * 100 if ia_mean != 0 else np.nan
        ia_biases = [abs(((means_dict[analyzer] - ia_mean) / ia_mean) * 100) for analyzer in analyzers]
        mean_ia_bias = round(np.mean(ia_biases), 2)
        mean_a1 = means_dict.get(analyzers[0], np.nan)
        mean_a2 = means_dict.get(analyzers[1], np.nan)
        formatted_p_value = p_value if not np.isnan(p_value) else np.nan
        
        analyser_comparison.append({
            'Analyte': analyte,
            'Material': material,
            'Sample ID': sample_id, 
            f'Mean {analyzers[0]} ({units})': round(mean_a1, 2) if not np.isnan(mean_a1) else np.nan,
            f'Mean {analyzers[1]} ({units})': round(mean_a2, 2) if not np.isnan(mean_a2) else np.nan,
            f'Inter-Analyser Mean ({units})': round(ia_mean, 2),
            'Inter-Analyser SD': round(ia_sd, 2),
            'Inter-Analyser CV (%)': round(ia_cv, 2), 
            'Mean % Difference Between Analysers': round(mean_ia_bias, 2),
            'F-statistic': round(f_statistic, 4) if not np.isnan(f_statistic) else np.nan,
            'F Critical (0.05 one-tail)': round(f_critical, 4) if not np.isnan(f_critical) else np.nan,
            'P(F <= f) one-tail': round(formatted_p_value, 4) if not np.isnan(formatted_p_value) else np.nan,
            'Significant (p < 0.05)': 'Yes' if not np.isnan(formatted_p_value) and formatted_p_value < 0.05 else 'No'
        })
        
        differences.append({
            'Analyte': analyte,
            'Material': material,
            'Sample ID': sample_id, 
            'Mean % Difference Between Analysers': round(np.mean(diffs), 2)
        })

    stats_df = pd.DataFrame(results)
    diff_df = pd.DataFrame(differences)

    return (
        stats_df[stats_df['Test'] == 'Intra_Well_Imprecision'],
        stats_df[stats_df['Test'] == 'Intra_Batch_Imprecision'],
        stats_df[stats_df['Test'] == 'Inter_Batch_Imprecision'],
        analyser_comparison,
        filtered_data, 
        outlier_indices 
    )

# --- File Upload ---
with st.expander("📤 Upload Your CSV File", expanded=True):
    uploaded_file = st.file_uploader("   ", type=["csv"])
    
if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            st.stop()
    except Exception as e:
        st.error(f"Error reading the file: {e}")
        st.stop()

    st.subheader("📖 Data Preview")
    st.dataframe(df.head(5), use_container_width=True)
    required_columns = ['Material', 'Sample ID', 'Analyser', 'Test']
    if not all(col in df.columns for col in required_columns):
        st.error(f"The uploaded file must contain the columns: {', '.join(required_columns)}.")
        st.stop()

    if len(df.columns) <= 5:
        st.warning("❗️ Not enough analyte columns detected.")
    else:
        analyte_options = df.columns[7:]

        data_type = st.selectbox("🔍 Select Data Type", 
                options=['QC', 'Standard', 'Both'])
        
        selected_analyte = st.selectbox("🔎 Select Analyte to View", analyte_options)

        units = st.selectbox(
            "Select Units",
            options=units_list, 
            index=0
        )

        # --- Westgard & Grubbs Controls Section ---
        with st.expander("⚙️ Settings: Westgard Rules & Outlier Detection", expanded=True):
            tab1, tab2 = st.tabs(["❌ Westgard Rules", "🟪 Perform Grubbs` Test"])
            with tab1:
                col1, col2, col3 = st.columns(3)
                with col1:
                    rule_1_2s = st.checkbox("❌ 1-2s (warning)", value=True)
                    rule_2_2s = st.checkbox("❌ 2-2s", value=False)
                    rule_4_1s = st.checkbox("❌ 4-1s", value=False)
                    rule_7T = st.checkbox("❌ 7T (trend)", value=False)
                with col2:
                    rule_1_3s = st.checkbox("❌ 1-3s", value=False)
                    rule_R_4s = st.checkbox("❌ R-4s", value=False)
                    rule_10x = st.checkbox("❌ 10x", value=False)
                    rule_8x = st.checkbox("❌8x", value=False)
                with col3:
                    exclude_westgard = st.checkbox("🚫 Exclude Westgard rule violations from calculations", value=False)
                    st.markdown("**Note:** If you choose to exclude Westgard rule violations, the app will remove any data points that violate the selected rules before performing the analysis.")
            with tab2:
                perform_grubbs = st.checkbox("🟪 Perform Grubbs' test to identify outliers", value=False)
                exclude_grubbs = False
                if perform_grubbs:
                    exclude_grubbs = st.checkbox("🚫 Exclude outliers from calculations", value=False)

            rules_enabled = {
                '1_2s': rule_1_2s,
                '1_3s': rule_1_3s,
                '2_2s': rule_2_2s,
                'R_4s': rule_R_4s,
                '4_1s': rule_4_1s,
                '10x': rule_10x,
                '7T': rule_7T,
                '8x': rule_8x
            }

            grubbs_outliers = {
                "perform_grubbs": perform_grubbs,
                "exclude_grubbs": exclude_grubbs
            }

        with st.spinner("Analyzing..."):
            intra_well_df, intra_batch_df, inter_batch_df, analyser_comparison, filtered_data, outlier_indices = precision_studies(df, selected_analyte, rules_enabled, grubbs_outliers, units=units, data_type=data_type)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")

        # --- Results Output  ---
        with st.expander("📊 View Results", expanded=True):
            tab1, tab2, tab3 = st.tabs(["Intra-Well", "Intra-Batch", "Inter-Batch"])
            with tab1:
                st.dataframe(intra_well_df)
            with tab2:
                st.dataframe(intra_batch_df)
            with tab3:
                st.dataframe(inter_batch_df)

            st.subheader("📏 Inter-Analyser Summary Statistics")
            inter_analyser_df = pd.DataFrame(analyser_comparison)
            st.dataframe(inter_analyser_df)

# --- Reference Section ---
with st.expander("📚 References"):
    st.markdown("""
    **Westgard, J.O., Barry, P.L., and Hunt, M.R. (1981)**, *A Multi-Rule Shewhart Chart for Quality Control in Clinical Chemistry*, Clinical Chemistry, 27 (3), pp.493-501
    (https://westgard.com/downloads/papers-downloads/27-westgard-rules-paper/file.html
    \n **Westgard, J.O., and Barry, P.L.** (1986) Improving Quality Control by use of Multirule Control Procedures. Chapter 4 in Cost-Effective Quality Control: Managing the quality and productivity of analytical processes. AACC Press, Washington, DC, 1986, pp.92-117
    \n **Westgard J.O., and Klee, G.G.** (1994) Quality Management. Chapter 17 in Textbook of Clinical Chemistry, 2nd edition. Burtis C, ed., WB Saunders Company, Philadelphia, pp.548-592.
    \n **Westgard J.O., and Klee, G.G.** (1996) Quality Management. Chapter 16 in Fundamentals of Clinical Chemistry, 4th edition. Burtis C, ed., WB Saunders Company, Philadelphia, 1996, pp.211-223.
    """)
