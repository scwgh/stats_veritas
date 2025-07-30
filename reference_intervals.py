import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm, shapiro, jarque_bera, anderson, gaussian_kde
import plotly.graph_objects as go
import plotly.express as px
from utils import apply_app_styling
from plotly.subplots import make_subplots
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# --- Page Setup ---
st.set_page_config(
    page_title="Reference Interval Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_app_styling()

st.markdown('<h1 class="main-header">📊 Reference Interval Analysis</h1>', unsafe_allow_html=True)

with st.expander("📘 What is Reference Interval Analysis?", expanded=True):
    st.markdown("""
    This page is designed for clinical laboratories and researchers to calculate **reference intervals** from healthy population data in compliance with **CLSI C28-A3 guidelines**.

    A **reference interval** defines the central range of expected values—typically the **2.5th to 97.5th percentiles**—within a healthy population, serving as a critical benchmark for diagnostic interpretation. This tool supports both **non-parametric** (percentile-based) and **parametric** (normal distribution-based) estimation methods, offering flexibility and robustness in diverse data scenarios.
        
    #### ✅ Key Capabilities
    - **Interactive Visualizations** for distributions, normality, and Q-Q plots
    - **Normality Assessment** using Shapiro-Wilk, Jarque-Bera, Anderson-Darling, skewness, and kurtosis
    - **Outlier Detection** (3SD, IQR, Grubbs' test) with optional exclusion
    - **Age-Partitioned Intervals**: Auto-generated age groups with user-defined group sizes
    - **Sex-Partitioned Intervals**: Stratified intervals for different sex groups
    - **Quality Scoring**: Classifies results based on sample size (e.g., Excellent, Good, Adequate, Insufficient)
    - **Summary Reporting** with downloadable tables and graphical outputs

    """)

with st.expander("📘 Reference Interval Metrics Explained:", expanded=False):
    st.markdown("""
    - **Lower Limit**: The 2.5th percentile (or mean - 1.96 × SD for parametric)
    - **Upper Limit**: The 97.5th percentile (or mean + 1.96 × SD for parametric)
    - **N**: Number of valid observations used in the calculation
    - **Mean**: Average value of the analyte
    - **Median**: Middle value when data is sorted (robust central tendency)
    - **SD**: Standard deviation (measure of spread)
    - **CV%**: Coefficient of variation (SD/Mean × 100) - relative variability
    - **Skewness**: Measure of asymmetry (0 = symmetric, >0 = right-skewed, <0 = left-skewed)
    - **Kurtosis**: Measure of tail heaviness (3 = normal, >3 = heavy tails, <3 = light tails)
    """)

with st.expander("📘 Instructions:", expanded=False):
    st.markdown("""
                
    #### 🛠️ Workflow Summary
    1. Upload your dataset and ensure required fields are present
    2. The app will calculate missing age columns and identify analytes automatically
    3. Select your method of reference interval estimation
    4. Enable optional filters or partitioning by sex and/or age
    5. Review calculated intervals, quality metrics, and visual diagnostics
    6. Download results for documentation or reporting

    ---

    #### ⚠️ Best Practice Notes
    - **Minimum of 120 samples per analyte** is recommended for high-confidence intervals
    - If fewer samples are provided, the app will still compute intervals but will display cautionary quality flags
    - Non-normal distributions are better served with **non-parametric** methods
    - Outlier removal should be used cautiously and reviewed case-by-case

    ---
    
    #### 📂 Input Requirements
    Upload a `.csv` file containing de-identified results from healthy individuals with the following columns:

    - **Required**: `Date of Analysis`, `Date of Birth`, `Sample ID`, `Batch ID`
    - **Optional**: `Sex`, precomputed `Age (y)`, `Age (m)`, or `Age (w)`
    - **Analytes**: All other columns (after demographic/metadata) are assumed to contain numeric analyte values
                
    **To get started:**

    1. Upload your `.csv` file containing results from healthy individuals.
    2. **Required columns:**
        - `Date of Analysis`, `Batch ID`, `Sample ID`, `Date of Birth`
    3. **Optional columns** (will be calculated if missing):
        - `Age (y)`, `Age (m)`, `Age (w)` - Ages in years, months, weeks
        - `Sex` - For sex-specific analysis
    4. **Analyte columns** should appear after the demographic columns.
    5. Apply optional filters for sex, age, or specific batches.
    6. Choose the reference interval method (Non-parametric recommended).
    7. View distributions and download your calculated reference intervals.
    
    **Advanced Features**:
    - **Outlier Detection**: Exclude extreme values (>3SD from mean)
    - **Age Partitioning**: Create age-specific reference intervals
    - **Batch Analysis**: Check for batch effects
    - **Quality Metrics**: Multiple normality tests and distribution assessments
    
    **Note:** Ages will be automatically calculated from Date of Birth and Date of Analysis if not provided.
    All analyte columns will be automatically converted to numeric values.
    """)

def calculate_age_columns(df, dob_col='Date of Birth', analysis_col='Date of Analysis'):
    """
    Calculate ages in years, months, and weeks from date of birth and analysis date.
    Returns the dataframe with added age columns and a summary of the calculation.
    """
    df_with_ages = df.copy()
    summary = {
        'total_records': len(df_with_ages),
        'successful_conversions': 0,
        'date_errors': 0,
        'negative_ages': 0,
        'columns_added': [],
        'status': 'success'
    }
    
    try:
        # Convert date columns to datetime with error handling
        st.info(f"Converting {dob_col} and {analysis_col} to datetime...")
        
        df_with_ages[dob_col] = pd.to_datetime(df_with_ages[dob_col], errors='coerce')
        df_with_ages[analysis_col] = pd.to_datetime(df_with_ages[analysis_col], errors='coerce')
        
        # Check for conversion errors
        dob_nulls = df_with_ages[dob_col].isnull().sum()
        analysis_nulls = df_with_ages[analysis_col].isnull().sum()
        summary['date_errors'] = max(dob_nulls, analysis_nulls)
        
        st.info(f"Date conversion results: {dob_nulls} DOB nulls, {analysis_nulls} analysis date nulls")
        
        # Calculate age in days first
        age_days = (df_with_ages[analysis_col] - df_with_ages[dob_col]).dt.days
        
        # Check for negative ages (analysis date before birth date)
        valid_ages_mask = (age_days >= 0) & (age_days.notna())
        negative_ages = (~valid_ages_mask & age_days.notna()).sum()
        summary['negative_ages'] = negative_ages
        
        if negative_ages > 0:
            st.warning(f"Found {negative_ages} records with analysis date before birth date")
            # Set negative ages to NaN
            age_days = age_days.where(valid_ages_mask)
        
        # Calculate ages in different units
        # Age in years (most accurate)
        df_with_ages['Age (y)'] = (age_days / 365.25).round(2)
        summary['columns_added'].append('Age (y)')
        
        # Age in months (average days per month)
        df_with_ages['Age (m)'] = (age_days / 30.44).round(1)
        summary['columns_added'].append('Age (m)')
        
        # Age in weeks
        df_with_ages['Age (w)'] = (age_days / 7).round(1)
        summary['columns_added'].append('Age (w)')
        
        # Count successful conversions
        summary['successful_conversions'] = df_with_ages['Age (y)'].notna().sum()
        summary['success_rate'] = (summary['successful_conversions'] / summary['total_records']) * 100
        
        st.success(f"Age calculation completed: {summary['successful_conversions']}/{summary['total_records']} successful conversions")
        
    except Exception as e:
        summary['status'] = 'error'
        summary['error_message'] = str(e)
        st.error(f"Error in age calculation: {str(e)}")
        return df_with_ages, summary
    
    return df_with_ages, summary

def convert_to_numeric_enhanced(df, columns):
    conversion_summary = {}
    for col in columns:
        if col in df.columns:
            original_data = df[col].copy()
            original_count = len(original_data.dropna())
            
            original_non_null = original_data.dropna()
            
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
            numeric_data = df[col].dropna()
            numeric_count = len(numeric_data)
            failed_conversions = original_count - numeric_count
            
            problematic_values = []
            if failed_conversions > 0:
                for idx, (orig, new) in enumerate(zip(original_non_null, pd.to_numeric(original_non_null, errors='coerce'))):
                    if pd.isna(new) and not pd.isna(orig):
                        problematic_values.append(str(orig))
                        if len(problematic_values) >= 5:  
                            break
            
            conversion_summary[col] = {
                'original_count': original_count,
                'numeric_count': numeric_count,
                'failed_conversions': failed_conversions,
                'success_rate': (numeric_count / original_count * 100) if original_count > 0 else 0,
                'problematic_values': problematic_values
            }
    
    return df, conversion_summary

def perform_comprehensive_normality_tests(data):
    """Perform multiple normality tests on the data"""
    clean_data = pd.to_numeric(data, errors='coerce').dropna()
    
    if len(clean_data) < 3:
        return {
            'n': len(clean_data),
            'error': 'Insufficient data for normality testing (n < 3)'
        }
    
    results = {'n': len(clean_data)}
    
    try:
        # Shapiro-Wilk test (works best for n <= 5000)
        if 3 <= len(clean_data) <= 5000:
            try:
                shapiro_stat, shapiro_p = shapiro(clean_data)
                results['shapiro_stat'] = shapiro_stat
                results['shapiro_p'] = shapiro_p
                results['shapiro_normal'] = shapiro_p > 0.05
                print(f"Shapiro-Wilk: stat={shapiro_stat:.4f}, p={shapiro_p:.4f}, normal={shapiro_p > 0.05}")
            except Exception as e:
                results['shapiro_error'] = str(e)
                print(f"Shapiro-Wilk error: {e}")
        else:
            results['shapiro_note'] = f"Shapiro-Wilk not applicable (n={len(clean_data)} > 5000)"
        
        # Jarque-Bera test (good for larger samples, needs n >= 8)
        if len(clean_data) >= 8:
            try:
                jb_stat, jb_p = jarque_bera(clean_data)
                results['jarque_bera_stat'] = jb_stat
                results['jarque_bera_p'] = jb_p
                results['jarque_bera_normal'] = jb_p > 0.05
                print(f"Jarque-Bera: stat={jb_stat:.4f}, p={jb_p:.4f}, normal={jb_p > 0.05}")
            except Exception as e:
                results['jarque_bera_error'] = str(e)
                print(f"Jarque-Bera error: {e}")
        else:
            results['jarque_bera_note'] = f"Jarque-Bera not applicable (n={len(clean_data)} < 8)"
        
        # Anderson-Darling test
        if len(clean_data) >= 8:
            try:
                ad_result = anderson(clean_data, dist='norm')
                results['anderson_darling_stat'] = ad_result.statistic
                # Use 5% significance level (index 2)
                results['anderson_darling_critical'] = ad_result.critical_values[2]
                results['anderson_darling_normal'] = ad_result.statistic < ad_result.critical_values[2]
                print(f"Anderson-Darling: stat={ad_result.statistic:.4f}, critical={ad_result.critical_values[2]:.4f}, normal={ad_result.statistic < ad_result.critical_values[2]}")
            except Exception as e:
                results['anderson_darling_error'] = str(e)
                print(f"Anderson-Darling error: {e}")
        
        # Calculate skewness and kurtosis
        try:
            from scipy.stats import skew, kurtosis
            results['skewness'] = skew(clean_data)
            results['kurtosis'] = kurtosis(clean_data, fisher=False)  # Pearson's kurtosis
            print(f"Skewness: {results['skewness']:.4f}, Kurtosis: {results['kurtosis']:.4f}")
        except Exception as e:
            results['skew_kurt_error'] = str(e)
            print(f"Skewness/Kurtosis error: {e}")
        
        # Overall normality assessment (only count valid tests)
        normal_tests = []
        if 'shapiro_normal' in results:
            normal_tests.append(results['shapiro_normal'])
        if 'jarque_bera_normal' in results:
            normal_tests.append(results['jarque_bera_normal'])
        if 'anderson_darling_normal' in results:
            normal_tests.append(results['anderson_darling_normal'])
        
        if normal_tests:
            # Majority vote - if most tests indicate normality
            results['overall_normal'] = sum(normal_tests) >= len(normal_tests) / 2
            results['tests_passed'] = sum(normal_tests)
            results['total_tests'] = len(normal_tests)
        else:
            results['overall_normal'] = None
            results['note'] = 'No valid normality tests could be performed'
        
    except Exception as e:
        st.warning(f"Error in normality testing: {str(e)}")
        results['error'] = str(e)
    
    return results


def display_normality_results_in_quality_section(normality_results):
    """Enhanced display of normality results in the quality assessment section"""
    
    if not normality_results or 'error' in normality_results:
        st.error("❌ Normality testing failed")
        return
    
    st.markdown("**Normality Tests**")
    
    # Shapiro-Wilk results
    if 'shapiro_p' in normality_results:
        if normality_results['shapiro_normal']:
            st.success(f"✅ Shapiro-Wilk: Normal (p={normality_results['shapiro_p']:.4f})")
        else:
            st.warning(f"⚠️ Shapiro-Wilk: Non-normal (p={normality_results['shapiro_p']:.4f})")
    elif 'shapiro_note' in normality_results:
        st.info(f"ℹ️ {normality_results['shapiro_note']}")
    elif 'shapiro_error' in normality_results:
        st.error(f"❌ Shapiro-Wilk failed: {normality_results['shapiro_error']}")
    
    # Jarque-Bera results
    if 'jarque_bera_p' in normality_results:
        if normality_results['jarque_bera_normal']:
            st.success(f"✅ Jarque-Bera: Normal (p={normality_results['jarque_bera_p']:.4f})")
        else:
            st.warning(f"⚠️ Jarque-Bera: Non-normal (p={normality_results['jarque_bera_p']:.4f})")
    elif 'jarque_bera_note' in normality_results:
        st.info(f"ℹ️ {normality_results['jarque_bera_note']}")
    elif 'jarque_bera_error' in normality_results:
        st.error(f"❌ Jarque-Bera failed: {normality_results['jarque_bera_error']}")
    
    # Anderson-Darling results
    if 'anderson_darling_normal' in normality_results:
        if normality_results['anderson_darling_normal']:
            st.success(f"✅ Anderson-Darling: Normal")
        else:
            st.warning(f"⚠️ Anderson-Darling: Non-normal")
        st.caption(f"Statistic: {normality_results.get('anderson_darling_stat', 'N/A'):.4f}, Critical: {normality_results.get('anderson_darling_critical', 'N/A'):.4f}")
    
    # Overall assessment
    if normality_results.get('overall_normal') is not None:
        tests_passed = normality_results.get('tests_passed', 0)
        total_tests = normality_results.get('total_tests', 0)
        
        if normality_results['overall_normal']:
            st.success(f"✅ **Overall Assessment: Normal** ({tests_passed}/{total_tests} tests passed)")
        else:
            st.warning(f"⚠️ **Overall Assessment: Non-normal** ({tests_passed}/{total_tests} tests passed)")
    else:
        st.error("❌ Unable to determine overall normality")


# Fix the quality assessment section in the main code:
def display_quality_assessment_fixed(ref_result, selected_analyte):
    """Fixed quality assessment display"""
    st.subheader("🔍 Quality Assessment")
    
    normality_results = ref_result.get('normality_results', {})
    
    if normality_results and 'error' not in normality_results:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Sample Size Assessment**")
            n = ref_result['n']
            if n >= 120:
                st.success(f"✅ Excellent (n={n})")
                st.write("Meets CLSI guidelines")
            elif n >= 40:
                st.info(f"✅ Good (n={n})")
                st.write("Acceptable sample size")
            elif n >= 20:
                st.warning(f"⚠️ Adequate (n={n})")
                st.write("Minimum acceptable")
            else:
                st.error(f"❌ Insufficient (n={n})")
                st.write("Below recommended minimum")
        
        with col2:
            st.markdown("**Distribution Assessment**")
            skewness = normality_results.get('skewness', 0)
            if abs(skewness) < 0.5:
                st.success("✅ Approximately symmetric")
            elif abs(skewness) < 1:
                st.info("ℹ️ Moderately skewed")
            else:
                st.warning("⚠️ Highly skewed")
            st.write(f"Skewness: {skewness:.3f}")
            
            kurtosis_val = normality_results.get('kurtosis', 3)
            if 2 < kurtosis_val < 4:
                st.success("✅ Normal tail behavior")
            else:
                st.info("ℹ️ Non-normal tails")
            st.write(f"Kurtosis: {kurtosis_val:.3f}")
        
        with col3:
            display_normality_results_in_quality_section(normality_results)
    else:
        st.error("❌ Quality assessment failed - normality testing error")
        if 'error' in normality_results:
            st.write(f"Error: {normality_results['error']}")

def calculate_reference_intervals(data, method="non-parametric", confidence_level=0.95, 
                                        exclude_outliers=False, outlier_method="3SD"):
    """Calculate reference intervals with comprehensive statistics"""
    clean_data = pd.to_numeric(data, errors='coerce').dropna()
    
    if len(clean_data) == 0:
        return None
    
    # Outlier detection and removal
    if exclude_outliers and len(clean_data) > 10:
        if outlier_method == "3SD":
            mean_val = clean_data.mean()
            std_val = clean_data.std()
            outlier_mask = np.abs(clean_data - mean_val) <= (3 * std_val)
            clean_data = clean_data[outlier_mask]
        elif outlier_method == "IQR":
            Q1 = clean_data.quantile(0.25)
            Q3 = clean_data.quantile(0.75)
            IQR = Q3 - Q1
            outlier_mask = (clean_data >= Q1 - 1.5 * IQR) & (clean_data <= Q3 + 1.5 * IQR)
            clean_data = clean_data[outlier_mask]
        elif outlier_method == "Grubbs' Test":
            from scipy.stats import t
            n = len(clean_data)
            if n < 3:
                st.warning("Grubbs' test requires at least 3 data points")
                return None
            
            mean_val = clean_data.mean()
            std_val = clean_data.std(ddof=1)
            G = np.max(np.abs(clean_data - mean_val)) / std_val
            t_critical = t.ppf(1 - 0.05 / (2 * n), n - 2)  # Two-tailed test
            if G > t_critical:
                st.warning("Grubbs' test detected outliers, but this method is not implemented in this version.")
                return None
    else:
        outlier_method = None
    
    # Calculate basic statistics
    n = len(clean_data)
    mean_val = clean_data.mean()
    median_val = clean_data.median()
    std_val = clean_data.std(ddof=1)  # Sample standard deviation
    cv_val = (std_val / mean_val) * 100 if mean_val != 0 else np.nan
    
    # Calculate percentiles for reference interval
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    if method.lower() == "non-parametric":
        # Use percentile method (recommended)
        if n >= 3:
            # Use different interpolation methods based on sample size
            interp_method = 'linear' if n >= 20 else 'nearest'
            lower_limit = np.percentile(clean_data, lower_percentile, method=interp_method)
            upper_limit = np.percentile(clean_data, upper_percentile, method=interp_method)
        else:
            # Too few data points
            lower_limit = clean_data.min()
            upper_limit = clean_data.max()
    
    else:  # parametric method
        # Assume normal distribution
        z_score = norm.ppf(1 - alpha/2)  # 1.96 for 95% CI
        lower_limit = mean_val - z_score * std_val
        upper_limit = mean_val + z_score * std_val
    
    # Comprehensive normality testing
    normality_results = perform_comprehensive_normality_tests(clean_data)
    
    return {
        'n': n,
        'mean': mean_val,
        'median': median_val,
        'std': std_val,
        'cv_percent': cv_val,
        'lower_limit': lower_limit,
        'upper_limit': upper_limit,
        'normality_results': normality_results,
        'method': method,
        'outliers_excluded': exclude_outliers,
        'outlier_method': outlier_method if exclude_outliers else None
    }

def calculate_age_intervals(df, analyte_cols, method="non-parametric", min_age_samples_per_group=20, max_age_partitions=10):
    """
    Calculate age-partitioned reference intervals with comprehensive age handling
    """
    results = []
    
    if 'Age (y)' not in df.columns:
        st.error("❌ Age (y) column not found. Cannot calculate age-partitioned intervals.")
        return pd.DataFrame()
    
    age_data = pd.to_numeric(df['Age (y)'], errors='coerce')
    valid_age_mask = age_data.notna()
    valid_age_count = valid_age_mask.sum()
    
    if valid_age_count == 0:
        st.error("❌ No valid age data found. Cannot calculate age-partitioned intervals.")
        return pd.DataFrame()
    
    st.info(f"📊 Processing age-partitioned intervals for {len(analyte_cols)} analytes with {valid_age_count} valid age records")
    
    age_stats = age_data.describe()
    st.write("**Age Distribution:**")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Min Age", f"{age_stats['min']:.1f} years")
    col2.metric("Max Age", f"{age_stats['max']:.1f} years")
    col3.metric("Mean Age", f"{age_stats['mean']:.1f} years")
    col4.metric("Median Age", f"{age_stats['50%']:.1f} years")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for analyte_idx, analyte in enumerate(analyte_cols):
        status_text.text(f"Processing age partitions for {analyte}... ({analyte_idx + 1}/{len(analyte_cols)})")
        
        analyte_data = pd.to_numeric(df[analyte], errors='coerce')
        combined_data = pd.DataFrame({'analyte': analyte_data, 'age': age_data}).dropna()
        n_total = len(combined_data)
        
        if n_total < min_age_samples_per_group * 2:
            st.warning(f"⚠️ Insufficient data for {analyte} age partitioning (n={n_total}, need at least {min_age_samples_per_group * 2})")
            
            if n_total >= 10:
                ref_result = calculate_reference_intervals(combined_data['analyte'], method=method)
                if ref_result:
                    results.append({
                        'Analyte': analyte,
                        'Age Group': 'All ages (insufficient for partitioning)',
                        'Age Range': f"{combined_data['age'].min():.1f}-{combined_data['age'].max():.1f}",
                        'n': ref_result['n'],
                        'Lower Limit': ref_result['lower_limit'],
                        'Upper Limit': ref_result['upper_limit'],
                        'Mean': ref_result['mean'],
                        'Median': ref_result['median'],
                        'STD': ref_result['std'],
                        'CV (%)': ref_result['cv_percent'],
                        'Quality Score': f'Single interval - only {n_total} samples'
                    })
            continue

        combined_data = combined_data.sort_values('age').reset_index(drop=True)
        max_age_partitions = min(n_total // min_age_samples_per_group, max_age_partitions)
        num_age_partitions = max(2, min(max_age_partitions, 10))

        partition_boundaries = [combined_data['age'].quantile(i / num_age_partitions) for i in range(num_age_partitions + 1)]

        for age_partition_idx in range(num_age_partitions):
            lower_bound = partition_boundaries[age_partition_idx]
            upper_bound = partition_boundaries[age_partition_idx + 1]

            if age_partition_idx == 0:
                mask = combined_data['age'] <= upper_bound
                label = f"≤{upper_bound:.1f}"
            elif age_partition_idx == num_age_partitions - 1:
                mask = combined_data['age'] > lower_bound
                label = f">{lower_bound:.1f}"
            else:
                mask = (combined_data['age'] > lower_bound) & (combined_data['age'] <= upper_bound)
                label = f"{lower_bound:.1f}-{upper_bound:.1f}"

            partition_data = combined_data[mask]
            n_partition = len(partition_data)

            if n_partition < min_age_samples_per_group:
                st.warning(f"  - ⚠️ Age group {age_partition_idx+1} ({label}) has insufficient data ({n_partition} samples)")
                continue

            ref_result = calculate_reference_intervals(partition_data['analyte'], method=method)

            if ref_result:
                quality_score = (
                    "Excellent" if n_partition >= 120 else
                    "Good" if n_partition >= 40 else
                    "Adequate" if n_partition >= 20 else
                    "Insufficient"
                )

                results.append({
                    'Analyte': analyte,
                    'Age Group': f"Group {age_partition_idx+1}: {label} years",
                    'Age Range': f"{partition_data['age'].min():.1f}-{partition_data['age'].max():.1f}",
                    'n': n_partition,
                    'Lower Limit': ref_result['lower_limit'],
                    'Upper Limit': ref_result['upper_limit'],
                    'Mean': ref_result['mean'],
                    'Median': ref_result['median'],
                    'STD': ref_result['std'],
                    'CV (%)': ref_result['cv_percent'],
                    'Quality Score': quality_score
                })
            else:
                st.warning(f"  - ❌ Failed to calculate reference interval for age group {label}")
        
        progress_bar.progress((analyte_idx + 1) / len(analyte_cols))
    
    status_text.text("Age-partitioned analysis complete!")
    progress_bar.empty()
    status_text.empty()
    
    age_results_df = pd.DataFrame(results)
    st.success(f"✅ Generated {len(age_results_df)} age-partitioned intervals")
    return age_results_df

def calculate_sex_partitioned_intervals(df, analyte_cols, method="non-parametric", 
                                                      min_sex_samples_per_group=20):
    """
    Calculate sex-partitioned reference intervals
    """
    results = []

    # Ensure we have sex data
    if 'Sex' not in df.columns:
        st.error("❌ 'Sex' column not found. Cannot calculate sex-partitioned intervals.")
        return pd.DataFrame()

    # Clean and standardize sex data
    df['Sex'] = df['Sex'].astype(str).str.strip().str.capitalize()
    unique_sexes = df['Sex'].dropna().unique()

    if len(unique_sexes) < 1:
        st.error("❌ No valid sex data found. Cannot calculate sex-partitioned intervals.")
        return pd.DataFrame()

    # st.info(f"📊 Processing sex-partitioned intervals for {len(analyte_cols)} analytes and {len(unique_sexes)} sex group(s)")

    # # Show group sizes
    # sex_counts = df['Sex'].value_counts()
    # st.write("**Sex Group Distribution:**")
    # for sex_val in sex_counts.index:
    #     st.metric(f"{sex_val}", f"{sex_counts[sex_val]} samples")

    # Process each analyte
    progress_bar = st.progress(0)
    status_text = st.empty()

    for analyte_idx, analyte in enumerate(analyte_cols):
        status_text.text(f"Processing sex partitions for {analyte}... ({analyte_idx+1}/{len(analyte_cols)})")

        analyte_data = pd.to_numeric(df[analyte], errors='coerce')

        for sex_val in unique_sexes:
            sex_mask = df['Sex'] == sex_val
            partition_data = pd.DataFrame({
                'analyte': analyte_data[sex_mask],
                'sex': df['Sex'][sex_mask]
            }).dropna()

            n_partition = len(partition_data)

            if n_partition < min_sex_samples_per_group:
                st.warning(f"⚠️ {sex_val} group for {analyte} has insufficient data ({n_partition} samples)")
                continue

            ref_result = calculate_reference_intervals(partition_data['analyte'], method=method)

            if ref_result:
                # Quality score based on size
                if n_partition >= 120:
                    quality_score = "Excellent"
                elif n_partition >= 40:
                    quality_score = "Good"
                elif n_partition >= 20:
                    quality_score = "Adequate"
                else:
                    quality_score = "Insufficient"

                results.append({
                    'Analyte': analyte,
                    'Sex Group': sex_val,
                    'n': n_partition,
                    'Lower Limit': ref_result['lower_limit'],
                    'Upper Limit': ref_result['upper_limit'],
                    'Mean': ref_result['mean'],
                    'Median': ref_result['median'],
                    'STD': ref_result['std'],
                    'CV (%)': ref_result['cv_percent'],
                    'Quality Score': quality_score
                })
            else:
                st.warning(f"❌ Failed to calculate reference interval for {analyte} in sex group {sex_val}")

        progress_bar.progress((analyte_idx + 1) / len(analyte_cols))

    status_text.text("Sex-partitioned analysis complete!")
    progress_bar.empty()
    status_text.empty()

    results_df = pd.DataFrame(results)
    st.success(f"✅ Generated {len(results_df)} sex-partitioned intervals")

    return results_df


def create_enhanced_distribution_plot(analyte_data, ref_interval, selected_analyte):
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            f"Distribution - {selected_analyte}",
            "Cumulative Distribution",
            "Q-Q Plot for Normality",
            "Summary Statistics"
        ),
        specs=[
            [{"secondary_y": True}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "table"}]
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    clean_data = pd.to_numeric(analyte_data, errors='coerce').dropna()

    # 1. Main distribution histogram with KDE
    nbins = min(150, max(15, len(clean_data)//8))
    fig.add_trace(
        go.Histogram(
            x=clean_data,
            nbinsx=nbins,
            name="Population Data",
            marker_color="lightblue",
            opacity=0.7,
            histnorm="probability density",
            hovertemplate="Range: %{x}<br>Density: %{y}<extra></extra>",
            showlegend=True
        ),
        row=1, col=1
    )
    
    try:
        kde = gaussian_kde(clean_data)
        x_range = np.linspace(clean_data.min(), clean_data.max(), 200)
        kde_values = kde(x_range)
        
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=kde_values,
                mode="lines",
                name="KDE",
                line=dict(color="blue", width=2),
                hovertemplate="Value: %{x:.3f}<br>Density: %{y:.3f}<extra></extra>",
                showlegend=True
            ),
            row=1, col=1
        )
    except:
        pass
    
    lower_limit = ref_interval['lower_limit']
    upper_limit = ref_interval['upper_limit']
    
    fig.add_vline(x=lower_limit, line=dict(color="red", width=2, dash="dash"), row=1, col=1)
    fig.add_vline(x=upper_limit, line=dict(color="red", width=2, dash="dash"), row=1, col=1)
    
    # 2. Cumulative distribution
    sorted_data = np.sort(clean_data)
    cumulative_prob = np.arange(1, len(sorted_data) + 1) / len(sorted_data) * 100
    
    fig.add_trace(
        go.Scatter(
            x=sorted_data,
            y=cumulative_prob,
            mode="lines",
            name="Empirical CDF",
            line=dict(color="blue", width=2),
            hovertemplate="Value: %{x:.3f}<br>Percentile: %{y:.1f}%<extra></extra>",
            showlegend=False
        ),
        row=1, col=2
    )
    
    percentiles = [2.5, 25, 50, 75, 97.5]
    percentile_values = np.percentile(clean_data, percentiles)
    
    fig.add_trace(
        go.Scatter(
            x=percentile_values,
            y=percentiles,
            mode="markers",
            name="Key Percentiles",
            marker=dict(color="red", size=8, symbol="diamond"),
            hovertemplate="Value: %{x:.3f}<br>Percentile: %{y}%<extra></extra>",
            showlegend=False
        ),
        row=1, col=2
    )
    
    # 3. Q-Q Plot for normality assessment
    try:
        from scipy.stats import probplot
        (theoretical_quantiles, ordered_values), (slope, intercept, r_squared) = probplot(clean_data, dist="norm", plot=None)
        
        fig.add_trace(
            go.Scatter(
                x=theoretical_quantiles,
                y=ordered_values,
                mode="markers",
                name="Q-Q Plot",
                marker=dict(color="blue", size=4),
                hovertemplate="Theoretical: %{x:.3f}<br>Observed: %{y:.3f}<extra></extra>",
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Add the theoretical line (what we'd expect if data were perfectly normal)
        fig.add_trace(
            go.Scatter(
                x=theoretical_quantiles,
                y=slope * theoretical_quantiles + intercept,
                mode="lines",
                line=dict(color="red", dash="dash", width=2),
                name="Theoretical Normal Line",
                showlegend=False
            ),
            row=2, col=1
        )
        
    except Exception as e:
        st.warning(f"Could not generate Q-Q plot: {str(e)}")
    
    # 4. Summary statistics table
    n = ref_interval['n']
    if n >= 120:
        quality = "Excellent"
    elif n >= 40:
        quality = "Good"
    elif n >= 20:
        quality = "Adequate"
    else:
        quality = "Insufficient"
    
    stats_data = [
        ["Statistic", "Value", "Interpretation"],
        ["Sample Size (N)", f"{n}", quality],
        ["Lower Limit (2.5%)", f"{lower_limit:.4f}", "Values below potentially abnormal"],
        ["Upper Limit (97.5%)", f"{upper_limit:.4f}", "Values above potentially abnormal"],
        ["Reference Range", f"{lower_limit:.4f} - {upper_limit:.4f}", "95% of healthy population"],
        ["Mean", f"{ref_interval['mean']:.4f}", "Average value"],
        ["Median", f"{ref_interval['median']:.4f}", "Middle value"],
        ["Standard Deviation", f"{ref_interval['std']:.4f}", "Data spread"],
        ["CV%", f"{ref_interval['cv_percent']:.2f}%" if not np.isnan(ref_interval['cv_percent']) else "N/A", "Relative variability"]
    ]
    
    fig.add_trace(
        go.Table(
            header=dict(
                values=stats_data[0],
                fill_color="blue",
                font=dict(color="white", size=14),
                align="left"
            ),
            cells=dict(
                values=list(zip(*stats_data[1:])),
                fill_color=[["#E1EEFF", "#E1ECFF", "#E1E8FF"]]*len(stats_data[1:]),
                align="left"
            )
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        height=800,
        width=1100,
        title_text=f"Enhanced Distribution and Reference Interval Analysis - {selected_analyte}",
        title_x=0.5,
        showlegend=True,
        font=dict(size=12),
        margin=dict(t=100, b=50)
    )
    
    fig.update_xaxes(title_text=selected_analyte, row=1, col=1)
    fig.update_yaxes(title_text="Density", row=1, col=1)
    fig.update_xaxes(title_text=selected_analyte, row=1, col=2)
    fig.update_yaxes(title_text="Percentile (%)", row=1, col=2)
    fig.update_xaxes(title_text="Theoretical Quantiles", row=2, col=1)
    fig.update_yaxes(title_text="Observed Quantiles", row=2, col=1)
    
    return fig

# --- Main Application ---
if 'data_processed' not in st.session_state:
    st.session_state.data_processed = False
if 'df' not in st.session_state:
    st.session_state.df = None
if 'analyte_cols' not in st.session_state:
    st.session_state.analyte_cols = []
if 'conversion_summary' not in st.session_state:
    st.session_state.conversion_summary = {}

# Sidebar for controls
with st.sidebar:
    st.header("⚙️ Controls")
       
    # Reference interval method
    ri_method = st.selectbox(
        "Reference Interval Method",
        ["Non-parametric (Recommended)", "Parametric (Normal Distribution)"],
        help="Non-parametric uses percentiles and works with any distribution. Parametric assumes normal distribution."
    )
    
    # Confidence level
    confidence_level = st.slider(
        "Confidence Level",
        min_value=0.90,
        max_value=0.99,
        value=0.95,
        step=0.01,
        format="%.2f",
        help="Typically 95% (2.5th to 97.5th percentile)"
    )
    
    # Outlier handling
    st.subheader("🎯 Outlier Detection")
    exclude_outliers = st.checkbox(
        "Exclude outliers",
        value=False,
        help="Remove extreme values before calculating reference intervals"
    )
    
    if exclude_outliers:
        outlier_method = st.selectbox(
            "Outlier Detection Method",
            ["3SD", "IQR", "Grubbs' Test"],
            help="3SD: Remove values >3 standard deviations from mean. \n IQR: Remove values outside 1.5×IQR from Q1/Q3. \n Grubbs: Use Grubbs' test for outlier detection.",
        )
    else:
        outlier_method = "3SD"
    
    # Sex partitioning options
    st.subheader("⚥ Sex Partitioning")
    enable_sex_partitioning = st.checkbox(
        "Calculate sex-partitioned intervals",
        value=False,
        help="Create separate reference intervals for different sex groups"
    )
    
    if enable_sex_partitioning:
        min_sex_samples_per_group = st.number_input(
            "Minimum samples per sex group",
            min_value=10,
            max_value=50,
            value=20,
            help="Minimum number of samples required for each sex group"
        )

    # Age partitioning options
    st.subheader("👥 Age Partitioning")
    enable_age_partitioning = st.checkbox(
        "Calculate age-partitioned intervals",
        value=False,
        help="Create separate reference intervals for different age groups"
    )

    if enable_age_partitioning:
        min_age_samples_per_group = st.number_input(
            "Minimum samples per age group",
            min_value=10,
            max_value=500000000,
            value=20,
            help="Minimum number of samples required for each age group"
        )

    if enable_age_partitioning:
        max_age_partitions = st.number_input(
            "Maximum number of age partitions to calculate ranges",
            min_value=2,
            max_value=500000000,
            value=3,
            help="Maximum number of age partitions to calculate ranges"
        )
        
# ----- Main content area ------
# File upload
with st.expander("📤 Upload Your CSV File", expanded=True):
    st.markdown("Upload a CSV file containing healthy population data with required columns: `Date of Analysis`, `Batch ID`, `Sample ID`, `Date of Birth`.")
    uploaded_file = st.file_uploader("Choose a file to get started", type=["csv"])
    if uploaded_file is not None: 
        st.success(f"✅ File successfully uploaded!")

if uploaded_file is not None:
    with st.spinner("Loading data..."):
        df = pd.read_csv(uploaded_file)
    
    with st.expander("👀 Data Preview", expanded=False):
        st.dataframe(df.head(10), use_container_width=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        st.metric("Total Columns", len(df.columns))
    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())
    with col4:
        completeness = ((df.size - df.isnull().sum().sum()) / df.size) * 100
        st.metric("Data Completeness", f"{completeness:.1f}%")
    
    # Check for required columns
    required_cols = ['Date of Birth', 'Date of Analysis']
    missing_required = [col for col in required_cols if col not in df.columns]
    
    if missing_required:
        st.error(f"❌ Missing required columns: {missing_required}")
        st.info("Required columns: Date of Birth, Date of Analysis, Batch ID, Sample ID")
    else:
        # Check if age columns already exist
        age_cols_exist = any(col in df.columns for col in ['Age (y)', 'Age (m)', 'Age (w)'])
        
        if not age_cols_exist:
            if st.button("🔄 Calculate Ages", type="primary"):
                with st.spinner("Calculating ages..."):
                    df, age_summary = calculate_age_columns(df)
                    st.session_state.df = df
                    
                    # Display age calculation results
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Successful Conversions", f"{age_summary['successful_conversions']}/{age_summary['total_records']}")
                    with col2:
                        st.metric("Success Rate", f"{age_summary.get('success_rate', 0):.1f}%")
                    with col3:
                        st.metric("Date Errors", age_summary['date_errors'])
                    
                    if age_summary['negative_ages'] > 0:
                        st.warning(f"⚠️ Found {age_summary['negative_ages']} records with analysis date before birth date")
        else:
            st.session_state.df = df
    
    # Identify analyte columns
    if st.session_state.df is not None:
        st.subheader("🧪 Analyte Identification")
        
        # Define expected demographic/administrative columns
        expected_admin_cols = [
            'Date of Analysis', 'Batch ID', 'Sample ID', 'Date of Birth',
            'Age (y)', 'Age (m)', 'Age (w)', 'Sex'
        ]
        
        # Identify potential analyte columns (numeric columns not in admin list)
        potential_analytes = []
        for col in st.session_state.df.columns:
            if col not in expected_admin_cols:
                # Check if column contains numeric-like data
                sample_data = st.session_state.df[col].dropna().head(10)
                numeric_count = 0
                for val in sample_data:
                    try:
                        float(val)
                        numeric_count += 1
                    except:
                        pass
                
                if numeric_count >= len(sample_data) * 0.5:  # At least 50% numeric
                    potential_analytes.append(col)
        
        if potential_analytes:
            
            # Allow user to select/deselect analytes
            selected_analytes = st.multiselect(
                "Select analytes for analysis:",
                potential_analytes,
                default=potential_analytes,
                help="Choose which columns contain analyte measurements"
            )
            
            if selected_analytes:
                # Convert analyte columns to numeric
                if st.button("🔢 Process Analyte Data", type="primary"):
                    with st.spinner("Converting analyte data to numeric..."):
                        st.session_state.df, conversion_summary = convert_to_numeric_enhanced(
                            st.session_state.df, selected_analytes
                        )
                        st.session_state.conversion_summary = conversion_summary
                        st.session_state.analyte_cols = selected_analytes
                        st.session_state.data_processed = True
                    
            else:
                st.warning("⚠️ Please select at least one analyte for analysis")
        else:
            st.error("❌ No potential analyte columns found. Please check your data format.")

# Analysis section
if st.session_state.data_processed and st.session_state.analyte_cols:
    df = st.session_state.df
    analyte_cols = st.session_state.analyte_cols
    
    st.header("🔬 Reference Interval Analysis")
    
    # # Filtering options
    # st.subheader("🎛️ Data Filtering")
    
    # # Sex filter
    # if 'Sex' in df.columns or 'Sex' in df.columns:
    #     sex_col = 'Sex' if 'Sex' in df.columns else 'Sex'
    #     available_sexs = df[sex_col].dropna().unique()
    #     selected_sexs = st.multiselect(
    #         f"Filter by {sex_col}:",
    #         available_sexs,
    #         default=list(available_sexs),
    #         help="Select specific sex(s) for analysis"
    #     )
        
    #     if selected_sexs:
    #         df = df[df[sex_col].isin(selected_sexs)]
        
    #     st.info(f"📊 After sex filtering: {len(df)} records")
    
    # Age filter
    if 'Age (y)' in df.columns:
        age_data = pd.to_numeric(df['Age (y)'], errors='coerce').dropna()
        if len(age_data) > 0:
            min_age, max_age = float(age_data.min()), float(age_data.max())
            age_range = st.slider(
                "Age Range (years):",
                min_value=min_age,
                max_value=max_age,
                value=(min_age, max_age),
                step=0.1,
                help="Filter data by age range"
            )
            
            age_mask = (pd.to_numeric(df['Age (y)'], errors='coerce') >= age_range[0]) & \
                      (pd.to_numeric(df['Age (y)'], errors='coerce') <= age_range[1])
            df = df[age_mask.fillna(False)]
            
            st.info(f"📊 After age filtering: {len(df)} records")
    

    if len(df) < 10:
        st.error("❌ Insufficient data after filtering. Need at least 10 records.")
    else:
        # Sample size warning
        if len(df) < 120:
            st.warning(f"⚠️ Sample size ({len(df)}) is below CLSI recommended minimum of 120. Results should be interpreted with caution.")
        

    method = "non-parametric" if "Non-parametric" in ri_method else "parametric"
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, analyte in enumerate(analyte_cols):
        status_text.text(f"Calculating reference interval for {analyte}...")
        
        ref_result = calculate_reference_intervals(
            df[analyte],
            method=method,
            confidence_level=confidence_level,
            exclude_outliers=exclude_outliers,
            outlier_method=outlier_method
        )
        
        if ref_result:
            n = ref_result['n']
            if n >= 120:
                quality = "Excellent"
            elif n >= 40:
                quality = "Good"
            elif n >= 20:
                quality = "Adequate"
            else:
                quality = "Insufficient"
            
            # Normality assessment
            normality = ref_result.get('normality_results', {})
            overall_normal = normality.get('overall_normal', 'Unknown')
            
            results.append({
                'Analyte': analyte,
                'N': n,
                'Quality': quality,
                'Lower Limit': ref_result['lower_limit'],
                'Upper Limit': ref_result['upper_limit'],
                'Reference Interval': f"{ref_result['lower_limit']:.4f} - {ref_result['upper_limit']:.4f}",
                'Mean': ref_result['mean'],
                'Median': ref_result['median'],
                'SD': ref_result['std'],
                'CV%': ref_result['cv_percent'],
                'Method': method.title(),
                'Normal Distribution': 'Yes' if overall_normal else 'No/Unknown',
                'Outliers Excluded': 'Yes' if exclude_outliers else 'No'
            })
        
    # If the age-partitioning checkbox is enabled, calculate age-partitioned intervals
    if enable_age_partitioning:
        age_results_df = calculate_age_intervals(
            df, analyte_cols, 
            method="non-parametric" if "Non-parametric" in ri_method else "parametric",
            min_age_samples_per_group=min_age_samples_per_group,
            max_age_partitions=max_age_partitions
        )
        
        if not age_results_df.empty:
            st.session_state.age_results_df = age_results_df
            
            st.subheader("👥 Age-Partitioned Reference Intervals")
            st.dataframe(age_results_df, use_container_width=True)


    # If the sex-partitioning checkbox is enabled, calculate sex-partitioned intervals         
    if enable_sex_partitioning:
        sex_results_df = calculate_sex_partitioned_intervals(
            df, analyte_cols, 
            method="non-parametric" if "Non-parametric" in ri_method else "parametric",
            min_sex_samples_per_group=min_sex_samples_per_group
        )
        
        if not sex_results_df.empty:
            st.session_state.sex_results_df = sex_results_df
            
            st.subheader("⚥ Sex-Partitioned Reference Intervals")
            st.dataframe(sex_results_df, use_container_width=True)


        progress_bar.progress((idx + 1) / len(analyte_cols))
    
    progress_bar.empty()
    status_text.empty()
        
    # Display results
    if results:
        st.success(f"✅ Calculated reference intervals for {len(results)} analytes")
        
        results_df = pd.DataFrame(results)
        st.session_state.results_df = results_df
        
        # Display results table
        st.subheader("📋 Reference Interval Results")
        st.dataframe(results_df, use_container_width=True)
        
    else:
        st.error("❌ No reference intervals could be calculated")
    
        

# Visualization section
if 'results_df' in st.session_state:
    st.header("📈 Visualization")
    
    # Analyte selection for detailed view
    selected_analyte = st.selectbox(
        "Select analyte for detailed analysis:",
        st.session_state.analyte_cols,
        help="Choose an analyte to view detailed distribution and quality metrics"
    )
    
    if selected_analyte:
        # Get the reference interval for this analyte
        ref_result = calculate_reference_intervals(
            st.session_state.df[selected_analyte],
            method="non-parametric" if "Non-parametric" in ri_method else "parametric",
            confidence_level=confidence_level,
            exclude_outliers=exclude_outliers,
            outlier_method=outlier_method
        )
        
        if ref_result:
            # Create enhanced distribution plot
            fig = create_enhanced_distribution_plot(
                st.session_state.df[selected_analyte], 
                ref_result, 
                selected_analyte
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Quality assessment
            st.subheader("🔍 Quality Assessment")
            
            normality_results = ref_result.get('normality_results', {})
            
            if normality_results:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**Sample Size Assessment**")
                    n = ref_result['n']
                    if n >= 120:
                        st.success(f"✅ Excellent (n={n})")
                        st.write("Meets CLSI guidelines")
                    elif n >= 40:
                        st.info(f"✅ Good (n={n})")
                        st.write("Acceptable sample size")
                    elif n >= 20:
                        st.warning(f"⚠️ Adequate (n={n})")
                        st.write("Minimum acceptable")
                    else:
                        st.error(f"❌ Insufficient (n={n})")
                        st.write("Below recommended minimum")
                
                with col2:
                    st.markdown("**Distribution Assessment**")
                    skewness = normality_results.get('skewness', 0)
                    if abs(skewness) < 0.5:
                        st.success("✅ Approximately symmetric")
                    elif abs(skewness) < 1:
                        st.info("ℹ️ Moderately skewed")
                    else:
                        st.warning("⚠️ Highly skewed")
                    st.write(f"Skewness: {skewness:.3f}")
                    
                    kurtosis_val = normality_results.get('kurtosis', 3)
                    if 2 < kurtosis_val < 4:
                        st.success("✅ Normal tail behavior")
                    else:
                        st.info("ℹ️ Non-normal tails")
                    st.write(f"Kurtosis: {kurtosis_val:.3f}")
                
                with col3:
                    st.markdown("**Normality Tests**")
                    if 'shapiro_p' in normality_results and normality_results['shapiro_p'] is not None:
                        if normality_results['shapiro_normal']:
                            st.success("✅ Shapiro-Wilk: Normal")
                        else:
                            st.warning("⚠️ Shapiro-Wilk: Non-normal")
                        st.write(f"p-value: {normality_results['shapiro_p']:.4f}")
                    
                    if 'jarque_bera_p' in normality_results:
                        if normality_results['jarque_bera_normal']:
                            st.success("✅ Jarque-Bera: Normal")
                        else:
                            st.warning("⚠️ Jarque-Bera: Non-normal")
                        st.write(f"p-value: {normality_results['jarque_bera_p']:.4f}")
                
# Summary section
if 'results_df' in st.session_state:
    # st.header("📊 Analysis Summary")
    
    results_df = st.session_state.results_df
    
    # # Overall statistics
    # col1, col2, col3, col4 = st.columns(4)
    
    # with col1:
    #     st.metric("Total Analytes", len(results_df))
    
    # with col2:
    #     excellent_count = (results_df['Quality'] == 'Excellent').sum()
    #     st.metric("Excellent Quality", excellent_count)
    
    # with col3:
    #     normal_count = (results_df['Normal Distribution'] == 'Yes').sum()
    #     st.metric("Normal Distributions", normal_count)
    
    # with col4:
    #     avg_cv = results_df['CV%'].mean()
    #     st.metric("Average CV%", f"{avg_cv:.1f}%")
    
      
    # Final recommendations
    st.subheader("💡 Recommendations")
    
    insufficient_analytes = results_df[results_df['Quality'] == 'Insufficient']['Analyte'].tolist()
    if insufficient_analytes:
        st.warning(f"⚠️ **Action Required**: The following analytes have insufficient sample sizes and should be re-evaluated with more data: {', '.join(insufficient_analytes)}")
    
    non_normal_analytes = results_df[results_df['Normal Distribution'] == 'No/Unknown']['Analyte'].tolist()
    if non_normal_analytes:
        st.info(f"ℹ️ **Note**: The following analytes show non-normal distributions. \n Non-parametric intervals are recommended: {', '.join(non_normal_analytes)}")
    
    if exclude_outliers:
        st.info("ℹ️ **Note**: Outliers were excluded from the analysis. \n Consider reviewing excluded data points for quality issues.")
    
    st.success("✅ **Analysis Complete**: Reference intervals have been calculated according to clinical laboratory standards. Please review quality metrics before implementation.")

