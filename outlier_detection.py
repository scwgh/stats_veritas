import numpy as np
import pandas as pd
from scipy import stats
import streamlit as st

class OutlierDetector:
    """
    Unified outlier detection system for both Passing-Bablok and Bland-Altman analyses.
    Provides consistent methods and interpretation across both analysis types.
    """
    
    def __init__(self, alpha=0.05, max_iterations=10):
        self.alpha = alpha
        self.max_iterations = max_iterations
    
    def grubbs_test_single(self, values):
        """
        Single Grubbs test - detects only the most extreme outlier.
        """
        values = pd.Series(values)
        n = len(values)
        if n < 3:
            return np.array([False] * n)

        abs_diff = abs(values - values.mean())
        max_diff_idx = abs_diff.idxmax()
        G = abs_diff[max_diff_idx] / values.std(ddof=1)

        # Critical value from Grubbs test table (two-sided)
        t_crit = stats.t.ppf(1 - self.alpha / (2 * n), df=n - 2)
        G_crit = ((n - 1) / np.sqrt(n)) * np.sqrt(t_crit**2 / (n - 2 + t_crit**2))

        is_outlier = np.array([False] * n)
        if G > G_crit:
            is_outlier[max_diff_idx] = True
        return is_outlier
    
    def grubbs_test_iterative(self, values):
        """
        Iterative Grubbs test - continues removing outliers until no more are found.
        """
        values = pd.Series(values).copy()
        outlier_indices = []
        original_indices = values.index.tolist()
        
        for iteration in range(self.max_iterations):
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
            t_crit = stats.t.ppf(1 - self.alpha / (2 * n), df=n - 2)
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
    
    def statistical_limits_test(self, values, n_std=1.96):
        """
        Identify points outside ±n standard deviations from the mean.
        Default uses 1.96 for 95% limits.
        """
        values = np.array(values)
        mean_val = np.mean(values)
        std_val = np.std(values, ddof=1)
        
        upper_limit = mean_val + n_std * std_val
        lower_limit = mean_val - n_std * std_val
        
        return (values > upper_limit) | (values < lower_limit)
    
    def percentage_difference_test(self, values1, values2, threshold=50.0):
        """
        Identify points with large percentage differences between two methods.
        Default threshold is 50%.
        """
        values1 = np.array(values1)
        values2 = np.array(values2)
        
        # Calculate means for percentage calculation
        means = (values1 + values2) / 2
        differences = np.abs(values1 - values2)
        
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            percent_diffs = np.where(means != 0, (differences / means) * 100, 0)
        
        return percent_diffs > threshold
    
    def iqr_test(self, values, multiplier=1.5):
        """
        Identify outliers using Interquartile Range (IQR) method.
        Points beyond Q1 - multiplier*IQR or Q3 + multiplier*IQR are flagged.
        """
        values = np.array(values)
        Q1 = np.percentile(values, 25)
        Q3 = np.percentile(values, 75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        return (values < lower_bound) | (values > upper_bound)
    
    def combined_method(self, values, values1=None, values2=None):
        """
        Combined approach using multiple detection methods.
        Flags a point as outlier if detected by ANY method.
        """
        values = np.array(values)
        
        # Start with Grubbs iterative
        outliers_grubbs = self.grubbs_test_iterative(values)
        
        # Add statistical limits after removing Grubbs outliers
        if outliers_grubbs.any():
            clean_values = values[~outliers_grubbs]
            if len(clean_values) > 0:
                clean_mean = np.mean(clean_values)
                clean_std = np.std(clean_values, ddof=1)
                clean_upper = clean_mean + 1.96 * clean_std
                clean_lower = clean_mean - 1.96 * clean_std
                
                # Check if any points are outside the new limits
                limit_outliers = (values > clean_upper) | (values < clean_lower)
                outliers_grubbs = outliers_grubbs | limit_outliers
        
        # Add percentage difference test if two sets of values provided
        if values1 is not None and values2 is not None:
            percent_outliers = self.percentage_difference_test(values1, values2, threshold=50.0)
            outliers_grubbs = outliers_grubbs | percent_outliers
        
        return outliers_grubbs
    
    def detect_outliers(self, data, method='statistical_limits', **kwargs):
        """
        Main detection method that routes to appropriate test based on method name.
        
        Parameters:
        -----------
        data : array-like or dict
            If method requires single array: pass as array
            If method requires comparison: pass as dict with keys 'values', 'values1', 'values2'
        method : str
            Detection method to use
        **kwargs : additional parameters for specific methods
        
        Returns:
        --------
        numpy.ndarray : boolean mask indicating outliers
        """
        
        if method == 'grubbs_single':
            return self.grubbs_test_single(data)
        
        elif method == 'grubbs_iterative':
            return self.grubbs_test_iterative(data)
        
        elif method == 'statistical_limits':
            n_std = kwargs.get('n_std', 1.96)
            return self.statistical_limits_test(data, n_std)
        
        elif method == 'percentage_difference':
            if not isinstance(data, dict) or 'values1' not in data or 'values2' not in data:
                raise ValueError("Percentage difference method requires dict with 'values1' and 'values2'")
            threshold = kwargs.get('threshold', 50.0)
            return self.percentage_difference_test(data['values1'], data['values2'], threshold)
        
        elif method == 'iqr':
            multiplier = kwargs.get('multiplier', 1.5)
            return self.iqr_test(data, multiplier)
        
        elif method == 'combined':
            if isinstance(data, dict):
                return self.combined_method(data['values'], data.get('values1'), data.get('values2'))
            else:
                return self.combined_method(data)
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def get_method_descriptions(self):
        """
        Returns descriptions for each detection method.
        """
        return {
            'grubbs_single': {
                'name': 'Grubbs (Single)',
                'description': 'Detects only the most extreme outlier using Grubbs test',
                'best_for': 'Conservative outlier detection, single extreme values'
            },
            'grubbs_iterative': {
                'name': 'Grubbs (Iterative)', 
                'description': 'Repeatedly applies Grubbs test until no more outliers found',
                'best_for': 'Multiple outliers, statistical rigor'
            },
            'statistical_limits': {
                'name': 'Statistical Limits (±1.96σ)',
                'description': 'Flags points outside ±1.96 standard deviations from mean',
                'best_for': 'Simple, interpretable limits based on normal distribution'
            },
            'percentage_difference': {
                'name': 'Large Percentage Difference',
                'description': 'Flags points with >50% relative difference between methods',
                'best_for': 'Clinical significance, method comparison studies'
            },
            'iqr': {
                'name': 'Interquartile Range (IQR)',
                'description': 'Uses Q1 - 1.5×IQR and Q3 + 1.5×IQR as bounds',
                'best_for': 'Non-normal distributions, robust to extreme values'
            },
            'combined': {
                'name': 'Combined Method',
                'description': 'Combines multiple detection approaches for comprehensive screening',
                'best_for': 'Comprehensive analysis, maximum sensitivity'
            }
        }


def create_outlier_detection_interface(merged_data, selected_analyte, analyzer_1, analyzer_2, 
                                     alpha=0.05, analysis_type='bland_altman'):
    """
    Create a standardized outlier detection interface for both analyses.
    
    Parameters:
    -----------
    merged_data : pandas.DataFrame
        Matched data with columns for both analyzers
    selected_analyte : str
        Name of the analyte being analyzed
    analyzer_1, analyzer_2 : str
        Names of the two analyzers being compared
    alpha : float
        Significance level for statistical tests
    analysis_type : str
        'bland_altman' or 'passing_bablok' - determines default method
        
    Returns:
    --------
    tuple : (selected_outliers_mask, method_name, exclude_outliers_flag)
    """
    
    # Initialize detector
    detector = OutlierDetector(alpha=alpha)
    
    # Extract values and calculate differences
    vals1 = merged_data[f'{selected_analyte}_1'].values
    vals2 = merged_data[f'{selected_analyte}_2'].values
    differences = vals1 - vals2
    
    # Get available methods and their descriptions
    methods = detector.get_method_descriptions()
    
    # Set default method based on analysis type
    if analysis_type == 'passing_bablok':
        default_method = 'combined'
        default_index = list(methods.keys()).index(default_method)
    else:  # bland_altman
        default_method = 'statistical_limits' 
        default_index = list(methods.keys()).index(default_method)
    
    # Run all methods for comparison
    st.markdown("**Outlier Detection Method Comparison:**")
    
    method_results = {}
    comparison_data = []
    
    for method_key, method_info in methods.items():
        try:
            if method_key == 'percentage_difference':
                # This method needs both sets of values
                outliers = detector.detect_outliers(
                    {'values1': vals1, 'values2': vals2}, 
                    method=method_key
                )
            elif method_key == 'combined':
                # Combined method can use differences and both value sets
                outliers = detector.detect_outliers(
                    {'values': differences, 'values1': vals1, 'values2': vals2},
                    method=method_key
                )
            else:
                # Other methods use differences
                outliers = detector.detect_outliers(differences, method=method_key)
            
            method_results[method_key] = outliers
            
            n_outliers = sum(outliers)
            outlier_sample_ids = merged_data['Sample ID'].iloc[outliers].tolist() if n_outliers > 0 else []
            
            comparison_data.append({
                'Method': method_info['name'],
                'Description': method_info['description'],
                'Outliers Found': n_outliers,
                'Sample IDs': ', '.join(map(str, outlier_sample_ids)) if outlier_sample_ids else 'None',
                'Best For': method_info['best_for']
            })
            
        except Exception as e:
            st.warning(f"Could not run {method_info['name']}: {str(e)}")
            method_results[method_key] = np.array([False] * len(differences))
            comparison_data.append({
                'Method': method_info['name'],
                'Description': f"Error: {str(e)}",
                'Outliers Found': 0,
                'Sample IDs': 'Error',
                'Best For': method_info['best_for']
            })
    
    # Display comparison table
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    # Method selection
    method_names = [methods[key]['name'] for key in methods.keys()]
    selected_method_name = st.selectbox(
        "Select outlier detection method:",
        options=method_names,
        index=default_index,
        help=f"""
        **Recommended for {analysis_type.replace('_', '-').title()}:** {methods[default_method]['name']}
        
        Choose based on your analysis needs:
        • **Conservative**: Grubbs (Single) - finds only extreme outliers
        • **Balanced**: Statistical Limits - standard ±1.96σ approach  
        • **Comprehensive**: Combined Method - maximum sensitivity
        • **Clinical Focus**: Percentage Difference - based on relative differences
        """
    )
    
    # Get the selected method key
    selected_method_key = list(methods.keys())[method_names.index(selected_method_name)]
    selected_outliers = method_results[selected_method_key]
    
    # Display outlier details if any found
    outlier_indices = np.where(selected_outliers)[0]
    
    if len(outlier_indices) == 0:
        st.success(f"No outliers detected using {selected_method_name}")
        return selected_outliers, selected_method_name, False
    
    else:
        st.error(f"{len(outlier_indices)} outlier(s) detected using {selected_method_name}")
        
        # Create detailed outlier table
        outlier_details = []
        for idx in outlier_indices:
            sample_id = merged_data['Sample ID'].iloc[idx]
            val1 = vals1[idx]
            val2 = vals2[idx]
            diff = differences[idx]
            mean_val = (val1 + val2) / 2
            
            # Calculate z-score relative to all differences
            mean_diff = np.mean(differences)
            std_diff = np.std(differences, ddof=1)
            z_score = abs(diff - mean_diff) / std_diff if std_diff != 0 else 0
            
            # Calculate percentage difference
            percent_diff = abs(diff / mean_val) * 100 if mean_val != 0 else 0
            
            outlier_details.append({
                'Sample ID': sample_id,
                f'{analyzer_1}': round(val1, 3),
                f'{analyzer_2}': round(val2, 3),
                'Difference': round(diff, 3),
                'Mean': round(mean_val, 3),
                '% Difference': round(percent_diff, 1),
                'Z-Score': round(z_score, 2),
                'Beyond ±1.96σ': 'Yes' if abs(z_score) > 1.96 else 'No'
            })
        
        outlier_df = pd.DataFrame(outlier_details)
        st.dataframe(outlier_df, use_container_width=True, hide_index=True)
        
        # Exclusion option
        exclude_outliers = st.checkbox(
            f"Exclude these {len(outlier_indices)} outliers from analysis",
            value=False,
            help=f"""
            When enabled:
            • Outliers will be completely removed from plots and calculations
            • Statistical measures will reflect only the remaining data
            • Sample size (n) will be reduced by {len(outlier_indices)}
            
            When disabled:
            • Outliers will be highlighted but included in analysis
            • All statistical measures include outlier data
            """
        )
        
        if exclude_outliers:
            outlier_sample_ids = [outlier_details[i]['Sample ID'] for i in range(len(outlier_indices))]
            st.warning(f"Analysis will exclude samples: {', '.join(map(str, outlier_sample_ids))}")
        else:
            st.info("Outliers will be highlighted in red but included in calculations")
        
        return selected_outliers, selected_method_name, exclude_outliers


def create_outlier_explanation_section(method_name, n_excluded, excluded_sample_ids, 
                                     original_stats, final_stats, alpha):
    """
    Create a comprehensive explanation section for outlier handling.
    """
    if n_excluded == 0:
        st.success("No outliers were excluded from this analysis")
        return
    
    with st.expander(f"Why Were These {n_excluded} Outliers Excluded?", expanded=False):
        st.markdown(f"""
        **{n_excluded} sample(s) were excluded** using the **{method_name}** method at significance level α = {alpha}.
        
        ### Method Explanation: {method_name}
        """)
        
        # Method-specific explanations
        if 'Grubbs' in method_name:
            st.markdown("""
            The Grubbs test (extreme studentized deviate test) identifies outliers by:
            1. Calculating how many standard deviations each point is from the mean
            2. Comparing this to a critical value based on sample size and significance level
            3. Flagging points that exceed the critical threshold
            
            **Why use Grubbs?**
            • Statistically rigorous approach based on t-distribution
            • Accounts for sample size in determining critical values
            • Widely accepted in analytical chemistry and method validation
            """)
        elif 'Statistical Limits' in method_name:
            st.markdown("""
            Statistical Limits method identifies outliers as points beyond ±1.96 standard deviations:
            • Based on assumption that differences follow normal distribution
            • ±1.96σ contains ~95% of normally distributed data
            • Simple, interpretable, and widely used in clinical laboratories
            
            **When to use:**
            • Data approximately normally distributed
            • Need simple, explainable criteria
            • Consistent with Bland-Altman interpretation
            """)
        elif 'Combined' in method_name:
            st.markdown("""
            Combined Method uses multiple approaches:
            1. **Grubbs iterative test** for statistical outliers
            2. **±1.96σ limits** after removing Grubbs outliers
            3. **Large percentage differences** (>50%) for clinical significance
            
            **Advantages:**
            • Comprehensive detection across multiple criteria
            • Captures both statistical and clinically significant outliers
            • Robust across different data distributions
            """)
        elif 'Percentage' in method_name:
            st.markdown("""
            Percentage Difference method flags samples with >50% relative difference:
            • Focuses on clinical/practical significance rather than statistical
            • Useful when absolute differences vary with concentration
            • Identifies samples where methods disagree substantially in relative terms
            """)
        
        st.markdown("""
        ### Why Exclude Outliers in Method Comparison Studies?
        
        **Scientific Justification:**
        • **Reduce bias:** Extreme values can skew limits of agreement and regression parameters
        • **Improve precision:** Remove sources of excessive variability not representative of routine performance
        • **Clinical relevance:** Focus on typical method agreement under normal conditions
        • **Quality control:** May indicate measurement errors, sample issues, or instrument problems
        
        **Statistical Benefits:**
        • More reliable confidence intervals
        • Better estimates of method precision
        • Improved normality of residuals
        • Reduced influence of extreme leverage points
        """)
        
        # Show impact on statistics
        if original_stats and final_stats:
            st.markdown("### Statistical Impact of Exclusion")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Before Exclusion:**")
                st.markdown(f"• Sample size: {original_stats['n']}")
                st.markdown(f"• Mean difference: {original_stats['mean_diff']:.3f}")
                st.markdown(f"• Standard deviation: {original_stats['std_diff']:.3f}")
                if 'correlation' in original_stats:
                    st.markdown(f"• Correlation (R²): {original_stats['correlation']:.3f}")
            
            with col2:
                st.markdown("**After Exclusion:**")
                st.markdown(f"• Sample size: {final_stats['n']}")
                st.markdown(f"• Mean difference: {final_stats['mean_diff']:.3f}")
                st.markdown(f"• Standard deviation: {final_stats['std_diff']:.3f}")
                if 'correlation' in final_stats:
                    st.markdown(f"• Correlation (R²): {final_stats['correlation']:.3f}")
        
        # Show excluded samples
        st.markdown(f"""
        ### Excluded Sample Details
        
        **Sample IDs excluded:** {', '.join(map(str, excluded_sample_ids))}
        
        ### Recommendations
        
        1. **Investigate excluded samples:**
           • Review measurement procedures for these samples
           • Check for transcription errors or instrument issues
           • Consider sample-specific factors (hemolysis, interference, etc.)
        
        2. **Clinical assessment:**
           • Determine if differences are clinically significant
           • Consider method-specific characteristics
           • Evaluate impact on patient care decisions
        
        3. **Quality improvement:**
           • Use findings to improve analytical procedures
           • Enhance training on critical measurement steps
           • Consider additional quality control measures
        
        4. **Documentation:**
           • Record rationale for outlier exclusion
           • Maintain audit trail for regulatory compliance
           • Include in method validation documentation
        
        **Remember:** Outlier exclusion should be based on scientific rationale, not statistical convenience. Always investigate the root cause of extreme differences.
        """)


# Example usage function that both analyses can call
def standardized_outlier_detection(merged_data, selected_analyte, analyzer_1, analyzer_2, 
                                 alpha=0.05, analysis_type='bland_altman'):
    """
    Standardized outlier detection function for both Passing-Bablok and Bland-Altman analyses.
    
    Returns:
    --------
    dict with keys:
        - 'outliers_mask': boolean array indicating outliers
        - 'method_name': name of detection method used
        - 'exclude_outliers': whether user chose to exclude outliers
        - 'n_outliers': number of outliers detected
        - 'outlier_sample_ids': list of outlier sample IDs
    """
    
    # Create the interface
    outliers_mask, method_name, exclude_outliers = create_outlier_detection_interface(
        merged_data, selected_analyte, analyzer_1, analyzer_2, alpha, analysis_type
    )
    
    # Compile results
    n_outliers = sum(outliers_mask)
    outlier_sample_ids = merged_data['Sample ID'].iloc[outliers_mask].tolist() if n_outliers > 0 else []
    
    return {
        'outliers_mask': outliers_mask,
        'method_name': method_name,
        'exclude_outliers': exclude_outliers,
        'n_outliers': n_outliers,
        'outlier_sample_ids': outlier_sample_ids
    }