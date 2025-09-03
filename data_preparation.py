# ===== STEP 1: Create a new shared utilities file =====
# Create a new file called "data_preparation.py" in your project directory

import pandas as pd
import numpy as np
import streamlit as st

def prepare_matched_data_standardized(df, material_type, selected_analyte, analyzer_1, analyzer_2, 
                                    handle_duplicates='mean', verbose=True):
    """
    Standardized data preparation function for both Bland-Altman and Deming regression.
    This ensures consistent sample counts across both analyses.
    
    Parameters:
    -----------
    df : DataFrame
        Input dataframe with columns: Material, Analyser, Sample ID, and analyte columns
    material_type : str
        Selected material type to filter data
    selected_analyte : str
        Column name of the analyte to analyze
    analyzer_1 : str
        Name of first analyzer
    analyzer_2 : str
        Name of second analyzer
    handle_duplicates : str, default 'mean'
        How to handle duplicate Sample IDs: 'mean', 'first', 'last', 'error'
    verbose : bool, default True
        Whether to print detailed information about data processing
        
    Returns:
    --------
    merged_data : DataFrame
        DataFrame with columns: Sample ID, {analyte}_1, {analyte}_2
    """
    
    if verbose:
        st.info(f"Starting data preparation for {selected_analyte}: {analyzer_1} vs {analyzer_2}")
    
    # Step 1: Filter for selected material
    initial_count = len(df)
    data = df[df['Material'] == material_type].copy()
    material_count = len(data)
    
    # if verbose:
    #     st.write(f"• Filtered for material '{material_type}': {material_count}/{initial_count} rows")
    
    # Step 2: Convert analyte to numeric and handle missing values
    data[selected_analyte] = pd.to_numeric(data[selected_analyte], errors='coerce')
    before_na_removal = len(data)
    data = data.dropna(subset=[selected_analyte])
    after_na_removal = len(data)
    
    # if verbose and before_na_removal != after_na_removal:
    #     st.write(f"• Removed {before_na_removal - after_na_removal} rows with missing {selected_analyte} values")
    
    # Step 3: Separate data by analyzer
    data_analyzer1 = data[data['Analyser'] == analyzer_1][['Sample ID', selected_analyte]].copy()
    data_analyzer2 = data[data['Analyser'] == analyzer_2][['Sample ID', selected_analyte]].copy()
    
    # if verbose:
    #     st.write(f"• {analyzer_1}: {len(data_analyzer1)} measurements")
    #     st.write(f"• {analyzer_2}: {len(data_analyzer2)} measurements")
    
    # Check if we have data for both analyzers
    if len(data_analyzer1) == 0:
        raise ValueError(f"No data found for analyzer '{analyzer_1}' with material '{material_type}' and analyte '{selected_analyte}'")
    if len(data_analyzer2) == 0:
        raise ValueError(f"No data found for analyzer '{analyzer_2}' with material '{material_type}' and analyte '{selected_analyte}'")
    
    # Step 4: Handle duplicate Sample IDs
    original_count_1 = len(data_analyzer1)
    original_count_2 = len(data_analyzer2)
    
    if handle_duplicates == 'mean':
        data_analyzer1 = data_analyzer1.groupby('Sample ID')[selected_analyte].mean().reset_index()
        data_analyzer2 = data_analyzer2.groupby('Sample ID')[selected_analyte].mean().reset_index()
    elif handle_duplicates == 'first':
        data_analyzer1 = data_analyzer1.drop_duplicates('Sample ID', keep='first')
        data_analyzer2 = data_analyzer2.drop_duplicates('Sample ID', keep='first')
    elif handle_duplicates == 'last':
        data_analyzer1 = data_analyzer1.drop_duplicates('Sample ID', keep='last')
        data_analyzer2 = data_analyzer2.drop_duplicates('Sample ID', keep='last')
    elif handle_duplicates == 'error':
        dup1 = data_analyzer1['Sample ID'].duplicated().any()
        dup2 = data_analyzer2['Sample ID'].duplicated().any()
        if dup1 or dup2:
            duplicates_info = []
            if dup1:
                dups1 = data_analyzer1[data_analyzer1['Sample ID'].duplicated(keep=False)]['Sample ID'].unique()
                duplicates_info.append(f"{analyzer_1}: {list(dups1)}")
            if dup2:
                dups2 = data_analyzer2[data_analyzer2['Sample ID'].duplicated(keep=False)]['Sample ID'].unique()
                duplicates_info.append(f"{analyzer_2}: {list(dups2)}")
            raise ValueError(f"Duplicate Sample IDs found: {'; '.join(duplicates_info)}")
    
    # if verbose and (len(data_analyzer1) != original_count_1 or len(data_analyzer2) != original_count_2):
    #     st.write(f"• After handling duplicates ({handle_duplicates}):")
    #     st.write(f"  - {analyzer_1}: {len(data_analyzer1)} samples (was {original_count_1})")
    #     st.write(f"  - {analyzer_2}: {len(data_analyzer2)} samples (was {original_count_2})")
    
    # Step 5: Final cleanup - remove any remaining NaN values
    data_analyzer1 = data_analyzer1.dropna()
    data_analyzer2 = data_analyzer2.dropna()
    
    # Step 6: Merge on Sample ID to get matched pairs
    merged_data = pd.merge(
        data_analyzer1, 
        data_analyzer2, 
        on='Sample ID', 
        suffixes=('_1', '_2'),
        how='inner'
    )
    
    # Step 7: Final NaN cleanup (shouldn't be needed but ensures consistency)
    final_before = len(merged_data)
    merged_data = merged_data.dropna()
    final_after = len(merged_data)
    
    # if verbose:
    #     st.success(f"• Final result: {len(merged_data)} matched sample pairs")
    #     if final_before != final_after:
    #         st.warning(f"• Removed {final_before - final_after} pairs with NaN values during final cleanup")
        
    #     # Show sample IDs that couldn't be matched
    #     unmatched_1 = set(data_analyzer1['Sample ID']) - set(merged_data['Sample ID'])
    #     unmatched_2 = set(data_analyzer2['Sample ID']) - set(merged_data['Sample ID'])
        
    #     if unmatched_1 or unmatched_2:
    #         st.write("• Unmatched samples:")
    #         if unmatched_1:
    #             st.write(f"  - {analyzer_1} only: {sorted(list(unmatched_1))}")
    #         if unmatched_2:
    #             st.write(f"  - {analyzer_2} only: {sorted(list(unmatched_2))}")
    
    return merged_data


def get_analysis_ready_data(df, material_type, selected_analyte, analyzer_1, analyzer_2, 
                          handle_duplicates='mean', verbose=True):
    """
    Get standardized data arrays ready for analysis.
    
    Returns:
    --------
    tuple: (x_vals, y_vals, sample_ids, n_samples, merged_data)
        x_vals : numpy array of analyzer_1 values
        y_vals : numpy array of analyzer_2 values  
        sample_ids : numpy array of Sample IDs
        n_samples : int, number of matched samples
        merged_data : DataFrame, the full merged dataset
    """
    
    merged_data = prepare_matched_data_standardized(
        df, material_type, selected_analyte, analyzer_1, analyzer_2, 
        handle_duplicates, verbose
    )
    
    x_vals = merged_data[f'{selected_analyte}_1'].values
    y_vals = merged_data[f'{selected_analyte}_2'].values
    sample_ids = merged_data['Sample ID'].values
    n_samples = len(merged_data)
    
    return x_vals, y_vals, sample_ids, n_samples, merged_data

