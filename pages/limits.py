import streamlit as st
import pandas as pd
import numpy as np
import re
from utils import apply_app_styling

# Page setup
st.set_page_config(
    page_title="Limits",
    page_icon="♾️",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_app_styling()
st.title("♾️ Limits")

# --- Method Explanation ---
with st.expander("📘 What are limits?", expanded=True):
    st.markdown("""
    It is important to characterise the analytical performance of a laboratory assay to determine if it is "_fit for purpose_". This includes understanding the limits of detection (LOD), quantification (LOQ), and blank (LOB) for the analytes of interest. 
    \n LOB and LOD are important to discriminate between the presence or absence of an analyte; whereas LOQ is important for clinical diagnosis and management. 
    \n **LOD**, or **Limit of Detection**, refers to the **lowest concentration of an analyte** that can be reliably distinguished from background noise or a blank signal—but not necessarily quantified with accuracy or precision.
    """)
    st.latex(r"\text{LOD} = \mu_{\text{blank}} + {\text{3SD}}")
    st.markdown("""**LOB**, or **Limit of Blank**, is the highest concentration of analyte that is likely to be found in a blank sample.""")
    st.latex(r"\text{LOB} = \mu_{\text{blank}} + 1.645 \cdot \sigma_{\text{blank}}")
    st.markdown("""For the purpose of these analyses, LOB is not included. """)
    st.markdown("""
    **LOQ**, or **Limit of Quantification**, is the lowest concentration at which an analyte can be quantitatively detected with acceptable **precision** and **accuracy**.
    \n If the concentration of the lowest non-zero calibrator meets clinical requirements and has an IQC material available, the LLOQ may be determined through inter-assay imprecision and accuracy studies.
    \n LOQ may be calculated using **serial dilutions**, as described below; or using **low concentration samples**. For example:
    """)
    st.latex(r"\text{LOQ} = 10 \cdot \sigma_{\text{low concentration sample}}")


# --- Instructions ---
with st.expander("📘 Instructions"):
    # st.markdown("""**For LOB analysis:** Upload a CSV file with repeated blank samples labeled in the `Material` column.""")
    st.markdown("""**For LOD/LOQ analysis:** Upload a CSV file with repeated low concentration samples labeled (e.g., LowConc1), and ensure analyte columns contain numeric values. If you are using a serial dilution to calculate LOQ., please label the dilution in the 'Sample Name' column using the following format: *** Dilution (1 in x ) *** """)
    st.markdown("""
    ### 🔬 To calculate LOQ using serial dilutions:

    - Perform **serial dilution** of the low-level calibrator to produce multiple intermediate concentrations at the low end of the standard curve.  
    - Conduct **5 replicate extractions** of:
    - The low-level calibrator, and  
    - Each calibrator dilution.  
    - **Inject the extracts** across at least **3 separate runs** and record:
        - Analyte response (e.g., peak area), and  
        - Measured concentration.  
        - Calculate the **mean response** at each concentration (this may be used to plot a **precision profile**).  
        - Calculate the **mean percentage accuracy** and **coefficient of variation (CV, %)** of the replicates compared to the nominal concentration.
    """)

def get_analyte_names(df):
    pattern = r'Calculated (.+)'  # Look for 'Calculated <Analyte>'
    analyte_names = [re.match(pattern, col).group(1) for col in df.columns if re.match(pattern, col)]
    return analyte_names


# --- Upload ---
def upload_data():
    with st.expander("📤 Upload Your CSV File", expanded=True):
        st.markdown("Upload a CSV containing your analyte data. Ensure it includes: `Material`, `Analyser`, and `Sample ID`.")
        uploaded_file = st.file_uploader("Choose a file to get started", type=["csv"], label_visibility="collapsed")
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.success("✅ File uploaded successfully!")
            st.markdown("### 📖 Data Preview")
            st.dataframe(df.head())
            return df
        else:
            st.info("Awaiting file upload...")
            return None


# # --- LOB ---
# def calculate_lob(df):
#     analyte_names = get_analyte_names(df)
#     blank_data = df[df['Material'].str.lower() == 'blank']
#     results_lob = {
#         'Analyte': [],
#         'Blank Mean': [
#         'Blank SD': [],
#         'LOB': []
#     }

#     for analyte in analyte_names:
#         col_name = f'Calculated {analyte}'
#         if col_name not in df.columns:
#             continue

#         blank_vals = pd.to_numeric(blank_data[col_name], errors='coerce').dropna()
#         if blank_vals.empty:
#             continue

#         blank_mean = round(blank_vals.mean(), 5)
#         blank_sd = round(blank_vals.std(), 5)
#         lob = round(blank_mean + 1.645 * blank_sd, 5)

#         results_lob['Analyte'].append(analyte)
#         results_lob['Blank Mean'].append(blank_mean)
#         results_lob['Blank SD'].append(blank_sd)
#         results_lob['LOB'].append(lob)

#     result_lob_df = pd.DataFrame(results_lob)
#     st.subheader("📊 Limit of Blank (LOB) Summary")
#     st.dataframe(result_lob_df)


# --- LOD (response-based) ---
def calculate_lod_response(df):
    analyte_names = get_analyte_names(df)
    blank_data = df[df['Material'].str.lower().str.contains('blank', na=False)]
    low_data = df[df['Material'].str.lower().str.contains('low', na=False)]

    results = {
        'Analyte': [],
        'Blank SD (Response)': [],
        'LOD (Response)': []
    }

    for analyte in analyte_names:
        response_col = f'{analyte} Response'
        if response_col not in df.columns:
            st.warning(f"⚠️ Missing response column for {analyte}: '{response_col}'")
            continue

        vals = pd.to_numeric(low_data[response_col], errors='coerce').dropna()
        if len(vals) < 2:
            st.warning(f"⚠️ Not enough valid data for {analyte}")
            continue

        sd = round(vals.std(), 5)
        mean = round(vals.mean(), 5)
        lod = round(mean + 3 * sd, 5)

        results['Analyte'].append(analyte)
        results['Blank SD (Response)'].append(sd)
        results['LOD (Response)'].append(lod)

    result_df = pd.DataFrame(results)
    st.subheader("📊 LOD Summary (Response-based)")
    st.dataframe(result_df)


# --- LOD (concentration-based) ---
def calculate_lod_concentration(df):
    analyte_names = get_analyte_names(df)
    blank_data = df[df['Material'].str.lower().str.contains('blank', na=False)]    
    results = {
        'Analyte': [],
        'Blank SD (Concentration)': [],
        'LOD (Concentration)': []
    }

    for analyte in analyte_names:
        col_name = f'Calculated {analyte}'
        if col_name not in df.columns:
            continue

        vals = pd.to_numeric(blank_data[col_name], errors='coerce').dropna()
        if len(vals) < 2:
            continue

        sd = round(vals.std(), 5)
        mean = round(vals.mean(), 5)
        lod = round(3 * sd, 5)

        results['Analyte'].append(analyte)
        results['Blank SD (Concentration)'].append(sd)
        results['LOD (Concentration)'].append(lod)

    result_df = pd.DataFrame(results)
    st.subheader("📊 LOD Summary (Concentration-based)")
    st.dataframe(result_df)

# --- LOQ (low concentration-based) ---
def calculate_loq_low_concentration(df):
    analyte_columns = get_analyte_names(df)
    low_data = df[df['Material'].str.lower().str.contains('low', na=False)]

    results = {
        'Analyte': [],
        'Mean Concentration': [],
        'Mean Response': [],
        'SD': [],
        'CV%': [],
        'LOQ (10 × SD)': []
    }

    for analyte in analyte_columns:
        conc_col = f'Calculated {analyte}'
        if conc_col not in df.columns:
            continue

        response_col = f'{analyte} Response'
        if response_col not in df.columns:
            continue

        values = pd.to_numeric(low_data[conc_col], errors='coerce').dropna()
        if len(values) < 2:
            continue

        response = pd.to_numeric(low_data[response_col], errors='coerce').dropna()

        mean_val = values.mean()
        mean_response = response.mean()
        std_val = values.std(ddof=1)
        cv = (std_val / mean_val) * 100 if mean_val != 0 else np.nan
        loq = 10 * std_val

        results['Analyte'].append(analyte)
        results['Mean Concentration'].append(round(mean_val, 4))
        results['Mean Response'].append(round(mean_response, 2))
        results['SD'].append(round(std_val, 4))
        results['CV%'].append(round(cv, 2))
        results['LOQ (10 × SD)'].append(round(loq, 4))

    st.subheader("📊 LOQ Based on Low Concentration Samples")
    st.dataframe(pd.DataFrame(results))


# --- LOQ (serial dilution-based) ---
def extract_dilution_factor(name):
    """Extract dilution factor from Sample Name, e.g., 'Diluted (1/66)' -> 66"""
    match = re.search(r"\(1/(\d+)\)", name)
    return int(match.group(1)) if match else np.nan

def calculate_loq_dilution(df):
    analyte_columns = get_analyte_names(df)
    loq_data = df[df['Test'].str.lower() == 'loq']
    blank_data = df[df['Material'].str.lower() == 'blank']
    dilution_data = df[df['Sample Name'].str.lower().str.contains('dil', na=False)]

    results = {
        'Analyte': [],
        'Dilution Level': [],
        'Mean Concentration': [],
        'Mean Response': [],
        'SD': [],
        'CV%': [],
        'Pass LOQ Criteria': []
    }

    for analyte in analyte_columns:
        response_col = f'{analyte} Response'
        conc_col = f'Calculated {analyte}'
        std_col = f'Expected {analyte}'
        if conc_col not in df.columns or std_col not in df.columns:
            continue
        if response_col not in df.columns:
            continue
        blank_mean = pd.to_numeric(blank_data[conc_col], errors='coerce').mean()

        for level in dilution_data['Sample Name'].unique():
            level_data = dilution_data[dilution_data['Sample Name'] == level]

            calc_conc = pd.to_numeric(level_data[conc_col], errors='coerce')

            response = pd.to_numeric(level_data[response_col], errors='coerce').dropna()

            nominal = pd.to_numeric(level_data[std_col], errors='coerce')

            if len(calc_conc.dropna()) < 2 or len(nominal.dropna()) == 0:
                continue
            mean_conc = calc_conc.mean()
            mean_response = response.mean()
            std_dev = calc_conc.std(ddof=1)  
            cv = (std_dev / mean_conc) * 100 if mean_conc != 0 else np.nan
            acc = (mean_conc - nominal.mean()) / nominal.mean() * 100

            passed = (
                mean_conc > 5 * blank_mean
                and abs(acc) <= 20
                and cv <= 20
            )

            results['Analyte'].append(analyte)
            results['Dilution Level'].append(level)
            results['Mean Concentration'].append(round(mean_conc, 4))
            results['Mean Response'].append(round(mean_response, 2))
            results['SD'].append(round(std_dev, 4))
            results['CV%'].append(round(cv, 2))
            results['Pass LOQ Criteria'].append("✅" if passed else "❌")

    st.subheader("📊 Limit of Quantification (LOQ) Assessment")
    st.dataframe(pd.DataFrame(results))
    

# --- Run sections ---
df = upload_data()

# with st.expander("📊 Calculate Limit of Blank (LOB)", expanded=True):
#     if df is not None:
#         calculate_lob(df)

with st.expander("📊 Calculate Limit of Detection (LOD) (Response-based)", expanded=False):
    if df is not None:
        calculate_lod_response(df)

with st.expander("📊 Calculate Limit of Detection (LOD) (Concentration-based)", expanded=False):
    if df is not None:
        calculate_lod_concentration(df)

with st.expander("📊 Calculate Limit of Quantification (LOQ)", expanded=True):
    if df is not None:
        loq_method = st.radio(
            "Choose LOQ calculation method:",
            options=["Use Serial Dilution", "Use Low Concentration Samples"],
            horizontal=True
        )

        if loq_method == "Use Serial Dilution":
            calculate_loq_dilution(df)
        elif loq_method == "Use Low Concentration Samples":
            calculate_loq_low_concentration(df)


with st.expander("📚 References"):
    st.markdown("""
    **Armbruster, D.A. and Pry, T. (2008)**, *Limit of Blank, Limit of Detection and Limit of Quantitation*, [Clinical Biochemist Reviews](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2556583/)
    """)
    st.markdown("""
    **Dolan, J. W. (2013)**, *What's the problem with the LLOQ?*, [LCGC Europe](https://www.chromatographyonline.com/view/whats-problem-lloq-case-study-1)
    """)
    st.markdown("""
    **Pum, J. (2019)**, *A practical guide to validation and verification of analytical methods*, [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S006524231930006X)
    """)
