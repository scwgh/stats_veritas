import pandas as pd
import os
import glob

def select_file():
    """
    For Codespaces: Lists available Excel files in current directory
    and prompts user to enter filename
    """
    # Look for Excel files in current directory
    excel_files = glob.glob("*.xlsx") + glob.glob("*.xls")
    
    if excel_files:
        print("Available Excel files:")
        for i, file in enumerate(excel_files, 1):
            print(f"{i}. {file}")
        
        while True:
            try:
                choice = input("\nEnter the number of the file to use (or full filename): ")
                
                # Try to parse as number first
                if choice.isdigit():
                    file_index = int(choice) - 1
                    if 0 <= file_index < len(excel_files):
                        return excel_files[file_index]
                    else:
                        print("Invalid number. Please try again.")
                        continue
                
                # Try as filename
                if choice in excel_files or os.path.exists(choice):
                    return choice
                
                print("File not found. Please try again.")
                
            except KeyboardInterrupt:
                return None
    else:
        # No Excel files found, ask for manual input
        filename = input("No Excel files found in current directory. Enter full path to Excel file: ")
        if os.path.exists(filename):
            return filename
        else:
            print("File not found.")
            return None

def clean_qc_data(filepath):
    """
    Clean QC data from Excel file, reshape from wide to long format.
    """
    try:
        if filepath.endswith('.xlsx'):
            engine = 'openpyxl'
        elif filepath.endswith('.xls'):
            engine = 'xlrd'
        else:
            engine = None
        
        xls = pd.ExcelFile(filepath, engine=engine) if engine else pd.ExcelFile(filepath)
        print(f"Found {len(xls.sheet_names)} sheets: {xls.sheet_names}")
        
        all_data = []
        for sheet in xls.sheet_names:
            print(f"Processing sheet: {sheet}")
            
            try:
                df = pd.read_excel(filepath, sheet_name=sheet, engine=engine)
            except:
                df = xls.parse(sheet)
            
            if df.empty:
                print(f"  - Skipping empty sheet: {sheet}")
                continue

            df.columns = [str(col).strip() for col in df.columns]
            print(f"  - Columns: {list(df.columns)}")

            # Identify metadata and analyte columns
            metadata_cols = ['Batch', 'Instrument', 'QC Name', 'Date QC']
            analyte_cols = [col for col in df.columns if col not in metadata_cols]
            
            # Check if required metadata columns exist
            missing_cols = [col for col in metadata_cols if col not in df.columns]
            if missing_cols:
                print(f"  - Skipping sheet due to missing columns: {missing_cols}")
                continue

            # Melt into long format
            df_long = df.melt(
                id_vars=metadata_cols,
                value_vars=analyte_cols,
                var_name='Analyte',
                value_name='Result'
            )

            all_data.append(df_long)
            print(f"  - Reshaped to {len(df_long)} rows")
        
        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            return combined
        else:
            print("No valid data found in any sheets.")
            return pd.DataFrame()

    except Exception as e:
        print(f"Error reading Excel file: {e}")
        print("Ensure you have the required libraries: pandas, openpyxl, xlrd")
        return None
1


# ---- Run this to select file and clean it ----
if __name__ == "__main__":
    print("QC Data Cleaner for Codespaces")
    print("=" * 40)
    
    path = select_file()
    if path:
        print(f"\nProcessing file: {path}")
        try:
            cleaned_df = clean_qc_data(path)
            print(f"\nSuccessfully processed {len(cleaned_df)} rows")
            print("\nFirst 10 rows:")
            print(cleaned_df.head(10))
            
            # Optionally save to CSV
            save_option = input("\nSave to CSV? (y/n): ").lower().strip()
            if save_option in ['y', 'yes']:
                output_name = input("Enter output filename (or press Enter for 'cleaned_data.csv'): ").strip()
                if not output_name:
                    output_name = "cleaned_data.csv"
                if not output_name.endswith('.csv'):
                    output_name += '.csv'
                
                cleaned_df.to_csv(output_name, index=False)
                print(f"Data saved to {output_name}")
                
        except Exception as e:
            print(f"Error processing file: {e}")
    else:
        print("No file selected or file not found.") 