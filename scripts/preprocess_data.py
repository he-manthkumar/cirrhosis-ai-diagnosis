"""
Data Preprocessing Script for Cirrhosis Dataset.

This script:
1. Loads raw data
2. Cleans and validates data
3. Handles missing values
4. Creates derived features
5. Saves processed data

Run: python -m scripts.preprocess_data
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_raw_data(filepath: str) -> pd.DataFrame:
    """Load raw CSV data."""
    print(f"Loading data from: {filepath}")
    df = pd.read_csv(filepath, na_values=['NA', 'N/A', '', ' '])
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


def explore_data(df: pd.DataFrame) -> None:
    """Print data exploration summary."""
    print("\n" + "=" * 60)
    print("DATA EXPLORATION")
    print("=" * 60)
    
    print("\n📊 Dataset Shape:", df.shape)
    
    print("\n📋 Column Types:")
    print(df.dtypes)
    
    print("\n❌ Missing Values:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({
        'Missing': missing,
        'Percentage': missing_pct
    })
    print(missing_df[missing_df['Missing'] > 0].sort_values('Missing', ascending=False))
    
    print("\n🎯 Target Variable (Status) Distribution:")
    print(df['Status'].value_counts())
    print(df['Status'].value_counts(normalize=True).mul(100).round(2).astype(str) + '%')
    
    print("\n📈 Numerical Features Summary:")
    print(df.describe().T)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate data."""
    print("\n" + "=" * 60)
    print("DATA CLEANING")
    print("=" * 60)
    
    df_clean = df.copy()
    
    # 1. Remove ID column (not a feature)
    if 'ID' in df_clean.columns:
        df_clean = df_clean.drop(columns=['ID'])
        print("✓ Removed 'ID' column")
    
    # 2. Handle N_Days (keep for potential survival analysis, but not for classification)
    # We'll keep it in processed data but exclude from features during training
    print("✓ Kept 'N_Days' for reference (excluded during training)")
    
    # 3. Validate target variable
    valid_status = ['C', 'CL', 'D']
    invalid_status = df_clean[~df_clean['Status'].isin(valid_status)]
    if len(invalid_status) > 0:
        print(f"⚠ Found {len(invalid_status)} rows with invalid Status, removing...")
        df_clean = df_clean[df_clean['Status'].isin(valid_status)]
    print(f"✓ Target variable validated: {df_clean['Status'].nunique()} classes")
    
    # 4. Validate categorical columns
    categorical_mappings = {
        'Sex': ['M', 'F'],
        'Ascites': ['Y', 'N'],
        'Hepatomegaly': ['Y', 'N'],
        'Spiders': ['Y', 'N'],
        'Edema': ['N', 'S', 'Y'],
        'Drug': ['D-penicillamine', 'Placebo']
    }
    
    for col, valid_values in categorical_mappings.items():
        if col in df_clean.columns:
            invalid = df_clean[~df_clean[col].isin(valid_values + [np.nan])]
            if len(invalid) > 0:
                print(f"⚠ Found {len(invalid)} invalid values in '{col}'")
    print("✓ Categorical columns validated")
    
    # 5. Validate numerical ranges (clinical plausibility)
    numerical_checks = {
        'Age': (0, 40000),  # ~0-110 years in days
        'Bilirubin': (0, 100),
        'Cholesterol': (0, 1000),
        'Albumin': (0, 10),
        'Copper': (0, 1000),
        'Alk_Phos': (0, 20000),
        'SGOT': (0, 500),
        'Tryglicerides': (0, 1000),
        'Platelets': (0, 1000),
        'Prothrombin': (0, 30),
        'Stage': (1, 4)
    }
    
    for col, (min_val, max_val) in numerical_checks.items():
        if col in df_clean.columns:
            outliers = df_clean[(df_clean[col] < min_val) | (df_clean[col] > max_val)]
            outliers = outliers[outliers[col].notna()]
            if len(outliers) > 0:
                print(f"⚠ Found {len(outliers)} outliers in '{col}' outside ({min_val}, {max_val})")
    print("✓ Numerical ranges checked")
    
    return df_clean


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values with appropriate strategies."""
    print("\n" + "=" * 60)
    print("HANDLING MISSING VALUES")
    print("=" * 60)
    
    df_imputed = df.copy()
    
    # Strategy for each column type
    # Numerical: Median imputation (robust to outliers)
    # Categorical: Mode imputation
    
    numerical_cols = ['Bilirubin', 'Cholesterol', 'Albumin', 'Copper', 
                      'Alk_Phos', 'SGOT', 'Tryglicerides', 'Platelets', 'Prothrombin']
    
    categorical_cols = ['Drug', 'Ascites', 'Hepatomegaly', 'Spiders', 'Edema']
    
    # Impute numerical columns with median
    for col in numerical_cols:
        if col in df_imputed.columns:
            missing_count = df_imputed[col].isnull().sum()
            if missing_count > 0:
                median_val = df_imputed[col].median()
                df_imputed[col] = df_imputed[col].fillna(median_val)
                print(f"  {col}: Imputed {missing_count} values with median ({median_val:.2f})")
    
    # Impute categorical columns with mode
    for col in categorical_cols:
        if col in df_imputed.columns:
            missing_count = df_imputed[col].isnull().sum()
            if missing_count > 0:
                mode_val = df_imputed[col].mode()[0]
                df_imputed[col] = df_imputed[col].fillna(mode_val)
                print(f"  {col}: Imputed {missing_count} values with mode ('{mode_val}')")
    
    # Handle Stage - special case (ordinal, missing may be meaningful)
    if 'Stage' in df_imputed.columns:
        missing_count = df_imputed['Stage'].isnull().sum()
        if missing_count > 0:
            # Use median for ordinal data
            median_stage = df_imputed['Stage'].median()
            df_imputed['Stage'] = df_imputed['Stage'].fillna(median_stage)
            print(f"  Stage: Imputed {missing_count} values with median ({median_stage:.0f})")
    
    # Verify no missing values remain
    remaining_missing = df_imputed.isnull().sum().sum()
    print(f"\n✓ Remaining missing values: {remaining_missing}")
    
    return df_imputed


def create_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create useful derived features."""
    print("\n" + "=" * 60)
    print("CREATING DERIVED FEATURES")
    print("=" * 60)
    
    df_features = df.copy()
    
    # 1. Age in years (more interpretable)
    if 'Age' in df_features.columns:
        df_features['Age_Years'] = (df_features['Age'] / 365.25).round(1)
        print("✓ Created 'Age_Years' from Age (days)")
    
    # 2. Bilirubin categories (clinical relevance)
    if 'Bilirubin' in df_features.columns:
        df_features['Bilirubin_Category'] = pd.cut(
            df_features['Bilirubin'],
            bins=[0, 1.2, 3.0, 10, float('inf')],
            labels=['Normal', 'Mild', 'Moderate', 'Severe']
        )
        print("✓ Created 'Bilirubin_Category' (Normal/Mild/Moderate/Severe)")
    
    # 3. Albumin categories
    if 'Albumin' in df_features.columns:
        df_features['Albumin_Category'] = pd.cut(
            df_features['Albumin'],
            bins=[0, 2.8, 3.5, float('inf')],
            labels=['Low', 'Borderline', 'Normal']
        )
        print("✓ Created 'Albumin_Category' (Low/Borderline/Normal)")
    
    # 4. Stage as ordinal integer
    if 'Stage' in df_features.columns:
        df_features['Stage'] = df_features['Stage'].astype(int)
        print("✓ Converted 'Stage' to integer")
    
    # 5. Binary complication indicator (any of: Ascites, Hepatomegaly, Spiders, Edema=Y)
    complication_cols = ['Ascites', 'Hepatomegaly', 'Spiders']
    if all(col in df_features.columns for col in complication_cols):
        df_features['Has_Complications'] = (
            (df_features['Ascites'] == 'Y') | 
            (df_features['Hepatomegaly'] == 'Y') | 
            (df_features['Spiders'] == 'Y') |
            (df_features['Edema'] == 'Y')
        ).astype(int)
        print("✓ Created 'Has_Complications' binary indicator")
    
    return df_features


def save_processed_data(df: pd.DataFrame, filepath: str) -> None:
    """Save processed data to CSV."""
    print("\n" + "=" * 60)
    print("SAVING PROCESSED DATA")
    print("=" * 60)
    
    df.to_csv(filepath, index=False)
    print(f"✓ Saved to: {filepath}")
    print(f"  Shape: {df.shape}")


def main():
    print("=" * 60)
    print("CIRRHOSIS DATA PREPROCESSING PIPELINE")
    print("=" * 60)
    
    # Paths
    raw_path = "data/raw/cirrhosis.csv"
    processed_path = "data/processed/cirrhosis_cleaned.csv"
    
    # 1. Load raw data
    df = load_raw_data(raw_path)
    
    # 2. Explore data
    explore_data(df)
    
    # 3. Clean data
    df_clean = clean_data(df)
    
    # 4. Handle missing values
    df_imputed = handle_missing_values(df_clean)
    
    # 5. Create derived features
    df_final = create_derived_features(df_imputed)
    
    # 6. Final exploration
    print("\n" + "=" * 60)
    print("FINAL DATASET SUMMARY")
    print("=" * 60)
    print(f"Shape: {df_final.shape}")
    print(f"Columns: {list(df_final.columns)}")
    print(f"\nTarget Distribution:")
    print(df_final['Status'].value_counts())
    
    # 7. Save processed data
    save_processed_data(df_final, processed_path)
    
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Run: python -m scripts.train_model")
    print("  2. Run: python -m scripts.evaluate_model")


if __name__ == "__main__":
    main()
