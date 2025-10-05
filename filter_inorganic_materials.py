"""
Loads a CSV file containing raw materials data (formula, formation energy, band gap),
filters out organic compounds (those containing both carbon and hydrogen),
and saves the remaining inorganic materials to a new CSV file.

How it works:
1. Loads the dataset from a specified local path.
2. Removes rows with missing chemical formulas.
3. Converts formulas to Composition objects using matminer.
4. Identifies and removes organic compounds (containing both C and H).
5. Saves the filtered inorganic dataset to a new local CSV file.

Files used/generated:
- Input CSV: materials_project_raw_data.csv (local)
- Output CSV: inorganic_materials_raw_data.csv (saved locally)
"""

import pandas as pd
import os
from matminer.featurizers.conversions import StrToComposition

# === Step 1: Define file paths ===
project_folder_path = os.path.join(os.getcwd(), "composition")
raw_data_filename = os.path.join(project_folder_path, 'materials_project_raw_data.csv')
inorganic_data_filename = os.path.join(project_folder_path, 'inorganic_materials_raw_data.csv')

if not os.path.exists(raw_data_filename):
    raise FileNotFoundError(f"Input file not found: {raw_data_filename}")

# === Step 2: Load the dataset ===
df = pd.read_csv(raw_data_filename)
print("Successfully loaded clean raw data from local path.")
print(f"Original shape of the dataset: {df.shape}")

# === Step 3: Remove rows with missing formulas ===
df.dropna(subset=['pretty_formula'], inplace=True)
print(f"Shape after dropping rows with missing formulas: {df.shape}")

# === Step 4: Convert formula strings to Composition objects ===
df = StrToComposition(target_col_id='composition').featurize_dataframe(
    df, "pretty_formula", ignore_errors=True
)
df.dropna(subset=['composition'], inplace=True)
print("Generated 'composition' objects for filtering.")

# === Step 5: Apply inorganic filter ===
is_organic_mask = df['composition'].apply(lambda comp: 'C' in comp and 'H' in comp)
df_inorganic = df[~is_organic_mask].copy()
df_inorganic.drop(columns=['composition'], inplace=True)

# === Step 6: Save the filtered inorganic dataset ===
os.makedirs(project_folder_path, exist_ok=True)
df_inorganic.to_csv(inorganic_data_filename, index=False)

# === Step 7: Verification ===
print(f"\nNumber of materials in starting dataset: {len(df)}")
print(f"Number of materials identified as organic and removed: {is_organic_mask.sum()}")
print(f"Number of materials in the new INORGANIC dataset: {len(df_inorganic)}")
print(f"Successfully saved inorganic dataset to: {inorganic_data_filename}")
print("\nFirst 5 rows of the inorganic dataset:")
print(df_inorganic.head())
