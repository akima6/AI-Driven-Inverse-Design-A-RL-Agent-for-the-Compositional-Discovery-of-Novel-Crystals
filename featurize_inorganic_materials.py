# --- Step 0: Install necessary libraries (if not already installed) ---
# Uncomment the next line if matminer is not installed
# !pip install matminer --quiet

# --- Step 1: Imports ---
import os
import pandas as pd
import joblib

# Matminer imports with error handling
try:
    from matminer.featurizers.conversions import StrToComposition
    from matminer.featurizers.composition import ElementProperty, ElementFraction, Stoichiometry
    from matminer.featurizers.base import MultipleFeaturizer
except ImportError as e:
    print("Error importing matminer. Install it via: pip install matminer")
    raise e

# --- Step 2: Define file paths ---
project_folder_path = os.path.join(os.getcwd(), "composition")
raw_input_filename = os.path.join(project_folder_path, 'inorganic_materials_raw_data.csv')
featurizer_path = os.path.join(project_folder_path, 'featurizer.joblib')
final_ml_data_filename = os.path.join(project_folder_path, 'inorganic_materials_featurized.csv')

# Ensure project folder exists
os.makedirs(project_folder_path, exist_ok=True)

# Check input file
if not os.path.exists(raw_input_filename):
    raise FileNotFoundError(
        f"Input file not found: {raw_input_filename}\n"
        "Place 'inorganic_materials_raw_data.csv' in the 'composition' folder."
    )

# --- Step 3: Load raw dataset ---
df = pd.read_csv(raw_input_filename)
print("✅ Loaded INORGANIC raw data.")
print(f"📊 Shape before featurization: {df.shape}")

# --- Step 4: Convert formulas to Composition objects ---
stc = StrToComposition(target_col_id='composition')
df = stc.featurize_dataframe(df, "pretty_formula", ignore_errors=True)
df.dropna(subset=['composition'], inplace=True)
print("✅ Converted formulas to composition objects.")
print(f"Shape after composition conversion: {df.shape}")

# --- Step 5: Apply multiple featurizers (Magpie + ElementFraction + Stoichiometry) ---
print("⚙️ Starting featurization (Magpie + ElementFraction + Stoichiometry)...")

featurizer = MultipleFeaturizer([
    ElementProperty.from_preset("magpie"),
    ElementFraction(),
    Stoichiometry()
])

df = featurizer.featurize_dataframe(df, col_id="composition", ignore_errors=True)
df.dropna(inplace=True)
print("✅ Featurization complete. Removed rows with missing values.")
print(f"Shape after featurization: {df.shape}")

# --- Step 6: Save featurizer and final dataset ---
print("\n💾 Saving featurizer and final dataset...")
try:
    joblib.dump(featurizer, featurizer_path)
    print(f"✅ Featurizer saved to: {featurizer_path}")
except Exception as e:
    print("⚠️ Failed to save featurizer. Check permissions or disk space.")
    raise e

df.to_csv(final_ml_data_filename, index=False)
print(f"✅ Featurized dataset saved to: {final_ml_data_filename}")

# --- Step 7: Final summary ---
print(f"\n✅ Final dataset shape: {df.shape}")
print("First 5 rows:")
print(df.head())
