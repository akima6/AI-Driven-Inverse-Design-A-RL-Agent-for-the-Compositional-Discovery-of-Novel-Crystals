"""
Downloads raw materials data (formula, formation energy, and band gap) from the Materials Project API
using the MPRester client and saves it locally as a CSV file.

How it works:
1. Connects to the Materials Project API using an API key.
2. Queries for non-deprecated materials with selected properties:
   - formula_pretty
   - formation_energy_per_atom
   - band_gap
3. Stores the results in a clean pandas DataFrame.
4. Saves the DataFrame to a specified local file path.

Files used/generated:
- Output CSV: materials_project_raw_data.csv (saved in the specified local folder)

"""

import pandas as pd
import os
from mp_api.client import MPRester

# === Step 1: Define save location and API Key ===
project_folder_path = os.path.join(os.getcwd(), "composition")
local_data_filename = os.path.join(project_folder_path, 'materials_project_raw_data.csv')
os.makedirs(project_folder_path, exist_ok=True)
print(f"Project folder is ready at: {project_folder_path}")

# === Step 2: Initialize MPRester and download the data ===
api_key = input("Enter your Materials Project API key: ").strip()

print("Querying the Materials Project database...")
with MPRester(api_key=api_key) as mpr:
    docs = mpr.materials.summary.search(
        deprecated=False,
        fields=["formula_pretty", "formation_energy_per_atom", "band_gap"]
    )
print(f"Data download complete. Retrieved {len(docs)} materials.")

# === Step 3: Create a clean DataFrame ===
data_for_df = []
for doc in docs:
    data_for_df.append({
        "pretty_formula": doc.formula_pretty,
        "formation_energy_per_atom": doc.formation_energy_per_atom,
        "band_gap": doc.band_gap
    })

df_clean = pd.DataFrame(data_for_df)

# === Step 4: Save to local file ===
df_clean.to_csv(local_data_filename, index=False)
print(f"Successfully saved clean data to: {local_data_filename}")

# === Step 5: Verification ===
print("\n--- SETUP COMPLETE ---")
print("First 5 rows of the clean data:")
print(df_clean.head())
print(f"\nFile saved at: {local_data_filename}")
