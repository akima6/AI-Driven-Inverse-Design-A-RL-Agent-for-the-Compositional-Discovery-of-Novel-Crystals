# evaluate_models.py (Corrected for Modularity - FINAL VERSION)
# This script now ONLY imports the custom model class from the central models.hybrid_class file
# and does NOT contain a local definition.

import os
import joblib
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# --- KEY CHANGE: Import the custom model class from the central models file ---
from models.hybrid_class import HybridBandGapModel

# Suppress common warnings for a cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --- The local class definition has been REMOVED from this script ---

# ==============================================================================
# --- Main Evaluation Script ---
# ==============================================================================
if __name__ == '__main__':
    print("--- Starting Model Evaluation ---")

    # --- 1. Define File Paths ---
    DATA_PATH = './data/inorganic_materials_featurized(1).csv'
    STABILITY_MODEL_PATH = './models/Real_stability_model.joblib'
    BANDGAP_MODEL_PATH = './models/Real_hybrid_band_gap_model.joblib'
    OUTPUT_DIR = './evaluation_results'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"✅ Plots and results will be saved to: {OUTPUT_DIR}")

    # --- 2. Load Data ---
    print("\n--- Loading and Preparing Data ---")
    try:
        df = pd.read_csv(DATA_PATH)
        df.dropna(subset=['formation_energy_per_atom', 'band_gap'], inplace=True)
        print(f"✅ Data loaded successfully. Shape: {df.shape}")
    except FileNotFoundError:
        print(f"❌ ERROR: Data file not found at '{DATA_PATH}'.")
        exit()

    # Define features (X) and targets (y)
    cols_to_drop = ['material_id', 'pretty_formula', 'composition', 'formation_energy_per_atom', 'band_gap', 'is_metal']
    existing_cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    X = df.drop(columns=existing_cols_to_drop)
    y_stability = df['formation_energy_per_atom']
    y_bandgap = df['band_gap']

    # --- 3. Split Data ---
    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X, y_stability, test_size=0.2, random_state=42)
    X_train_bg, X_test_bg, y_train_bg, y_test_bg = train_test_split(X, y_bandgap, test_size=0.2, random_state=42, stratify=(y_bandgap == 0))
    print("✅ Data split into training and test sets (80/20).")

    # --- 4. Load Trained Models ---
    print("\n--- Loading Trained Models ---")
    try:
        stability_model = joblib.load(STABILITY_MODEL_PATH)
        bandgap_model = joblib.load(BANDGAP_MODEL_PATH)
        print("✅ Models loaded successfully.")
    except Exception as e:
        print(f"❌ ERROR: Could not load the band gap model. {e}")
        print("This can happen if the model was saved with a different import path.")
        print("Please run the 'hybrid_band_gap_predictor.py' script to re-save the model correctly.")
        exit()

    # --- 5. Evaluate Stability Model ---
    print("\n--- Evaluating Stability Model (Formation Energy) ---")
    y_pred_s = stability_model.predict(X_test_s)
    print(f"  - R² Score: {r2_score(y_test_s, y_pred_s):.4f}")
    print(f"  - Mean Absolute Error (MAE): {mean_absolute_error(y_test_s, y_pred_s):.4f} eV/atom")
    print(f"  - Root Mean Squared Error (RMSE): {np.sqrt(mean_squared_error(y_test_s, y_pred_s)):.4f} eV/atom")

    # --- 6. Evaluate Band Gap Model ---
    print("\n--- Evaluating Band Gap Model ---")
    y_pred_bg = bandgap_model.predict(X_test_bg)
    print(f"  - R² Score: {r2_score(y_test_bg, y_pred_bg):.4f}")
    print(f"  - Mean Absolute Error (MAE): {mean_absolute_error(y_test_bg, y_pred_bg):.4f} eV")
    print(f"  - Root Mean Squared Error (RMSE): {np.sqrt(mean_squared_error(y_test_bg, y_pred_bg)):.4f} eV")

    # --- 7. Generate and Save Evaluation Plots ---
    # This part is optional but good for verification
    print("\n--- Generating and Saving Plots ---")
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=y_test_s, y=y_pred_s, alpha=0.5)
    plt.plot([y_test_s.min(), y_test_s.max()], [y_test_s.min(), y_test_s.max()], '--r', linewidth=2, label="Ideal Fit")
    plt.title("Stability Model: Predicted vs. Actual")
    plt.xlabel("Actual Formation Energy (eV/atom)")
    plt.ylabel("Predicted Formation Energy (eV/atom)")
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'stability_evaluation_plot.png'))
    plt.close()
    print(f"✅ Stability plot saved.")

    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=y_test_bg, y=y_pred_bg, alpha=0.5)
    plt.plot([y_test_bg.min(), y_test_bg.max()], [y_test_bg.min(), y_test_bg.max()], '--r', linewidth=2, label="Ideal Fit")
    plt.title("Band Gap Model: Predicted vs. Actual")
    plt.xlabel("Actual Band Gap (eV)")
    plt.ylabel("Predicted Band Gap (eV)")
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'bandgap_evaluation_plot.png'))
    plt.close()
    print(f"✅ Band gap plot saved.")
    
    print("\n--- Evaluation Script Finished ---")



