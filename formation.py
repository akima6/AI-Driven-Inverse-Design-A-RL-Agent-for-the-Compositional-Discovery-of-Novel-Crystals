"""
Formation Energy Prediction Script

This script trains a LightGBM model to predict the formation energy per atom
for inorganic materials based on precomputed material features.

Main workflow:
1. Loads featurized inorganic materials data from a local CSV file.
2. Drops non-feature columns and NaN target values.
3. Uses Optuna with K-fold cross-validation to find the best hyperparameters.
4. Trains a final LightGBM model with the best parameters.
5. Saves the trained model to the ./models directory.
6. Generates:
   - Predicted vs. Actual scatter plot.
   - SHAP feature importance summary plot.

Files used/generated:
- Input (expected): inorganic_materials_featurized(1).csv (local, ML-ready features)
- Outputs:
  - ./models/stability_model_final.joblib (trained model)
  - formation_energy_predicted_vs_actual.png (scatter plot)
  - formation_energy_shap_summary.png (SHAP feature importance plot)
"""

import pandas as pd
import numpy as np
import os
import joblib
import lightgbm as lgb
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from tqdm import tqdm

# ==============================================================================
# --- Configuration ---
# ==============================================================================
DATA_FILE = os.path.join(os.getcwd(), 'inorganic_materials_featurized(1).csv')
TARGET_VARIABLE = 'formation_energy_per_atom'
N_TRIALS = 100  # Number of Optuna trials
N_FOLDS = 5     # Cross-validation folds per trial

# ==============================================================================
# --- Step 1: Load and Prepare Data ---
# ==============================================================================
print("\n" + "="*80)
print("--- Step 1: Loading Data for Formation Energy Model ---")
print("="*80)
try:
    df = pd.read_csv(DATA_FILE)
    print(f"✅ Data loaded successfully. Initial shape: {df.shape}")
except FileNotFoundError:
    print(f"❌ ERROR: Data file '{DATA_FILE}' not found.")
    exit()

# Drop rows where either target is NaN
df = df.dropna(subset=['formation_energy_per_atom', 'band_gap'])

cols_to_drop = [
    'material_id', 'pretty_formula', 'composition',
    'formation_energy_per_atom', 'band_gap', 'is_metal'
]
cols_that_exist = [col for col in cols_to_drop if col in df.columns]
print(f"\nColumns to be dropped from features: {cols_that_exist}")

X = df.drop(columns=cols_that_exist)
y = df[TARGET_VARIABLE]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("✅ Data loaded and prepared.")
print(f"Final feature shape for training: {X_train.shape}")

# ==============================================================================
# --- Step 2: Find Best Hyperparameters ---
# ==============================================================================
print("\n" + "="*80)
print(f"--- Step 2: Finding Best Hyperparameters ({N_TRIALS} trials, {N_FOLDS} folds each) ---")
print("="*80)
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective_fe(trial):
    params = {
        'objective': 'regression_l1', 'metric': 'mae',
        'n_estimators': trial.suggest_int('n_estimators', 200, 2000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'num_leaves': trial.suggest_int('num_leaves', 20, 200),
        'max_depth': trial.suggest_int('max_depth', 5, 30),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'verbose': -1, 'n_jobs': -1, 'seed': 42,
    }

    model = lgb.LGBMRegressor(**params)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    mae_scores = []

    for train_idx, val_idx in kf.split(X_train):
        X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
        model.fit(X_train_fold, y_train_fold)
        preds = model.predict(X_val_fold)
        mae_scores.append(mean_absolute_error(y_val_fold, preds))

    return np.mean(mae_scores)

with tqdm(total=N_TRIALS, desc="Optimizing Hyperparameters") as pbar:
    def callback(study, trial): pbar.update(1)
    study_fe = optuna.create_study(direction='minimize', study_name='LGBM_FE_Tuning')
    study_fe.optimize(objective_fe, n_trials=N_TRIALS, callbacks=[callback])

print("\n✅ Optimization Complete.")
print(f"Best cross-validated MAE: {study_fe.best_value:.4f}")
print("--- Best Hyperparameters ---")
print(study_fe.best_params)

# ==============================================================================
# --- Step 3: Train Final Model & Save ---
# ==============================================================================
print("\n" + "="*80)
print("--- Step 3: Training and Saving the Final Stability Model ---")
print("="*80)
best_params_fe = study_fe.best_params
final_model_fe = lgb.LGBMRegressor(objective='regression_l1', metric='mae',
                                   **best_params_fe, random_state=42, verbose=-1)

final_model_fe.fit(X_train, y_train)

model_dir = './models'
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, 'stability_model_final.joblib')
joblib.dump(final_model_fe, model_path)
print(f"✅ Model saved to: {model_path}")

# ==============================================================================
# --- Step 4: Final Evaluation & Plots ---
# ==============================================================================
print("\n" + "="*80)
print("--- Step 4: Final Model Evaluation on Test Set ---")
print("="*80)
y_pred = final_model_fe.predict(X_test)

print(f"  - R²: {r2_score(y_test, y_pred):.4f}")
print(f"  - MAE: {mean_absolute_error(y_test, y_pred):.4f} eV/atom")
print(f"  - RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f} eV/atom")

# Predicted vs Actual Plot
fig, ax = plt.subplots(figsize=(8, 8))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.5, ax=ax, label="Predictions")
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2, label="Ideal Fit")
ax.set_xlabel("Actual Formation Energy (eV/atom)")
ax.set_ylabel("Predicted Formation Energy (eV/atom)")
ax.set_title("Final Stability Model: Predicted vs. Actual")
ax.legend()
plt.tight_layout()
plt.savefig("formation_energy_predicted_vs_actual.png")
print("✅ Saved Predicted vs. Actual plot.")

# SHAP Feature Importance
print("\nCalculating SHAP values... (this may take a moment)")
explainer = shap.TreeExplainer(final_model_fe)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, show=False)
plt.title("SHAP Feature Importance - Formation Energy")
plt.tight_layout()
plt.savefig("formation_energy_shap_summary.png")
print("✅ Saved SHAP summary plot.")

print("\n" + "="*80)
print("--- Script Finished ---")
print("="*80)
