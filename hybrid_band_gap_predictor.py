"""
Hybrid Band Gap Prediction Script

This script trains or uses a hybrid LightGBM-based model to predict the band gap of inorganic materials.
The hybrid approach:
1. A classifier first predicts whether a material is metallic (band gap = 0).
2. A regressor predicts the band gap for non-metallic materials.

Modes:
- train: Reads featurized dataset, runs optional Optuna hyperparameter optimization, trains the hybrid model, evaluates it, and saves it.
- predict: Loads a trained model and makes predictions on new data.

Files used/generated:
- Input (expected): inorganic_materials_featurized(1).csv (local, ML-ready features)
- Output (train mode): final_hybrid_band_gap_model.joblib (saved in ./models/)
"""

import pandas as pd
import numpy as np
import os
import joblib
import lightgbm as lgb
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from tqdm import tqdm

# ✅ Import the HybridBandGapModel from central file
from models.hybrid_class import HybridBandGapModel

# ==============================================================================
# --- CONFIGURATION ---
# ==============================================================================
MODE = "train"  # "train" = train and save model, "predict" = load and predict
RUN_OPTUNA_SEARCH = True  # Only used in train mode
DATA_FILE = os.path.join('data', 'inorganic_materials_featurized(1).csv')
FINAL_MODEL_PATH = os.path.join('.', 'models', 'final_hybrid_band_gap_model.joblib')
N_TRIALS_CLS = 50
N_TRIALS_REG = 100

# ==============================================================================
# --- TRAIN MODE ---
# ==============================================================================
if MODE == "train":
    print("\n--- Step 1: Loading Data ---")
    df = pd.read_csv(DATA_FILE)
    df = df.dropna(subset=['formation_energy_per_atom', 'band_gap'])
    df['is_metal'] = (df['band_gap'] == 0).astype(int)

    cols_to_drop = ['material_id', 'pretty_formula', 'composition',
                    'formation_energy_per_atom', 'band_gap', 'is_metal']
    cols_that_exist = [col for col in cols_to_drop if col in df.columns]
    X = df.drop(columns=cols_that_exist)
    y = df[['band_gap', 'is_metal']]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y['is_metal']
    )
    print("✅ Data loaded and prepared.")

    if RUN_OPTUNA_SEARCH:
        print("\n--- Step 2: Finding Best Hyperparameters ---")
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective_cls(trial):
            params = {
                'objective': 'binary',
                'metric': 'logloss',
                'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'max_depth': trial.suggest_int('max_depth', 5, 20),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
                'verbose': -1, 'n_jobs': -1, 'seed': 42
            }
            model = lgb.LGBMClassifier(**params)
            model.fit(X_train, y_train['is_metal'])
            return model.score(X_test, y_test['is_metal'])

        with tqdm(total=N_TRIALS_CLS, desc="Optimizing Classifier") as pbar:
            def callback_cls(study, trial): pbar.update(1)
            study_cls = optuna.create_study(direction='maximize')
            study_cls.optimize(objective_cls, n_trials=N_TRIALS_CLS, callbacks=[callback_cls])
        best_params_cls = study_cls.best_params

        print(f"✅ Best Classifier Params: {best_params_cls}")

        X_train_reg = X_train.loc[y_train['is_metal'] == 0]
        y_train_reg = y_train.loc[y_train['is_metal'] == 0]['band_gap']
        X_test_reg = X_test.loc[y_test['is_metal'] == 0]
        y_test_reg = y_test.loc[y_test['is_metal'] == 0]['band_gap']

        def objective_reg(trial):
            params = {
                'objective': 'regression_l1',
                'metric': 'mae',
                'n_estimators': trial.suggest_int('n_estimators', 200, 1500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'verbose': -1, 'n_jobs': -1, 'seed': 42
            }
            model = lgb.LGBMRegressor(**params)
            model.fit(X_train_reg, y_train_reg)
            preds = model.predict(X_test_reg)
            return mean_absolute_error(y_test_reg, preds)

        with tqdm(total=N_TRIALS_REG, desc="Optimizing Regressor") as pbar:
            def callback_reg(study, trial): pbar.update(1)
            study_reg = optuna.create_study(direction='minimize')
            study_reg.optimize(objective_reg, n_trials=N_TRIALS_REG, callbacks=[callback_reg])
        best_params_reg = study_reg.best_params

        print(f"✅ Best Regressor Params: {best_params_reg}")

    else:
        print("\n--- Step 2: Using Pre-Saved Best Parameters ---")
        best_params_cls = {
            'objective': 'binary',
            'metric': 'logloss',
            'n_estimators': 876,
            'learning_rate': 0.1517684,
            'num_leaves': 82,
            'max_depth': 18,
            'feature_fraction': 0.8938953
        }
        best_params_reg = {
            'objective': 'regression_l1',
            'metric': 'mae',
            'n_estimators': 1489,
            'learning_rate': 0.01049219,
            'num_leaves': 148,
            'min_child_samples': 6
        }

    print("\n--- Step 3: Training Final Model ---")
    final_band_gap_model = HybridBandGapModel(cls_params=best_params_cls, reg_params=best_params_reg)
    final_band_gap_model.fit(X_train, y_train['band_gap'])

    print("\n--- Step 4: Evaluation ---")
    preds = final_band_gap_model.predict(X_test)
    print(f"R²: {r2_score(y_test['band_gap'], preds):.4f}")
    print(f"MAE: {mean_absolute_error(y_test['band_gap'], preds):.4f} eV")

    print("\n--- Step 5: Saving Model ---")
    os.makedirs(os.path.dirname(FINAL_MODEL_PATH), exist_ok=True)
    joblib.dump(final_band_gap_model, FINAL_MODEL_PATH)
    print(f"✅ Model saved to {FINAL_MODEL_PATH}")

# ==============================================================================
# --- PREDICT MODE ---
# ==============================================================================
elif MODE == "predict":
    print(f"Loading model from: {FINAL_MODEL_PATH}")
    if not os.path.exists(FINAL_MODEL_PATH):
        print("❌ Model file not found. Please run in train mode first.")
        exit()

    band_gap_model = joblib.load(FINAL_MODEL_PATH)
    print("✅ Model loaded.")

    print("\n--- Loading New Data ---")
    df_new_samples = pd.read_csv(DATA_FILE).head(10)
    cols_to_drop = ['material_id', 'pretty_formula', 'composition',
                    'formation_energy_per_atom', 'band_gap', 'is_metal']
    cols_that_exist = [col for col in cols_to_drop if col in df_new_samples.columns]
    X_new = df_new_samples.drop(columns=cols_that_exist)

    print("\n--- Making Predictions ---")
    predicted_values = band_gap_model.predict(X_new)

    results_df = pd.DataFrame({
        'Formula': df_new_samples['pretty_formula'],
        'Actual_Band_Gap': df_new_samples.get('band_gap', pd.Series([None]*len(X_new))),
        'Predicted_Band_Gap': np.round(predicted_values, 4)
    })
    print(results_df.to_string())

else:
    print("❌ Invalid MODE. Choose 'train' or 'predict'.")
