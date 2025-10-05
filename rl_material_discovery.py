"""
RL Non-Metal Material Discovery

This script retrains a PPO reinforcement learning agent to discover stable, 
non-metallic materials with target band gaps. It uses pre-trained models for 
stability and band gap predictions, along with a featurizer from matminer.

Files used:
- Real_stability_model.joblib : Pre-trained stability model
- Real_hybrid_band_gap_model.joblib : Pre-trained hybrid band gap model
- featurizer.joblib : Matminer-based featurizer
- known_materials_set.pkl : Set of known material formulas

Files generated:
- RL_output/agent_ppo_final.zip : Trained RL agent
- RL_output/candidate_materials.csv : Predicted novel candidates
"""

import os
import pickle
import random
import time
import warnings
import datetime

import gymnasium as gym
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from gymnasium import spaces
from matminer.featurizers.composition import ElementFraction, ElementProperty, Stoichiometry
from pymatgen.core import Composition
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from sklearn.base import BaseEstimator, RegressorMixin
from tqdm.auto import tqdm

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Class definitions (HybridBandGapModel) remain unchanged ---
class HybridBandGapModel(BaseEstimator, RegressorMixin):
    def __init__(self, cls_params=None, reg_params=None):
        self.cls_params, self.reg_params = cls_params, reg_params
    def fit(self, X, y):
        self.classifier_ = lgb.LGBMClassifier(**(self.cls_params or {}), random_state=42, verbose=-1).fit(X, (y == 0).astype(int))
        non_metal_indices = y[y > 0].index
        self.regressor_ = lgb.LGBMRegressor(**(self.reg_params or {}), random_state=42, verbose=-1).fit(X.loc[non_metal_indices], y.loc[non_metal_indices])
        return self
    def predict(self, X):
        if not isinstance(X, pd.DataFrame): raise TypeError("Input X must be a pandas DataFrame.")
        is_metal_preds = self.classifier_.predict(X)
        regressor_preds = self.regressor_.predict(X)
        return np.where(is_metal_preds == 1, 0.0, regressor_preds)
    def predict_proba_non_metal(self, X):
        if not isinstance(X, pd.DataFrame): raise TypeError("Input X must be a pandas DataFrame.")
        return self.classifier_.predict_proba(X)[:, 0]

# ==============================================================================

class MaterialDiscoveryEnv(gym.Env):
    def __init__(self, stability_model, bandgap_model, featurizer, element_list,
                 known_formulas_set, model_feature_list, target_band_gap=2.5,
                 max_steps=25, stability_weight=1.5):
        super().__init__()
        self.stability_model, self.bandgap_model = stability_model, bandgap_model
        self.featurizer, self.element_list = featurizer, element_list
        self.known_formulas_set, self.model_feature_list = known_formulas_set, model_feature_list
        self.n_features = len(model_feature_list)
        self.target_band_gap, self.max_steps = target_band_gap, max_steps
        self.stability_weight, self.n_elements = stability_weight, len(element_list)
        self.action_space = spaces.Discrete(2 * self.n_elements)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_features,), dtype=np.float32)
        self.current_composition = None
        self.current_step = 0

    def _get_obs(self, composition):
        try:
            features_df = self.featurizer.featurize_dataframe(
                pd.DataFrame({'composition': [composition]}), col_id='composition',
                ignore_errors=True, pbar=False).drop(columns=['composition'])
            return features_df.reindex(columns=self.model_feature_list, fill_value=0).iloc[0].values.astype(np.float32)
        except Exception:
            return np.zeros(self.n_features, dtype=np.float32)

    def _calculate_reward(self, obs, formula):
        feature_df = pd.DataFrame([obs], columns=self.model_feature_list)
        pred_stability = self.stability_model.predict(feature_df)[0]
        pred_bandgap = self.bandgap_model.predict(feature_df)[0]

        stability_reward = self.stability_weight * max(0, -pred_stability)
        bandgap_weight = 5.0
        sigma = 1.0
        bandgap_reward = bandgap_weight * np.exp(-((pred_bandgap - self.target_band_gap)**2) / (2 * sigma**2))
        if pred_bandgap == 0:
            bandgap_reward -= 1.0
        novelty_reward = 0.5 if formula not in self.known_formulas_set else -2.0
        total_reward = stability_reward + bandgap_reward + novelty_reward
        return (total_reward, pred_stability, pred_bandgap, novelty_reward)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.current_composition = Composition({random.choice(self.element_list): 1})
        return self._get_obs(self.current_composition), {'composition': str(self.current_composition.formula)}

    def step(self, action_id):
        elements = self.current_composition.get_el_amt_dict()
        if 0 <= action_id < self.n_elements:
            elements[self.element_list[action_id]] = elements.get(self.element_list[action_id], 0) + 1
        else:
            elem_to_remove = self.element_list[action_id - self.n_elements]
            if elem_to_remove in elements:
                elements[elem_to_remove] -= 1
                if elements[elem_to_remove] <= 0: del elements[elem_to_remove]
        if not elements: elements = {random.choice(self.element_list): 1}
        self.current_composition = Composition(elements)
        obs = self._get_obs(self.current_composition)
        reward, stability, bandgap, novelty = self._calculate_reward(obs, self.current_composition.reduced_formula)
        done = self.current_step >= self.max_steps
        self.current_step += 1
        info = {'formula': self.current_composition.reduced_formula, 'pred_stability': stability, 'pred_bandgap': bandgap, 'reward': reward, 'novelty_reward': novelty}
        return obs, reward, done, False, info

# --- Callback for progress bar ---
class TqdmCallback(BaseCallback):
    def __init__(self, total_steps):
        super().__init__()
        self.pbar = None
        self.total_steps = total_steps
    def _on_training_start(self) -> None:
        self.pbar = tqdm(total=self.total_steps, initial=self.model.num_timesteps, desc="Training RL Agent", unit="step")
    def _on_step(self) -> bool:
        self.pbar.update(1)
        return True
    def _on_training_end(self) -> None:
        self.pbar.close()

# --- Main Execution ---
if __name__ == '__main__':
    script_start_time = time.time()

    BASE_PATH = './data'  # Local path where models and data are stored
    OUTPUT_PATH = './RL_output' 
    CHECKPOINT_DIR = os.path.join(OUTPUT_PATH, 'checkpoints')
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    TOTAL_TRAINING_STEPS, EVALUATION_EPISODES = 200_000, 500
    TARGET_BAND_GAP, CHECKPOINT_FREQ = 2.5, 5000

    print("\n--- Loading Models and Data ---")
    stability_model = joblib.load(os.path.join(BASE_PATH, 'Real_stability_model.joblib'))
    bandgap_model = joblib.load(os.path.join(BASE_PATH, 'Real_hybrid_band_gap_model.joblib'))
    featurizer = joblib.load(os.path.join(BASE_PATH, 'featurizer.joblib'))
    with open(os.path.join(BASE_PATH, 'known_materials_set.pkl'), 'rb') as f:
        known_formulas_set = pickle.load(f)
    model_feature_list = stability_model.booster_.feature_name()
    element_list = ['H','Li','Be','B','C','N','O','F','Na','Mg','Al','Si','P','S','Cl','K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga','Ge','As','Se','Br','Rb','Sr','Y','Zr','Nb','Mo','Ag','Cd','In','Sn','Sb','Te','I','Cs','Ba','La','Hf','Ta','W','Pt','Au','Pb','Bi']

    env = MaterialDiscoveryEnv(stability_model, bandgap_model, featurizer, element_list, known_formulas_set, model_feature_list, TARGET_BAND_GAP)
    checkpoint_callback = CheckpointCallback(save_freq=CHECKPOINT_FREQ, save_path=CHECKPOINT_DIR, name_prefix='rl_agent')

    latest_checkpoint, agent, remaining_steps = None, None, TOTAL_TRAINING_STEPS
    if os.path.exists(CHECKPOINT_DIR):
        checkpoints = [os.path.join(CHECKPOINT_DIR, f) for f in os.listdir(CHECKPOINT_DIR) if f.endswith('.zip')]
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=os.path.getctime)

    if latest_checkpoint:
        print(f"\n--- Resuming Training from: {os.path.basename(latest_checkpoint)} ---")
        agent = PPO.load(latest_checkpoint, env=env)
        already_trained_steps = agent.num_timesteps
        remaining_steps = TOTAL_TRAINING_STEPS - already_trained_steps
        print(f"   Steps already completed: {already_trained_steps}")
        if remaining_steps > 0: print(f"   Steps remaining: {remaining_steps}")
    else:
        print(f"\n--- Starting New Training Run ---")
        agent = PPO("MlpPolicy", env, verbose=0)

    if remaining_steps > 0:
        agent.learn(
            total_timesteps=remaining_steps,
            callback=[TqdmCallback(TOTAL_TRAINING_STEPS), checkpoint_callback],
            reset_num_timesteps=False
        )
    agent_filename = 'agent_ppo_final.zip'
    agent_path = os.path.join(OUTPUT_PATH, agent_filename)
    agent.save(agent_path)
    print(f"\n📦 Final agent saved to: {agent_path}")

    # --- Evaluation ---
    discovered_materials = {}
    for _ in tqdm(range(EVALUATION_EPISODES), desc="Generating Candidates", unit="episode"):
        obs, info = env.reset(); done = False
        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            formula = info['formula']
            if formula not in known_formulas_set and formula not in discovered_materials:
                 discovered_materials[formula] = {'pred_stability': info['pred_stability'],'pred_bandgap': info['pred_bandgap']}

    if discovered_materials:
        df_candidates = pd.DataFrame.from_dict(discovered_materials, orient='index').reset_index(names='formula')
        def featurize_for_prediction(formula):
            df = pd.DataFrame({'composition': [Composition(formula)]})
            features_df = featurizer.featurize_dataframe(df, 'composition', ignore_errors=True, pbar=False).drop(columns=['composition'])
            return features_df.reindex(columns=model_feature_list, fill_value=0)
        all_features_df = pd.concat([featurize_for_prediction(f) for f in df_candidates['formula']], ignore_index=True)
        df_candidates['non_metal_confidence'] = bandgap_model.predict_proba_non_metal(all_features_df)
        stability_score = np.maximum(0, -df_candidates['pred_stability'])
        bandgap_score = np.exp(-((df_candidates['pred_bandgap'] - TARGET_BAND_GAP)**2) / (2 * 1.0**2))
        df_candidates['promise_score'] = stability_score * bandgap_score
        df_candidates.sort_values(by='promise_score', ascending=False, inplace=True, ignore_index=True)
        candidates_path = os.path.join(OUTPUT_PATH, 'candidate_materials.csv')
        df_candidates.to_csv(candidates_path, index=False)
        print(f"✅ Top candidates saved to: {candidates_path}")
        print("\n--- Top 5 Most Promising Candidates ---")
        print(df_candidates.head(5).to_string())
    else:
        print("\n--- No novel materials were discovered during evaluation. ---")

    total_seconds = time.time() - script_start_time
    total_runtime = str(datetime.timedelta(seconds=int(total_seconds)))
    print(f"\n--- RL Script Finished ---\nTotal runtime: {total_runtime}")
