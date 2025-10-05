# app_material_suite.py

import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
import random
import warnings
from pymatgen.core import Composition
from matminer.featurizers.conversions import StrToComposition
from stable_baselines3 import PPO
import gymnasium as gym
from gymnasium import spaces

# --- Suppress Warnings ---
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# ==============================================================================
# --- FILE PATHS AND STATIC DATA ---
# ==============================================================================

BASE_PATH = './data'
MODELS_PATH = './models'
RL_OUTPUT_PATH = './RL_output'

# Define paths for all required files
FEATURIZER_PATH = os.path.join(BASE_PATH, "featurizer.joblib")
KNOWN_FORMULAS_PATH = os.path.join(BASE_PATH, "known_materials_set.pkl")
BAND_GAP_MODEL_PATH = os.path.join(MODELS_PATH, "Real_hybrid_band_gap_model.joblib")
STABILITY_MODEL_PATH = os.path.join(MODELS_PATH, "Real_stability_model.joblib")
RL_AGENT_PATH = os.path.join(RL_OUTPUT_PATH, "agent_ppo_final.zip")

# Pre-defined material data for the 'Explore' tab
materials_data = {
    "Metals": {
        "description": "Metals generally have a zero band gap.",
        "properties": [
            {"Material": "Iron (Fe)", "Formula": "Fe", "Band Gap (eV)": 0, "Formation Energy": "Varies"},
            {"Material": "Copper (Cu)", "Formula": "Cu", "Band Gap (eV)": 0, "Formation Energy": "Varies"},
            {"Material": "Aluminum (Al)", "Formula": "Al", "Band Gap (eV)": 0, "Formation Energy": "Varies"},
            {"Material": "Silver (Ag)", "Formula": "Ag", "Band Gap (eV)": 0, "Formation Energy": "Varies"},
            {"Material": "Gold (Au)", "Formula": "Au", "Band Gap (eV)": 0, "Formation Energy": "Varies"},
        ]
    },
    "Semiconductors": {
        "description": "Band gap: ~0.1–3eV",
        "properties": [
            {"Material": "Silicon (Si)", "Formula": "Si", "Band Gap (eV)": 1.12, "Formation Energy": "~−4.62eV/atom"},
            {"Material": "Germanium (Ge)", "Formula": "Ge", "Band Gap (eV)": 0.67, "Formation Energy": "~−4.13eV/atom"},
            {"Material": "Gallium Arsenide (GaAs)", "Formula": "GaAs", "Band Gap (eV)": 1.42, "Formation Energy": "~−0.65eV/atom"},
            {"Material": "Indium Phosphide (InP)", "Formula": "InP", "Band Gap (eV)": 1.35, "Formation Energy": "~−1.8eV/atom"},
            {"Material": "Silicon Carbide (SiC, 4H)", "Formula": "SiC", "Band Gap (eV)": 3.3, "Formation Energy": "~−0.62eV/atom"},
        ]
    },
    "Insulators / Wide-gap Semiconductors": {
        "description": "Band gap: >3eV",
        "properties": [
            {"Material": "Diamond (C)", "Formula": "C", "Band Gap (eV)": 5.47, "Formation Energy": "−1.9eV/atom"},
            {"Material": "Boron Nitride (c-BN)", "Formula": "BN", "Band Gap (eV)": 6.36, "Formation Energy": "−3.5eV/atom"},
            {"Material": "Silicon Dioxide (SiO2)", "Formula": "SiO2", "Band Gap (eV)": 9.0, "Formation Energy": "−8.4eV/atom"},
        ]
    },
    "Oxides": {
        "description": "A class of materials containing oxygen.",
        "properties": [
            {"Material": "Zinc Oxide (ZnO)", "Formula": "ZnO", "Band Gap (eV)": 3.37, "Formation Energy": "−3.5eV/atom"},
            {"Material": "Tin Oxide (SnO2)", "Formula": "SnO2", "Band Gap (eV)": "3.6–3.7", "Formation Energy": "−5.3eV/atom"},
            {"Material": "Titanium Dioxide (TiO2)", "Formula": "TiO2", "Band Gap (eV)": "3.0–3.2", "Formation Energy": "−9.7eV/unit"},
        ]
    },
}

# ==============================================================================
# --- RL ENVIRONMENT AND GENERATION LOGIC ---
# ==============================================================================

class MaterialDiscoveryEnv(gym.Env):
    """A modified environment that accepts dynamic targets for generation."""
    def __init__(self, stability_model, bandgap_model, featurizer, element_list,
                 known_formulas_set, model_feature_list, target_band_gap,
                 target_formation_energy, max_steps=25):
        super().__init__()
        self.stability_model, self.bandgap_model = stability_model, bandgap_model
        self.featurizer, self.element_list = featurizer, element_list
        self.known_formulas_set, self.model_feature_list = known_formulas_set, model_feature_list
        self.n_features = len(model_feature_list)
        
        self.target_band_gap, self.target_formation_energy = target_band_gap, target_formation_energy
        self.max_steps, self.n_elements = max_steps, len(element_list)
        
        self.action_space = spaces.Discrete(2 * self.n_elements)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_features,), dtype=np.float32)

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

        stability_reward = 5.0 * np.exp(-((pred_stability - self.target_formation_energy)**2) / (2 * 0.5**2))
        bandgap_reward = 5.0 * np.exp(-((pred_bandgap - self.target_band_gap)**2) / (2 * 1.0**2))
        
        if self.target_band_gap > 0 and pred_bandgap == 0: bandgap_reward -= 2.0
            
        novelty_reward = 1.0 if formula not in self.known_formulas_set else -2.0
        return (stability_reward + bandgap_reward + novelty_reward, pred_stability, pred_bandgap)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.current_composition = Composition({random.choice(self.element_list): 1})
        return self._get_obs(self.current_composition), {}

    def step(self, action_id):
        elements = self.current_composition.get_el_amt_dict()
        element_symbol = self.element_list[action_id % self.n_elements]

        if 0 <= action_id < self.n_elements: # Add action
            elements[element_symbol] = elements.get(element_symbol, 0) + 1
        else: # Remove action
            if element_symbol in elements:
                elements[element_symbol] = max(0, elements[element_symbol] - 1)
                if elements[element_symbol] == 0: del elements[element_symbol]

        if not elements: elements = {random.choice(self.element_list): 1}
            
        self.current_composition = Composition(elements)
        obs = self._get_obs(self.current_composition)
        reward, stability, bandgap = self._calculate_reward(obs, self.current_composition.reduced_formula)
        
        self.current_step += 1
        done = self.current_step >= self.max_steps
        info = {'formula': self.current_composition.reduced_formula, 'pred_stability': stability, 'pred_bandgap': bandgap, 'reward': reward}
        return obs, reward, done, False, info

def generate_materials_with_rl(agent, models, data, user_params, num_episodes, progress_bar):
    """Runs the RL agent to generate materials based on user inputs."""
    stability_model, bandgap_model = models['stability'], models['bandgap']
    featurizer, known_formulas = data['featurizer'], data['known_formulas']
    model_feature_list = stability_model.booster_.feature_name()

    env = MaterialDiscoveryEnv(stability_model, bandgap_model, featurizer, user_params['elements'], 
                             known_formulas, model_feature_list, user_params['band_gap'], user_params['formation_energy'])

    discovered = {}
    for i in range(num_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            formula = info.get('formula')
            if formula and formula not in known_formulas and len(Composition(formula)) > 1:
                if formula not in discovered or reward > discovered[formula].get('reward', -np.inf):
                    discovered[formula] = {k: v for k, v in info.items() if k != 'formula'}
        progress_bar.progress((i + 1) / num_episodes)

    if not discovered: return pd.DataFrame()

    df = pd.DataFrame.from_dict(discovered, orient='index').reset_index().rename(columns={'index': 'formula'})
    stability_dist = np.abs(df['pred_stability'] - user_params['formation_energy'])
    bandgap_dist = np.abs(df['pred_bandgap'] - user_params['band_gap'])
    df['promise_score'] = np.exp(-stability_dist) + np.exp(-bandgap_dist)
    return df.sort_values(by='promise_score', ascending=False, ignore_index=True)

# ==============================================================================
# --- CORE APP FUNCTIONS ---
# ==============================================================================

@st.cache_resource
def load_all_assets():
    """Load all models and data assets, returning None if any are missing."""
    required_files = [FEATURIZER_PATH, KNOWN_FORMULAS_PATH, BAND_GAP_MODEL_PATH, STABILITY_MODEL_PATH, RL_AGENT_PATH]
    if not all(os.path.exists(p) for p in required_files): return None
    
    models = {
        'bandgap': joblib.load(BAND_GAP_MODEL_PATH),
        'stability': joblib.load(STABILITY_MODEL_PATH),
        'rl_agent': PPO.load(RL_AGENT_PATH)
    }
    data = {
        'featurizer': joblib.load(FEATURIZER_PATH),
        'known_formulas': joblib.load(KNOWN_FORMULAS_PATH)
    }
    return models, data

def featurize_formula(formula: str, featurizer) -> pd.DataFrame:
    """Convert a chemical formula string into a feature vector."""
    df = pd.DataFrame({"pretty_formula": [formula]})
    stc = StrToComposition(target_col_id="composition")
    df = stc.featurize_dataframe(df, "pretty_formula", ignore_errors=True)
    if "composition" not in df.columns or df["composition"].isnull().any():
        raise ValueError(f"Invalid formula '{formula}' or unsupported element.")
    
    df_features = featurizer.featurize_dataframe(df, col_id="composition", ignore_errors=True)
    if df_features.empty or df_features.drop(columns=["pretty_formula", "composition"], errors="ignore").isnull().values.any():
        raise ValueError(f"Could not generate valid features for '{formula}'.")
        
    return df_features.drop(columns=["pretty_formula", "composition"], errors="ignore")


# ==============================================================================
# --- STREAMLIT UI ---
# ==============================================================================

st.set_page_config(page_title="Material Discovery Suite", page_icon="🔬", layout="wide")
st.title("🔬 AI-Powered Material Discovery Suite")

# --- Load assets and handle errors ---
asset_load_result = load_all_assets()
if asset_load_result is None:
    st.error("**Critical Error:** One or more essential model or data files are missing. Please ensure all files are in the correct directories (`/data`, `/models`, `/RL_output`).")
    st.stop()
models, data = asset_load_result


# --- Define Tabs ---
tab1, tab2, tab3 = st.tabs([
    "✨ Generate Novel Materials", 
    "📊 Explore & Compare Known Materials",
    "🧪 Predict by Formula"
])

# --- TAB 1: GENERATE NOVEL MATERIALS (New Interactive Feature) ---
with tab1:
    st.header("✨ Generate Novel Materials with AI")
    st.markdown(
        "Define your desired properties and select elemental building blocks. "
        "The AI agent will explore chemical combinations to find novel materials matching your criteria."
    )

    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("1. Define Target Properties")
        target_bg = st.slider("Target Band Gap (eV)", 0.0, 10.0, 3.0, 0.1)
        target_fe = st.slider("Target Formation Energy (eV/atom)", -6.0, 0.0, -2.0, 0.1, help="More negative values mean higher stability.")
        
        st.subheader("2. Set Generation Parameters")
        episodes = st.number_input("Number of Generation Episodes", 20, 1000, 150, 20, help="More episodes = longer search, potentially better results.")
        generate_button = st.button("🚀 Generate Materials", use_container_width=True, type="primary")

    with col2:
        st.subheader("3. Select Available Elements")
        all_elements = ['H','Li','Be','B','C','N','O','F','Na','Mg','Al','Si','P','S','Cl','K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga','Ge','As','Se','Br','Rb','Sr','Y','Zr','Nb','Mo','Ag','Cd','In','Sn','Sb','Te','I','Cs','Ba','La','Hf','Ta','W','Pt','Au','Pb','Bi']
        
        # Create a more compact grid for element selection
        element_cols = st.columns(12)
        selected_elements = []
        for i, element in enumerate(all_elements):
            if element_cols[i % 12].checkbox(element, key=f"elem_{element}"):
                selected_elements.append(element)
    
    st.divider()

    if generate_button:
        if len(selected_elements) < 2:
            st.error("Please select at least two elements to combine.")
        else:
            with st.spinner("AI agent is exploring... This may take a moment."):
                progress_bar = st.progress(0, text="Generation in progress...")
                user_params = {'band_gap': target_bg, 'formation_energy': target_fe, 'elements': selected_elements}
                df_results = generate_materials_with_rl(models['rl_agent'], models, data, user_params, episodes, progress_bar)
                progress_bar.empty()
                st.session_state['generated_materials'] = df_results

    if 'generated_materials' in st.session_state:
        df_display = st.session_state['generated_materials']
        st.subheader("Discovered Candidate Materials")
        if not df_display.empty:
            st.success(f"Found {len(df_display)} unique, novel candidates!")
            st.dataframe(df_display[['formula', 'pred_bandgap', 'pred_stability', 'promise_score']].rename(columns={
                'formula': 'Formula', 'pred_bandgap': 'Predicted BG (eV)',
                'pred_stability': 'Predicted FE (eV/atom)', 'promise_score': 'Promise Score'}), use_container_width=True)
        else:
            st.warning("No novel materials found. Try more episodes or different elements.")

# --- TAB 2: EXPLORE & COMPARE KNOWN MATERIALS ---
with tab2:
    st.header("📊 Explore & Compare Model Predictions vs. Known Data")
    selected_category = st.selectbox("1. Select Category", list(materials_data.keys()), key="category_select")
    if selected_category:
        materials_list = materials_data[selected_category]['properties']
        material_names = [m['Material'] for m in materials_list]
        selected_material_name = st.selectbox("2. Select Material", material_names, key="material_select")
        
        if st.button("Predict and Compare", key="compare_button"):
            details = next((m for m in materials_list if m["Material"] == selected_material_name), None)
            if details:
                st.write(f"### Results for: **{selected_material_name}** (`{details['Formula']}`)")
                try:
                    with st.spinner(f"Predicting for {details['Formula']}..."):
                        features = featurize_formula(details['Formula'], data['featurizer'])
                        bg_pred = models['bandgap'].predict(features)[0]
                        stab_pred = models['stability'].predict(features)[0]
                        
                        col_pred, col_actual = st.columns(2)
                        with col_pred:
                            st.subheader("Predicted Values (Model)")
                            st.metric("Predicted Band Gap", f"{bg_pred:.3f} eV")
                            st.metric("Predicted Formation Energy", f"{stab_pred:.3f} eV/atom")
                        with col_actual:
                            st.subheader("Actual Values (Dataset)")
                            st.metric("Actual Band Gap", f"{details['Band Gap (eV)']} eV")
                            st.metric("Actual Formation Energy", f"{details['Formation Energy']}")
                except Exception as e:
                    st.error(f"Prediction failed for `{details['Formula']}`: {e}")

# --- TAB 3: PREDICT BY FORMULA ---
with tab3:
    st.header("🧪 Predict Properties from a Chemical Formula")
    formula_input = st.text_input("Enter Chemical Formula", "ZnO", key="formula_input")
    if st.button("Predict", key="predict_button"):
        if formula_input:
            try:
                with st.spinner("Predicting..."):
                    features = featurize_formula(formula_input, data['featurizer'])
                    bg_pred = models['bandgap'].predict(features)[0]
                    stab_pred = models['stability'].predict(features)[0]
                    st.success(f"Predictions for **{formula_input}**")
                    col1, col2 = st.columns(2)
                    col1.metric("Predicted Band Gap", f"{bg_pred:.3f} eV")
                    col2.metric("Predicted Formation Energy", f"{stab_pred:.3f} eV/atom")
            except Exception as e:
                st.error(f"Prediction Failed: {e}")