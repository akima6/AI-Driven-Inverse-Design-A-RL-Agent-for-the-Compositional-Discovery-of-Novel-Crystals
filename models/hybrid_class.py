# models.py
# This file contains the definitions for custom machine learning models
# used in the materials discovery project.

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

# ==============================================================================
# --- Custom Hybrid Band Gap Model Class ---
# By defining this class in its own file, any other script can import it,
# allowing joblib to correctly load saved models that use this class.
# ==============================================================================
class HybridBandGapModel(BaseEstimator, RegressorMixin):
    """
    A custom scikit-learn compatible model that internally uses a classifier
    to separate metals from non-metals, and a regressor to predict the
    band gap for the non-metallic materials.
    """
    def __init__(self, cls_params=None, reg_params=None):
        self.cls_params = cls_params
        self.reg_params = reg_params

    def fit(self, X, y):
        """
        Trains the internal classifier and regressor.
        - X: DataFrame of features.
        - y: Series of band gap values.
        """
        # Train the classifier to predict metal (band_gap == 0 -> class 1) vs. non-metal (band_gap > 0 -> class 0)
        self.classifier_ = lgb.LGBMClassifier(**(self.cls_params or {}), random_state=42, verbose=-1)
        y_cls = (y == 0).astype(int)
        self.classifier_.fit(X, y_cls)

        # Train the regressor only on the non-metallic samples
        self.regressor_ = lgb.LGBMRegressor(**(self.reg_params or {}), random_state=42, verbose=-1)
        non_metal_indices = y[y > 0].index
        X_reg_train = X.loc[non_metal_indices]
        y_reg_train = y.loc[non_metal_indices]
        self.regressor_.fit(X_reg_train, y_reg_train)
        return self

    def predict(self, X):
        """
        Makes predictions by first classifying, then regressing.
        - X: DataFrame of features for prediction.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame.")

        # Predict which materials are metals (class 1)
        is_metal_preds = self.classifier_.predict(X)

        # Predict band gaps for all materials
        regressor_preds = self.regressor_.predict(X)

        # Use 0 for predicted metals, and the regressor's value for non-metals
        final_predictions = np.where(is_metal_preds == 1, 0.0, regressor_preds)
        return final_predictions

    def predict_proba_non_metal(self, X):
        """
        Predicts the probability of a material being a non-metal (class 0).
        This is useful for a "confidence score".
        - X: DataFrame of features for prediction.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame.")
        
        # In `fit`, we defined non-metals as class 0. This line robustly finds the
        # column corresponding to class 0's probability and returns it.
        # The result is a 1D array of probabilities.
        return self.classifier_.predict_proba(X)[:, self.classifier_.classes_ == 0].flatten()