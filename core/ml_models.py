"""
Machine Learning Models for Freshwater Algae Bloom Detection & Risk Prediction
Uses multi-spectral Landsat features, thermal dynamics, and spatial morphology.

Implements:
1. Random Forest & Hist-Gradient Boosting Bloom Classifiers (Low / Watch / Warning / Critical)
2. Continuous Bloom Severity Regressor (FAI / Chlorophyll proxy prediction)
3. Model evaluation metrics (F1-score, ROC-AUC, R², RMSE)
4. Explainable AI: Feature importance ranking (Thermal vs Optical vs Spatial features)
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional, Any
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, r2_score, mean_squared_error, f1_score


class BloomMachineLearningEngine:
    """
    Supervised learning framework mapping surface temperature patterns and EO optical indices to bloom risk.
    """
    
    FEATURE_NAMES = [
        "lst_current",               # Current Land/Lake Surface Temperature (°C)
        "lst_3d_mean",               # 3-day rolling mean temperature (°C)
        "lst_7d_mean",               # 7-day rolling mean temperature (°C)
        "consecutive_hot_days",      # Number of consecutive days >= 20°C
        "cdd_20",                    # Cumulative Degree Days above 20°C (14-day)
        "thermal_anomaly",           # Anomaly relative to seasonal expectation (°C)
        "turbidity_index",           # Turbidity proxy (NDTI = (Red - Green) / (Red + Green))
        "system_scale_code",         # 0=Small/River, 1=Medium Reservoir, 2=Large Lake
        "shoreline_dist_norm",       # Distance from shoreline normalized [0, 1]
        "solar_insolation_factor"    # Seasonal solar angle factor [0.4, 1.0]
    ]

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.classifier = HistGradientBoostingClassifier(
            max_iter=150, 
            learning_rate=0.08, 
            max_leaf_nodes=31, 
            random_state=random_state
        )
        self.regressor = HistGradientBoostingRegressor(
            max_iter=150, 
            learning_rate=0.08, 
            max_leaf_nodes=31, 
            random_state=random_state
        )
        self.rf_classifier = RandomForestClassifier(
            n_estimators=100, 
            max_depth=10, 
            random_state=random_state, 
            n_jobs=-1
        )
        self.is_trained = False
        self.feature_importance_: Dict[str, float] = {}
        self.metrics_: Dict[str, Any] = {}

    def generate_synthetic_training_corpus(self, n_samples: int = 5000) -> pd.DataFrame:
        """
        Synthesize scientifically grounded freshwater training samples based on limnological literature
        (e.g., Paerl & Huisman 2008, Michalak et al. 2013 on Lake Erie, Dokulil & Teubner 2000 on European lakes).
        
        Models the non-linear relationship between consecutive hot days, thermal stratification,
        turbidity, and cyanobacterial bloom eruption.
        """
        np.random.seed(self.random_state)
        
        # Temperature distribution: 12C to 30C
        lst_current = np.random.uniform(12.0, 31.0, n_samples)
        
        # 3-day and 7-day lags correlate with current temp with random walk
        lst_3d_mean = lst_current + np.random.normal(-0.5, 1.2, n_samples)
        lst_7d_mean = lst_3d_mean + np.random.normal(-0.8, 1.5, n_samples)
        
        # Consecutive hot days (more likely when temps are high)
        hot_prob = np.clip((lst_current - 18.0) / 10.0, 0.0, 1.0)
        consecutive_hot_days = np.random.geometric(p=np.clip(1.0 - hot_prob * 0.7, 0.1, 1.0), size=n_samples) - 1
        consecutive_hot_days = np.where(lst_current < 19.0, np.random.choice([0, 1], p=[0.85, 0.15], size=n_samples), consecutive_hot_days)
        consecutive_hot_days = np.clip(consecutive_hot_days, 0, 18)
        
        # Cumulative Degree Days (CDD_20)
        cdd_20 = np.maximum(0.0, lst_7d_mean - 20.0) * (consecutive_hot_days * 0.8 + 1.5) + np.random.uniform(0, 3, n_samples)
        cdd_20 = np.clip(cdd_20, 0.0, 75.0)
        
        # Thermal anomaly (-4C to +6C)
        thermal_anomaly = lst_current - (18.0 + 3.0 * np.sin(np.random.uniform(0, np.pi, n_samples)))
        
        # Turbidity (NDTI: -0.3 to 0.4)
        turbidity_index = np.random.uniform(-0.3, 0.4, n_samples)
        
        # System scale: 0 = small/river, 1 = medium, 2 = large
        system_scale_code = np.random.choice([0, 1, 2], p=[0.25, 0.40, 0.35], size=n_samples)
        
        # Distance from shore: 0.0 (bank) to 1.0 (pelagic center)
        shoreline_dist_norm = np.random.beta(2, 2, n_samples)
        
        # Solar insolation factor: 0.4 (winter/overcast) to 1.0 (summer solstice clear sky)
        solar_insolation = np.clip(0.4 + (lst_current - 12.0) / 32.0 + np.random.normal(0, 0.08, n_samples), 0.4, 1.0)
        
        # Bio-physical Bloom Response Equation:
        # Logistic thermal activation: cyanobacteria growth surges past 20-22°C
        thermal_driver = 1.0 / (1.0 + np.exp(-(lst_current - 22.5) / 1.8))
        streak_multiplier = 1.0 + 0.15 * consecutive_hot_days
        cdd_factor = np.clip(cdd_20 / 30.0, 0.0, 1.5)
        solar_factor = solar_insolation ** 1.5
        
        # Sheltered embayments and shallow shores (low dist from shore) bloom more readily
        bay_effect = 1.1 - 0.25 * shoreline_dist_norm
        
        # Combine drivers to create true Bloom Severity Index (BSI: 0 to 1)
        latent_bsi = (0.45 * thermal_driver * streak_multiplier + 
                      0.25 * cdd_factor + 
                      0.15 * solar_factor + 
                      0.10 * (turbidity_index + 0.3) * bay_effect + 
                      0.05 * (thermal_anomaly > 2.0))
                      
        bsi = np.clip(latent_bsi / 1.6 + np.random.normal(0, 0.04, n_samples), 0.0, 1.0)
        
        # Target classes:
        # 0: Low/Clear (< 0.20), 1: Watch (0.20-0.40), 2: Warning (0.40-0.65), 3: Critical (>= 0.65)
        target_tier = np.zeros(n_samples, dtype=int)
        target_tier[(bsi >= 0.20) & (bsi < 0.40)] = 1
        target_tier[(bsi >= 0.40) & (bsi < 0.65)] = 2
        target_tier[bsi >= 0.65] = 3

        df = pd.DataFrame({
            "lst_current": lst_current,
            "lst_3d_mean": lst_3d_mean,
            "lst_7d_mean": lst_7d_mean,
            "consecutive_hot_days": consecutive_hot_days,
            "cdd_20": cdd_20,
            "thermal_anomaly": thermal_anomaly,
            "turbidity_index": turbidity_index,
            "system_scale_code": system_scale_code,
            "shoreline_dist_norm": shoreline_dist_norm,
            "solar_insolation_factor": solar_insolation,
            "bloom_severity_index": bsi,
            "hazard_tier": target_tier
        })
        return df

    def train(self, data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Train ML classification and regression models and compute explainability metrics.
        """
        if data is None:
            data = self.generate_synthetic_training_corpus(n_samples=6000)

        X = data[self.FEATURE_NAMES].values
        y_class = data["hazard_tier"].values
        y_reg = data["bloom_severity_index"].values

        X_train, X_test, y_c_train, y_c_test, y_r_train, y_r_test = train_test_split(
            X, y_class, y_reg, test_size=0.25, random_state=self.random_state, stratify=y_class
        )

        # Train Classifier (Gradient Boosting)
        self.classifier.fit(X_train, y_c_train)
        y_c_pred = self.classifier.predict(X_test)
        f1 = float(f1_score(y_c_test, y_c_pred, average="weighted"))

        # Train Regressor
        self.regressor.fit(X_train, y_r_train)
        y_r_pred = self.regressor.predict(X_test)
        r2 = float(r2_score(y_r_test, y_r_pred))
        rmse = float(np.sqrt(mean_squared_error(y_r_test, y_r_pred)))

        # Fit Random Forest for direct Gini Feature Importance
        self.rf_classifier.fit(X_train, y_c_train)
        raw_importances = self.rf_classifier.feature_importances_
        self.feature_importance_ = {
            name: round(float(imp), 4) 
            for name, imp in zip(self.FEATURE_NAMES, raw_importances)
        }
        # Sort by importance descending
        self.feature_importance_ = dict(
            sorted(self.feature_importance_.items(), key=lambda item: item[1], reverse=True)
        )

        self.is_trained = True
        self.metrics_ = {
            "test_f1_score": round(f1, 4),
            "test_r2_score": round(r2, 4),
            "test_rmse": round(rmse, 4),
            "training_samples": len(X_train),
            "test_samples": len(X_test),
            "class_distribution": {int(k): int(v) for k, v in pd.Series(y_class).value_counts().items()}
        }
        return self.metrics_

    def predict(self, feature_matrix: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Run inference on feature array (shape [N, 10] or [H, W, 10]).
        
        Returns:
            predicted_tier: 0, 1, 2, 3
            probabilities: Probability of each hazard tier
            predicted_bsi: Continuous bloom severity index [0.0, 1.0]
        """
        if not self.is_trained:
            self.train()

        orig_shape = feature_matrix.shape
        if len(orig_shape) == 3:
            h, w, c = orig_shape
            flat_features = feature_matrix.reshape(-1, c)
        else:
            flat_features = feature_matrix

        # Handle NaNs (e.g. non-water pixels)
        valid_mask = ~np.isnan(flat_features).any(axis=1)
        n_total = flat_features.shape[0]

        pred_tier = np.full(n_total, -1, dtype=int)
        pred_bsi = np.full(n_total, np.nan, dtype=float)
        probs = np.full((n_total, 4), np.nan, dtype=float)

        if np.any(valid_mask):
            valid_feats = flat_features[valid_mask]
            pred_tier[valid_mask] = self.classifier.predict(valid_feats)
            probs[valid_mask] = self.classifier.predict_proba(valid_feats)
            pred_bsi[valid_mask] = np.clip(self.regressor.predict(valid_feats), 0.0, 1.0)

        if len(orig_shape) == 3:
            return {
                "hazard_tier": pred_tier.reshape(h, w),
                "bloom_severity_index": pred_bsi.reshape(h, w),
                "probabilities": probs.reshape(h, w, 4)
            }
        else:
            return {
                "hazard_tier": pred_tier,
                "bloom_severity_index": pred_bsi,
                "probabilities": probs
            }
