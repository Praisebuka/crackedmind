"""
Rating Predictor — scikit-learn regression head

Design decision: keep rating prediction SEPARATE from the LLM.
The LLM handles language; a numeric regressor handles rating.
This is because:
  1. LLMs are biased toward "polite" positive ratings
  2. A trained regressor on explicit signals is far more accurate for RMSE
  3. Interpretable features = something to write about in the solution paper

Feature set:
  - user_mean_rating          → user's baseline generosity
  - user_std                  → how variable they are
  - user_negativity_rate      → how often they give ≤2 stars
  - category_match_mean       → user's mean for this category
  - category_match_count      → how many reviews in this category
  - user_review_count         → experience level
  - item_global_mean          → prior from item catalog (if available)
  - text_length_norm          → normalized expected review length
"""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pickle
import os
from typing import Optional
import structlog

log = structlog.get_logger()


def _extract_features(profile: dict, item_metadata: dict) -> np.ndarray:
    """
    Build feature vector from user profile + item metadata.
    Returns shape (1, 8) float32 array.
    """
    rs = profile.get("rating_stats", {})
    cat_prefs = profile.get("category_preferences", {}).get("by_category", {})
    target_cat = item_metadata.get("category", "")

    user_mean = rs.get("mean", 3.5)
    user_std = rs.get("std", 1.0)
    user_neg_rate = rs.get("negativity_rate", 0.15)
    user_pos_rate = rs.get("positivity_rate", 0.5)
    user_count = profile.get("review_count", 0)

    cat_data = cat_prefs.get(target_cat, {})
    cat_mean = cat_data.get("mean_rating", user_mean)
    cat_count = cat_data.get("count", 0)
    cat_known = 1.0 if cat_count > 0 else 0.0

    item_global_mean = item_metadata.get("metadata", {}).get("global_mean_rating", 3.8)

    features = np.array([
        user_mean,
        user_std,
        user_neg_rate,
        user_pos_rate,
        np.log1p(user_count),
        cat_mean,
        cat_known,
        float(item_global_mean),
    ], dtype="float32")

    return features.reshape(1, -1)


class RatingPredictor:
    """
    Thin Ridge regression wrapper.
    Trained on synthetic data in __init__ if no saved model found.
    In production: retrain on actual Yelp/Amazon user-item pairs.
    """

    MODEL_PATH = "data/rating_model.pkl"

    def __init__(self):
        self.model: Optional[Pipeline] = None
        self._load_or_init()

    def _load_or_init(self):
        if os.path.exists(self.MODEL_PATH):
            with open(self.MODEL_PATH, "rb") as f:
                self.model = pickle.load(f)
            log.info("rating_model_loaded_from_disk")
        else:
            self._train_synthetic()

    def _train_synthetic(self):
        """
        Bootstrap the regressor with synthetic training data.
        The regression is simple enough that even synthetic data gives reasonable priors.
        In v1.1: retrain on ground-truth user-item ratings from Yelp.
        """
        rng = np.random.default_rng(42)
        n = 5000

        user_means = rng.uniform(1.5, 5.0, n)
        user_stds = rng.uniform(0.2, 1.5, n)
        neg_rates = rng.uniform(0.0, 0.5, n)
        pos_rates = 1.0 - neg_rates - rng.uniform(0.0, 0.3, n)
        counts = rng.integers(1, 200, n).astype(float)
        cat_means = user_means + rng.normal(0, 0.5, n)
        cat_known = rng.choice([0.0, 1.0], n)
        item_globals = rng.uniform(2.5, 4.8, n)

        X = np.column_stack([
            user_means, user_stds, neg_rates, pos_rates,
            np.log1p(counts), cat_means, cat_known, item_globals
        ]).astype("float32")

        # Target: blend of user mean + item global + noise
        y = (
            0.55 * user_means +
            0.30 * item_globals +
            0.15 * cat_means +
            rng.normal(0, 0.3, n)
        ).clip(1.0, 5.0)

        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0)),
        ])
        self.model.fit(X, y)
        os.makedirs("data", exist_ok=True)
        with open(self.MODEL_PATH, "wb") as f:
            pickle.dump(self.model, f)
        log.info("rating_model_trained_synthetic", n_samples=n)

    def predict(self, profile: dict, item_metadata: dict) -> float:
        """Return predicted star rating clipped to [1.0, 5.0]."""
        X = _extract_features(profile, item_metadata)
        raw = float(self.model.predict(X)[0])
        # Round to nearest 0.5 star (more natural)
        rounded = round(raw * 2) / 2
        return float(np.clip(rounded, 1.0, 5.0))

    def retrain(self, X: np.ndarray, y: np.ndarray):
        """Public hook for retraining on ground-truth data (scripts/train_rating_model.py)."""
        self.model.fit(X, y)
        with open(self.MODEL_PATH, "wb") as f:
            pickle.dump(self.model, f)
        log.info("rating_model_retrained", n_samples=len(y))
