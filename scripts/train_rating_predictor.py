"""
scripts/train_rating_predictor.py

Extract user-item-rating pairs from Yelp/Amazon review data,
build training features, and retrain the rating predictor model.

Usage:
  python scripts/train_rating_predictor.py \
    --yelp-reviews data/raw/yelp_review.json \
    --yelp-business data/raw/yelp_business.json \
    --output data/rating_model.pkl \
    --test-split 0.2

  python scripts/train_rating_predictor.py \
    --amazon-reviews data/raw/amazon_reviews.jsonl \
    --amazon-meta data/raw/amazon_meta.json \
    --output data/rating_model.pkl

Dataset download:
  Yelp reviews:  https://www.yelp.com/dataset (yelp_review.json)
  Amazon reviews: https://amazon-reviews-2023.github.io/
"""

import json
import argparse
import os
import sys
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import pickle

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.user_profiler import build_user_profile
from app.models.schemas import UserPersona, ReviewEntry


def extract_features(profile: dict, item_category: str, item_global_mean: float) -> np.ndarray:
    """
    Build feature vector from user profile + item metadata.
    Returns shape (8,) float32 array.
    
    Features:
      [0] user_mean_rating
      [1] user_std
      [2] user_negativity_rate
      [3] user_positivity_rate
      [4] log(user_review_count + 1)
      [5] category_match_mean
      [6] category_known (1 if user reviewed category, 0 else)
      [7] item_global_mean_rating
    """
    rs = profile.get("rating_stats", {})
    cat_prefs = profile.get("category_preferences", {}).get("by_category", {})

    user_mean = rs.get("mean", 3.5)
    user_std = rs.get("std", 1.0)
    user_neg_rate = rs.get("negativity_rate", 0.15)
    user_pos_rate = rs.get("positivity_rate", 0.5)
    user_count = profile.get("review_count", 0)

    cat_data = cat_prefs.get(item_category, {})
    cat_mean = cat_data.get("mean_rating", user_mean)
    cat_known = 1.0 if cat_data.get("count", 0) > 0 else 0.0

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

    return features


def parse_yelp_reviews(reviews_path: str, business_path: str, max_samples: int = 100000):
    """
    Parse Yelp review + business JSON files.
    Yields (user_id, item_id, rating, review_text, category, item_global_mean).
    """
    # Load business metadata first for category + ratings
    businesses = {}
    if os.path.exists(business_path):
        with open(business_path, "r") as f:
            for line in tqdm(f, desc="Loading Yelp businesses", unit="biz"):
                try:
                    biz = json.loads(line.strip())
                    businesses[biz["business_id"]] = {
                        "category": "restaurant",  # simplified
                        "global_mean": biz.get("stars", 3.5),
                    }
                except Exception:
                    continue

    # Parse reviews
    count = 0
    if os.path.exists(reviews_path):
        with open(reviews_path, "r") as f:
            for line in tqdm(f, desc="Parsing Yelp reviews", unit="review"):
                if count >= max_samples:
                    break
                try:
                    review = json.loads(line.strip())
                    biz_id = review.get("business_id")
                    if biz_id not in businesses:
                        continue
                    biz = businesses[biz_id]
                    yield (
                        review.get("user_id"),
                        biz_id,
                        review.get("stars", 3),
                        review.get("text", ""),
                        biz["category"],
                        biz["global_mean"],
                    )
                    count += 1
                except Exception:
                    continue


def parse_amazon_reviews(reviews_path: str, meta_path: str, max_samples: int = 100000):
    """
    Parse Amazon review JSONL file.
    Yields (user_id, item_id, rating, review_text, category, item_global_mean).
    """
    # Load product metadata
    products = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            for line in tqdm(f, desc="Loading Amazon products", unit="prod"):
                try:
                    prod = json.loads(line.strip())
                    asin = prod.get("asin")
                    if asin:
                        products[asin] = {
                            "category": "product",
                            "global_mean": prod.get("average_rating", 3.8),
                        }
                except Exception:
                    continue

    # Parse reviews
    count = 0
    if os.path.exists(reviews_path):
        with open(reviews_path, "r") as f:
            for line in tqdm(f, desc="Parsing Amazon reviews", unit="review"):
                if count >= max_samples:
                    break
                try:
                    review = json.loads(line.strip())
                    asin = review.get("asin")
                    if asin not in products:
                        continue
                    prod = products[asin]
                    rating = review.get("rating", review.get("overall", 3))
                    yield (
                        review.get("reviewerID"),
                        asin,
                        rating,
                        review.get("reviewText", ""),
                        prod["category"],
                        prod["global_mean"],
                    )
                    count += 1
                except Exception:
                    continue


def build_training_data(review_stream, min_reviews_per_user: int = 5):
    """
    Aggregate reviews by user, build profiles, and generate training (X, y).
    """
    # Group reviews by user
    user_reviews = defaultdict(list)
    for user_id, item_id, rating, text, category, global_mean in review_stream:
        user_reviews[user_id].append({
            "item_id": item_id,
            "rating": rating,
            "text": text,
            "category": category,
            "global_mean": global_mean,
        })

    X = []
    y = []
    skipped = 0

    for user_id, reviews in tqdm(user_reviews.items(), desc="Building training data", unit="user"):
        if len(reviews) < min_reviews_per_user:
            skipped += 1
            continue

        # Build persona from all but last review (test on last)
        persona_reviews = []
        for rev in reviews[:-1]:
            persona_reviews.append(ReviewEntry(
                item_id=rev["item_id"],
                item_name=rev["item_id"],  # simplified
                category=rev["category"],
                rating=float(rev["rating"]),
                review_text=rev["text"],
            ))

        # Build profile
        persona = UserPersona(user_id=user_id, review_history=persona_reviews)
        profile = build_user_profile(persona)

        # Use last review as test sample
        test_review = reviews[-1]
        features = extract_features(
            profile,
            test_review["category"],
            test_review["global_mean"]
        )
        X.append(features)
        y.append(float(test_review["rating"]))

    print(f"Generated {len(X)} training samples (skipped {skipped} users with <{min_reviews_per_user} reviews)")
    return np.vstack(X) if X else np.zeros((0, 8)), np.array(y)


def train_and_save(X, y, output_path: str, test_split: float = 0.2):
    """Train Ridge regressor and save."""
    if len(X) == 0:
        print("ERROR: No training data. Check your dataset files.")
        return

    # Train-test split
    split_idx = int(len(X) * (1 - test_split))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Build and train pipeline
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0)),
    ])
    model.fit(X_train, y_train)

    # Evaluate
    train_rmse = np.sqrt(mean_squared_error(y_train, model.predict(X_train)))
    test_rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))

    print(f"\n✓ Training complete")
    print(f"  Train RMSE: {train_rmse:.4f}")
    print(f"  Test RMSE:  {test_rmse:.4f}")

    # Save model
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(model, f)
    print(f"✓ Model saved → {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train rating predictor on real data")
    parser.add_argument("--yelp-reviews", default=None, help="Path to yelp_review.json")
    parser.add_argument("--yelp-business", default=None, help="Path to yelp_business.json")
    parser.add_argument("--amazon-reviews", default=None, help="Path to Amazon reviews JSONL")
    parser.add_argument("--amazon-meta", default=None, help="Path to Amazon product metadata")
    parser.add_argument("--output", default="data/rating_model.pkl")
    parser.add_argument("--test-split", type=float, default=0.2)
    parser.add_argument("--max-samples", type=int, default=100000)
    args = parser.parse_args()

    # Parse based on source
    if args.yelp_reviews and args.yelp_business:
        print("Processing Yelp data...")
        review_stream = parse_yelp_reviews(
            args.yelp_reviews,
            args.yelp_business,
            max_samples=args.max_samples
        )
    elif args.amazon_reviews and args.amazon_meta:
        print("Processing Amazon data...")
        review_stream = parse_amazon_reviews(
            args.amazon_reviews,
            args.amazon_meta,
            max_samples=args.max_samples
        )
    else:
        print("ERROR: Provide either --yelp-reviews + --yelp-business or --amazon-reviews + --amazon-meta")
        sys.exit(1)

    # Build training data
    X, y = build_training_data(review_stream, min_reviews_per_user=5)

    # Train
    train_and_save(X, y, args.output, test_split=args.test_split)


if __name__ == "__main__":
    main()
