"""
Lightweight data pipeline using Hugging Face datasets API.

Supports loading, filtering, and preprocessing reviews and items for evaluation.
"""

import structlog
from typing import List, Dict, Any, Optional, Literal
import json

log = structlog.get_logger()


class HFDataPipeline:
    """Lightweight pipeline for Hugging Face datasets."""

    def __init__(self):
        """Initialize pipeline."""
        self._datasets_lib = None

    def _get_datasets_lib(self):
        """Lazy-load datasets library."""
        if self._datasets_lib is None:
            try:
                import datasets
                self._datasets_lib = datasets
                log.info("datasets_library_loaded")
            except ImportError:
                log.error("datasets_import_failed", detail="Install datasets")
                return None
        return self._datasets_lib

    def load_reviews_dataset(
        self,
        dataset_name: str = "amazon_reviews_multi",
        split: str = "train",
        languages: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> Optional[Any]:
        """
        Load review dataset from Hugging Face Hub.

        Args:
            dataset_name: Dataset identifier on Hub (e.g., 'amazon_reviews_multi')
            split: Dataset split ('train', 'validation', 'test')
            languages: Language filters (e.g., ['en', 'yo'])
            limit: Max rows to load

        Returns:
            Hugging Face Dataset or None on error
        """
        datasets = self._get_datasets_lib()
        if datasets is None:
            return None

        try:
            # Load dataset
            ds = datasets.load_dataset(dataset_name, split=split)
            log.info("dataset_loaded", name=dataset_name, split=split, rows=len(ds))

            # Filter by language if specified
            if languages:
                ds = ds.filter(lambda x: x.get("language") in languages)
                log.info("dataset_filtered", languages=languages, rows=len(ds))

            # Limit rows
            if limit:
                ds = ds.select(range(min(limit, len(ds))))
                log.info("dataset_limited", limit=limit, rows=len(ds))

            return ds
        except Exception as e:
            log.error("dataset_loading_failed", error=str(e))
            return None

    def load_local_reviews(
        self,
        file_path: str,
        format: Literal["json", "jsonl", "csv"] = "json",
    ) -> List[Dict[str, Any]]:
        """
        Load local review data.

        Args:
            file_path: Path to review file
            format: File format

        Returns:
            List of review dicts
        """
        try:
            if format == "json":
                with open(file_path, "r") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        data = data.get("reviews", [data])
                    return data if isinstance(data, list) else [data]

            elif format == "jsonl":
                reviews = []
                with open(file_path, "r") as f:
                    for line in f:
                        reviews.append(json.loads(line.strip()))
                return reviews

            elif format == "csv":
                import pandas as pd
                df = pd.read_csv(file_path)
                return df.to_dict("records")

            else:
                log.error("unsupported_format", format=format)
                return []

        except Exception as e:
            log.error("local_file_loading_failed", error=str(e))
            return []

    def filter_reviews_by_rating(
        self,
        reviews: List[Dict[str, Any]],
        min_rating: Optional[float] = None,
        max_rating: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Filter reviews by rating range."""
        filtered = reviews
        if min_rating is not None:
            filtered = [r for r in filtered if r.get("rating", 0) >= min_rating]
        if max_rating is not None:
            filtered = [r for r in filtered if r.get("rating", 5) <= max_rating]
        return filtered

    def filter_reviews_by_category(
        self,
        reviews: List[Dict[str, Any]],
        categories: List[str],
    ) -> List[Dict[str, Any]]:
        """Filter reviews by category."""
        return [r for r in reviews if r.get("category") in categories]

    def augment_reviews_with_metadata(
        self,
        reviews: List[Dict[str, Any]],
        metadata_source: str,
    ) -> List[Dict[str, Any]]:
        """
        Augment reviews with metadata (e.g., item info, user demographics).

        Args:
            reviews: List of reviews
            metadata_source: Path to metadata file

        Returns:
            Augmented reviews
        """
        try:
            metadata = self.load_local_reviews(metadata_source)
            metadata_map = {m.get("item_id"): m for m in metadata}

            augmented = []
            for review in reviews:
                item_id = review.get("item_id")
                if item_id in metadata_map:
                    review.update(metadata_map[item_id])
                augmented.append(review)

            log.info("reviews_augmented", count=len(augmented))
            return augmented
        except Exception as e:
            log.error("augmentation_failed", error=str(e))
            return reviews

    @staticmethod
    def extract_reference_reviews(
        reviews: List[Dict[str, Any]],
        user_id: Optional[str] = None,
        limit: int = 5,
    ) -> List[str]:
        """
        Extract reference review texts for evaluation.

        Args:
            reviews: List of reviews
            user_id: Filter by user (optional)
            limit: Max reviews to extract

        Returns:
            List of review texts
        """
        filtered = reviews
        if user_id:
            filtered = [r for r in filtered if r.get("user_id") == user_id]

        texts = [r.get("review_text", r.get("text", "")) for r in filtered[:limit]]
        return [t for t in texts if t]  # Remove empty strings


def load_evaluation_dataset(
    dataset_type: Literal["reviews", "items"] = "reviews",
    limit: Optional[int] = None,
) -> Optional[Any]:
    """
    Convenient factory for loading evaluation datasets.

    Args:
        dataset_type: Type of dataset to load
        limit: Row limit

    Returns:
        Dataset or None
    """
    pipeline = HFDataPipeline()

    if dataset_type == "reviews":
        return pipeline.load_reviews_dataset(
            dataset_name="amazon_reviews_multi",
            split="train",
            limit=limit,
        )
    elif dataset_type == "items":
        return pipeline.load_reviews_dataset(
            dataset_name="amazon_reviews_multi",
            split="test",
            limit=limit,
        )
    else:
        log.warning("unknown_dataset_type", type=dataset_type)
        return None
