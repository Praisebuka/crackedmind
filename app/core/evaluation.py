"""
Evaluation utilities for Task A and Task B.

Task A: BERTScore evaluation for review quality.
Task B: NDCG@10 evaluator for ranking quality.
"""

import structlog
from typing import List, Dict, Any
import numpy as np

log = structlog.get_logger()


class BERTScoreEvaluator:
    """Evaluate Task A generated reviews using BERTScore."""

    def __init__(self, model: str = "bert-base-uncased"):
        """Initialize BERTScore evaluator."""
        self.model = model
        self._scorer = None

    def _get_scorer(self):
        """Lazy-load BERTScore scorer."""
        if self._scorer is None:
            try:
                from bert_score import score as bert_score
                self._scorer = bert_score
            except ImportError:
                log.error("bert_score_import_failed", detail="Install bert-score")
                return None
        return self._scorer

    def evaluate_review(
        self,
        generated_review: str,
        reference_reviews: List[str],
    ) -> Dict[str, Any]:
        """
        Evaluate generated review against user's reference reviews using BERTScore.

        Args:
            generated_review: The generated review text
            reference_reviews: User's historical reviews for comparison

        Returns:
            Dict with precision, recall, f1 scores
        """
        if not reference_reviews:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "status": "no_references"}

        scorer = self._get_scorer()
        if scorer is None:
            log.warning("bert_score_unavailable", detail="Returning zero scores")
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "status": "scorer_unavailable"}

        try:
            # Compare generated review against each reference
            precisions = []
            recalls = []
            f1s = []

            for ref in reference_reviews:
                P, R, F1 = scorer(
                    [generated_review],
                    [ref],
                    lang="en",
                    rescale_with_baseline=True,
                )
                precisions.append(P.item())
                recalls.append(R.item())
                f1s.append(F1.item())

            avg_precision = np.mean(precisions)
            avg_recall = np.mean(recalls)
            avg_f1 = np.mean(f1s)

            log.info(
                "bertscore_evaluated",
                precision=avg_precision,
                recall=avg_recall,
                f1=avg_f1,
            )

            return {
                "precision": float(avg_precision),
                "recall": float(avg_recall),
                "f1": float(avg_f1),
                "status": "success",
            }
        except Exception as e:
            log.error("bertscore_evaluation_failed", error=str(e))
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "status": "error"}


class NDCGEvaluator:
    """Evaluate Task B recommendations using NDCG@10."""

    @staticmethod
    def compute_ndcg(
        scored_items: List[Dict[str, Any]],
        ideal_items: List[Dict[str, Any]],
        k: int = 10,
    ) -> float:
        """
        Compute Normalized Discounted Cumulative Gain @k.

        Args:
            scored_items: Ranked recommendations with 'score' and 'item_id'
            ideal_items: Ideal ranking (user's preferred items)
            k: Cutoff for top-k

        Returns:
            NDCG score (0.0 to 1.0)
        """
        if not scored_items or not ideal_items:
            return 0.0

        # Build ideal ranking map
        ideal_set = {item.get("item_id"): i for i, item in enumerate(ideal_items[:k])}
        if not ideal_set:
            return 0.0

        # Compute DCG@k
        dcg = 0.0
        for i, item in enumerate(scored_items[:k]):
            item_id = item.get("item_id")
            relevance = 1.0 if item_id in ideal_set else 0.0
            if relevance > 0:
                dcg += relevance / np.log2(i + 2)  # position i+1, log2 discount

        # Compute IDCG@k (ideal DCG)
        idcg = 0.0
        for i in range(min(k, len(ideal_items))):
            idcg += 1.0 / np.log2(i + 2)

        ndcg = dcg / idcg if idcg > 0 else 0.0
        return float(ndcg)

    @staticmethod
    def compute_hit_rate(
        scored_items: List[Dict[str, Any]],
        ideal_items: List[Dict[str, Any]],
        k: int = 10,
    ) -> float:
        """
        Compute Hit Rate @k (fraction of ideal items present in top-k).

        Args:
            scored_items: Ranked recommendations
            ideal_items: Ideal items
            k: Cutoff

        Returns:
            Hit rate (0.0 to 1.0)
        """
        if not ideal_items:
            return 0.0

        ideal_ids = {item.get("item_id") for item in ideal_items}
        top_k_ids = {item.get("item_id") for item in scored_items[:k]}
        hits = len(ideal_ids & top_k_ids)
        return float(hits / len(ideal_ids))

    @staticmethod
    def evaluate_ranking(
        scored_items: List[Dict[str, Any]],
        ideal_items: List[Dict[str, Any]],
        k: int = 10,
    ) -> Dict[str, float]:
        """
        Comprehensive ranking evaluation.

        Returns:
            Dict with ndcg, hit_rate, and metrics
        """
        try:
            ndcg = NDCGEvaluator.compute_ndcg(scored_items, ideal_items, k)
            hit_rate = NDCGEvaluator.compute_hit_rate(scored_items, ideal_items, k)

            log.info(
                "ndcg_evaluated",
                ndcg=ndcg,
                hit_rate=hit_rate,
                k=k,
            )

            return {
                "ndcg@k": ndcg,
                "hit_rate@k": hit_rate,
                "k": k,
                "status": "success",
            }
        except Exception as e:
            log.error("ndcg_evaluation_failed", error=str(e))
            return {
                "ndcg@k": 0.0,
                "hit_rate@k": 0.0,
                "k": k,
                "status": "error",
            }
