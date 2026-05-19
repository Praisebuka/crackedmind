"""
Hybrid RAG Retriever — FAISS (dense) + BM25 (sparse) with Reciprocal Rank Fusion

Why hybrid?
  - Dense (FAISS): catches semantic matches ("something warm on a cold night" → stew)
  - Sparse (BM25): catches exact keyword hits ("jollof rice Lagos" → exact restaurant)
  - RRF fusion: principled combination that consistently outperforms either alone

Security note:
  - All retrieved content is XML-wrapped before entering LLM context (see gateway.py)
  - This isolates user-submitted or third-party text from the instruction space
"""

import json
import os
import pickle
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional, Tuple
import re
import structlog

from app.security.gateway import wrap_retrieved_content

log = structlog.get_logger()

# ─── RRF ─────────────────────────────────────────────────────────────────────

def _reciprocal_rank_fusion(
    dense_ranked: List[Tuple[int, float]],
    sparse_ranked: List[Tuple[int, float]],
    k: int = 60,
) -> List[Tuple[int, float]]:
    """
    RRF(d) = sum_r 1 / (k + rank_r(d))
    k=60 is the standard constant from the Cormack et al. 2009 paper.
    """
    scores: Dict[int, float] = {}

    for rank, (doc_id, _) in enumerate(dense_ranked, start=1):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)

    for rank, (doc_id, _) in enumerate(sparse_ranked, start=1):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ─── Retriever ────────────────────────────────────────────────────────────────

class HybridRetriever:
    """
    Dual-mode retriever.
    
    For Task A: retrieves user's own past reviews most similar to target item.
    For Task B: retrieves candidate items from the global item index.
    """

    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        log.info("loading_embedding_model", model=embedding_model_name)
        self.encoder = SentenceTransformer(embedding_model_name)
        self.dim = self.encoder.get_sentence_embedding_dimension()

        # Global item index (Task B)
        self._item_index: Optional[faiss.Index] = None
        self._item_docs: List[dict] = []
        self._item_bm25: Optional[BM25Okapi] = None
        self._item_corpus: List[List[str]] = []

    # ── Index management ──────────────────────────────────────────────────────

    def load_item_index(self, items_path: str, index_path: Optional[str] = None):
        """Load item catalog into FAISS + BM25."""
        if not os.path.exists(items_path):
            log.warning("items_not_found", path=items_path)
            self._load_sample_items()
            return

        with open(items_path, "r") as f:
            items = json.load(f)

        self._build_item_index(items, index_path)
        log.info("item_index_loaded", count=len(items))

    def _build_item_index(self, items: List[dict], index_path: Optional[str] = None):
        """Encode all items and build FAISS + BM25."""
        self._item_docs = items
        texts = [self._item_text(i) for i in items]

        # FAISS
        if index_path and os.path.exists(index_path):
            self._item_index = faiss.read_index(index_path)
            log.info("faiss_index_loaded_from_disk", path=index_path)
        else:
            embeddings = self.encoder.encode(texts, show_progress_bar=False, normalize_embeddings=True)
            embeddings = np.array(embeddings, dtype="float32")
            self._item_index = faiss.IndexFlatIP(self.dim)
            self._item_index.add(embeddings)
            if index_path:
                os.makedirs(os.path.dirname(index_path), exist_ok=True)
                faiss.write_index(self._item_index, index_path)
                log.info("faiss_index_saved", path=index_path)

        # BM25
        self._item_corpus = [re.findall(r"\w+", t.lower()) for t in texts]
        self._item_bm25 = BM25Okapi(self._item_corpus)

    def _item_text(self, item: dict) -> str:
        """Canonical text representation of an item for embedding."""
        parts = [
            item.get("item_name", ""),
            item.get("category", ""),
            item.get("description", ""),
        ]
        if item.get("metadata"):
            for k, v in item["metadata"].items():
                parts.append(f"{k}: {v}")
        return " ".join(filter(None, parts))

    def _load_sample_items(self):
        """Fallback sample catalog for cold start / testing."""
        sample_items = [
            {"item_id": "r001", "item_name": "Buka Restaurant", "category": "restaurant",
             "description": "Authentic Nigerian local cuisine, jollof rice, egusi soup, suya", "metadata": {"price_tier": "budget", "city": "Lagos"}},
            {"item_id": "r002", "item_name": "Yellow Chilli", "category": "restaurant",
             "description": "Contemporary Nigerian fine dining, amala, ewedu, banga soup", "metadata": {"price_tier": "mid", "city": "Lagos"}},
            {"item_id": "r003", "item_name": "Chicken Republic", "category": "fast_food",
             "description": "Nigerian fast food chain, fried chicken, jollof rice, chips", "metadata": {"price_tier": "budget", "city": "Abuja"}},
            {"item_id": "b001", "item_name": "Things Fall Apart", "category": "book",
             "description": "Chinua Achebe novel, Igbo culture Nigeria, colonialism, Okonkwo", "metadata": {"genre": "literary fiction", "year": "1958"}},
            {"item_id": "b002", "item_name": "Half of a Yellow Sun", "category": "book",
             "description": "Chimamanda Ngozi Adichie, Biafran war, Nigeria, love story", "metadata": {"genre": "historical fiction", "year": "2006"}},
            {"item_id": "b003", "item_name": "Purple Hibiscus", "category": "book",
             "description": "Chimamanda Ngozi Adichie, Nigerian family, religion, coming of age", "metadata": {"genre": "literary fiction", "year": "2003"}},
            {"item_id": "p001", "item_name": "Techno Camon 30", "category": "electronics",
             "description": "Smartphone popular in Nigeria, camera, battery life, affordable", "metadata": {"brand": "Tecno", "price_tier": "mid"}},
            {"item_id": "p002", "item_name": "Thermocool Refrigerator", "category": "appliances",
             "description": "Energy-efficient fridge, popular Nigerian brand, NEPA-friendly inverter", "metadata": {"brand": "Thermocool", "price_tier": "mid"}},
            {"item_id": "r004", "item_name": "The Place Restaurant", "category": "restaurant",
             "description": "Nigerian restaurant chain, suya, grills, rice dishes, Lagos", "metadata": {"price_tier": "mid", "city": "Lagos"}},
            {"item_id": "r005", "item_name": "Iya Basira", "category": "street_food",
             "description": "Famous street food spot, agege bread, beans, fried plantain, Lagos mainland", "metadata": {"price_tier": "budget", "city": "Lagos"}},
        ]
        self._build_item_index(sample_items, index_path=None)
        log.info("sample_item_catalog_loaded", count=len(sample_items))

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve_similar_reviews(
        self,
        query_text: str,
        review_history: List[dict],
        top_k: int = 5,
    ) -> List[dict]:
        """
        Task A: Retrieve user's own past reviews most similar to the target item.
        Runs FAISS over the user's own review corpus (built on-the-fly).
        Returns reviews wrapped in XML demarcation.
        """
        if not review_history:
            return []

        texts = [
            f"{r['item_name']} {r['category']} {r['review_text']}"
            for r in review_history
        ]
        embeddings = self.encoder.encode(texts, normalize_embeddings=True)
        embeddings = np.array(embeddings, dtype="float32")

        # Small index for user's reviews
        index = faiss.IndexFlatIP(self.dim)
        index.add(embeddings)

        query_emb = self.encoder.encode([query_text], normalize_embeddings=True)
        query_emb = np.array(query_emb, dtype="float32")

        k = min(top_k, len(review_history))
        distances, indices = index.search(query_emb, k)

        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], distances[0])):
            r = review_history[idx].copy()
            r["similarity_score"] = float(score)
            r["review_text_wrapped"] = wrap_retrieved_content(
                r["review_text"], source_id=f"user_review_{rank}"
            )
            results.append(r)

        return results

    def retrieve_items(
        self,
        query: str,
        top_k: int = 10,
        category_filter: Optional[str] = None,
    ) -> List[dict]:
        """
        Task B: Hybrid retrieval from global item catalog.
        Returns RRF-fused candidates sorted by relevance.
        """
        if self._item_index is None:
            self._load_sample_items()

        n_candidates = min(top_k * 3, len(self._item_docs))

        # Dense retrieval
        q_emb = self.encoder.encode([query], normalize_embeddings=True)
        q_emb = np.array(q_emb, dtype="float32")
        scores, indices = self._item_index.search(q_emb, n_candidates)
        dense_ranked = [(int(idx), float(s)) for idx, s in zip(indices[0], scores[0])]

        # Sparse retrieval (BM25)
        query_tokens = re.findall(r"\w+", query.lower())
        bm25_scores = self._item_bm25.get_scores(query_tokens)
        sparse_ranked = sorted(
            [(i, float(s)) for i, s in enumerate(bm25_scores)],
            key=lambda x: x[1], reverse=True
        )[:n_candidates]

        # RRF fusion
        fused = _reciprocal_rank_fusion(dense_ranked, sparse_ranked)

        results = []
        for idx, rrf_score in fused[:top_k]:
            item = self._item_docs[idx].copy()
            item["retrieval_score"] = round(rrf_score, 4)
            # Apply category filter post-retrieval
            if category_filter and category_filter != "all":
                if item.get("category", "").lower() != category_filter.lower():
                    continue
            item["text_wrapped"] = wrap_retrieved_content(
                self._item_text(item), source_id=item.get("item_id", str(idx))
            )
            results.append(item)

        return results[:top_k]

    @property
    def index_loaded(self) -> bool:
        return self._item_index is not None
