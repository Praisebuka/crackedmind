"""
scripts/build_index.py

Processes Yelp, Amazon Reviews, and Goodreads datasets into the CRACKEDMIND item catalog.
Run this ONCE before starting the API.

Usage:
  python scripts/build_index.py \
    --yelp data/raw/yelp_business.json \
    --amazon data/raw/amazon_meta.json \
    --goodreads data/raw/goodreads_books.json \
    --output data/items.json \
    --max-items 50000

Dataset download:
  Yelp:      https://www.yelp.com/dataset
  Amazon:    https://amazon-reviews-2023.github.io/
  Goodreads: https://mengtingwan.github.io/data/goodreads.html

If you have no data yet, run without arguments — a sample catalog is auto-generated.
"""

import json
import argparse
import os
import sys
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def parse_yelp(path: str, max_items: int) -> list:
    items = []
    with open(path, "r") as f:
        for line in tqdm(f, desc="Yelp", unit="biz"):
            try:
                biz = json.loads(line.strip())
                if not biz.get("name") or not biz.get("categories"):
                    continue
                items.append({
                    "item_id": f"yelp_{biz['business_id']}",
                    "item_name": biz["name"],
                    "category": "restaurant",
                    "description": biz.get("categories", ""),
                    "metadata": {
                        "city": biz.get("city", ""),
                        "state": biz.get("state", ""),
                        "stars": biz.get("stars", 3.5),
                        "review_count": biz.get("review_count", 0),
                        "global_mean_rating": biz.get("stars", 3.5),
                        "source": "yelp",
                    },
                })
            except Exception:
                continue
            if len(items) >= max_items:
                break
    return items


def parse_amazon(path: str, max_items: int) -> list:
    items = []
    with open(path, "r") as f:
        for line in tqdm(f, desc="Amazon", unit="prod"):
            try:
                prod = json.loads(line.strip())
                title = prod.get("title") or prod.get("name", "")
                if not title:
                    continue
                items.append({
                    "item_id": f"amz_{prod.get('asin', len(items))}",
                    "item_name": title,
                    "category": "product",
                    "description": (prod.get("description") or [""])[0]
                                    if isinstance(prod.get("description"), list)
                                    else prod.get("description", ""),
                    "metadata": {
                        "brand": prod.get("brand", ""),
                        "price": prod.get("price", ""),
                        "global_mean_rating": prod.get("average_rating", 3.8),
                        "source": "amazon",
                    },
                })
            except Exception:
                continue
            if len(items) >= max_items:
                break
    return items


def parse_goodreads(path: str, max_items: int) -> list:
    items = []
    with open(path, "r") as f:
        for line in tqdm(f, desc="Goodreads", unit="book"):
            try:
                book = json.loads(line.strip())
                if not book.get("title"):
                    continue
                authors = ", ".join(
                    a.get("name", "") for a in book.get("authors", [])
                    if isinstance(a, dict)
                ) if book.get("authors") else ""
                items.append({
                    "item_id": f"gr_{book.get('book_id', len(items))}",
                    "item_name": book["title"],
                    "category": "book",
                    "description": f"{book.get('description', '')} by {authors}",
                    "metadata": {
                        "author": authors,
                        "genre": ", ".join(book.get("genres", [])[:3]),
                        "global_mean_rating": float(book.get("average_rating", 3.8)),
                        "source": "goodreads",
                    },
                })
            except Exception:
                continue
            if len(items) >= max_items:
                break
    return items


def main():
    parser = argparse.ArgumentParser(description="Build CRACKEDMIND item catalog + FAISS index")
    parser.add_argument("--yelp", default=None)
    parser.add_argument("--amazon", default=None)
    parser.add_argument("--goodreads", default=None)
    parser.add_argument("--output", default="data/items.json")
    parser.add_argument("--index", default="data/faiss.index")
    parser.add_argument("--max-items", type=int, default=50000)
    args = parser.parse_args()

    all_items = []

    if args.yelp and os.path.exists(args.yelp):
        all_items.extend(parse_yelp(args.yelp, args.max_items // 3))
        print(f"Yelp: {len(all_items)} items")

    if args.amazon and os.path.exists(args.amazon):
        before = len(all_items)
        all_items.extend(parse_amazon(args.amazon, args.max_items // 3))
        print(f"Amazon: {len(all_items) - before} items")

    if args.goodreads and os.path.exists(args.goodreads):
        before = len(all_items)
        all_items.extend(parse_goodreads(args.goodreads, args.max_items // 3))
        print(f"Goodreads: {len(all_items) - before} items")

    if not all_items:
        print("No dataset files provided. Generating sample catalog...")
        from app.core.rag_retriever import HybridRetriever
        r = HybridRetriever()
        r._load_sample_items()
        all_items = r._item_docs

    # Write catalog
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_items, f, indent=2)
    print(f"Catalog saved: {len(all_items)} items → {args.output}")

    # Build FAISS index
    print("Building FAISS index (this may take a while for large catalogs)...")
    from app.core.rag_retriever import HybridRetriever
    r = HybridRetriever()
    r._build_item_index(all_items, index_path=args.index)
    print(f"FAISS index saved → {args.index}")
    print("Done. Start the API with: docker-compose up")


if __name__ == "__main__":
    main()
