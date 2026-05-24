# CRACKEDMIND v1.0

**LLM-based user modeling and intelligent recommendation — culturally contextualized for Nigerian users.**

Built for the LLM User Modeling & Recommendation Competition.  
Two tasks. One architecture. One cultural identity.

---
## For My Wonderful Judges

**Competition Task 1 → POST /v1/task-a/simulate-review**

Input: user persona + product details. Output: simulated review + predicted star rating.

**Competition Task 2 → POST /v1/task-b/recommend**

Input: user persona + optional conversation history. Output: ranked recommendations with CoT reasoning.

---

## What it does

| Task | Input | Output |
|------|-------|--------|
| **Task A - User Modeling** | User review history + unseen item | Simulated review in user's authentic voice + predicted star rating |
| **Task B - Recommendation** | User persona + optional conversation | Ranked personalized recommendations with CoT reasoning |

---

## Architecture

```
Security Gateway (injection defense, RAG poisoning mitigation)
        │
   ┌────┴────┐
   │         │
Task A     Task B
  │           │
User          Intent
Profiler      Analyzer
  │           │
RAG           Hybrid Retriever
Retriever     (FAISS + BM25 + RRF)
  │           │
LLM           CoT Reranker
Generator     (LLM scoring)
  │           │
Rating        Naija
Predictor     Adapter
  └────┬────┘
       │
Nigerian Cultural Context Layer
       │
  FastAPI + Docker
```

**Key components:**
- **Security Gateway** — prompt injection detection, XML demarcation of RAG content, input sanitization
- **User Profiler** — builds behavioral fingerprint: rating stats, vocabulary richness, tone signals, Nigerian linguistic markers
- **Hybrid Retriever** — FAISS (dense, semantic) + BM25 (sparse, keyword) with Reciprocal Rank Fusion
- **Rating Predictor** — scikit-learn regression head (separate from LLM, better RMSE)
- **CoT Reranker** — LLM chain-of-thought scoring per candidate
- **Nigerian Cultural Context Layer** — Pidgin English style transfer, cultural anchor boosting, AfriSenti-informed

---

## Quick Start

### 1. Prerequisites
- Docker + Docker Compose
- An Anthropic API key

### 2. Setup

```bash
git clone <repo>
cd crackedmind

cp .env.example .env
# Edit .env — add your ANTHROPIC_API_KEY
```

### 3. Build item catalog (optional — sample data auto-loads)

```bash
# With your own datasets:
python scripts/build_index.py \
  --yelp data/raw/yelp_academic_dataset_business.json \
  --amazon data/raw/meta_Electronics.jsonl \
  --goodreads data/raw/goodreads_books.json \
  --max-items 100000

# Without datasets (sample Nigerian catalog auto-loads):
python scripts/build_index.py
```

### 4. Start

```bash
docker-compose up --build
```

API available at `http://localhost:8000`  
Interactive docs: `http://localhost:8000/docs`

---

## API Usage

### Task A — Simulate a Review

```bash
curl -X POST http://localhost:8000/v1/task-a/simulate-review \
  -H "Content-Type: application/json" \
  -d '{
    "user_persona": {
      "user_id": "user_001",
      "preferred_language": "mixed",
      "location": "Lagos",
      "review_history": [
        {
          "item_id": "r001",
          "item_name": "Buka Restaurant",
          "category": "restaurant",
          "rating": 5.0,
          "review_text": "Omo this place dey correct die. The jollof rice e sweet me well well. Sharp sharp service too. I go definitely come back."
        },
        {
          "item_id": "r002",
          "item_name": "Generic Eatery",
          "category": "restaurant",
          "rating": 2.0,
          "review_text": "Na wa o. E no worth the money at all. The food was cold and the service was terrible. I no dey go back."
        }
      ]
    },
    "product_details": {
      "item_id": "r003",
      "item_name": "Yellow Chilli",
      "category": "restaurant",
      "description": "Contemporary Nigerian fine dining, amala, ewedu, banga soup"
    }
  }'
```

**Response:**
```json
{
  "user_id": "user_001",
  "item_id": "r003",
  "predicted_rating": 4.0,
  "generated_review": "E dey correct sha. Yellow Chilli delivered — the amala was smooth and the ewedu fresh...",
  "profile_summary": {
    "rating_mean": 3.5,
    "tone": "balanced",
    "is_nigerian_speaker": true
  },
  "naija_adapted": true,
  "retrieval_sources": 2
}
```

### Task B — Get Recommendations

```bash
curl -X POST http://localhost:8000/v1/task-b/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "user_persona": {
      "user_id": "user_001",
      "location": "Lagos",
      "preferred_language": "mixed",
      "review_history": []
    },
    "conversation_history": [
      {"role": "user", "content": "I want something spicy to eat on the island tonight"}
    ],
    "target_domain": "all",
    "top_k": 5
  }'
```

---

## Datasets

| Dataset | Source | Used for |
|---------|--------|----------|
| Yelp Open Dataset | yelp.com/dataset | Restaurant recommendations (Task B), review generation training |
| Amazon Reviews 2023 | amazon-reviews-2023.github.io | Product recommendations, cross-domain |
| Goodreads Books | mengtingwan.github.io | Book recommendations, cross-domain transfer |

---

## Security Architecture

crackedmind treats all external input as adversarial by default.

**Threat model (OWASP GenAI Top 10 2025):**
- `LLM01` — Prompt injection: pattern-based classifier on all inputs before LLM
- `LLM08` — RAG poisoning: all retrieved content is XML-demarcated (`<retrieved_content>` tags) to isolate it from instruction space
- `LLM06` — Excessive agency: agents have no write access to filesystem or external services

See `app/security/gateway.py` for implementation details.

---

## Evaluation Targets

| Metric | Component | Strategy |
|--------|-----------|----------|
| ROUGE / BERTScore | Task A | RAG over user's own reviews → vocabulary alignment |
| RMSE | Task A | Separate regression head, not LLM |
| Behavioural Fidelity | Task A | Nigerian voice, tone calibration |
| NDCG@10 / Hit Rate | Task B | Hybrid retrieval + CoT reranking |
| Cold-Start | Task B | Content-based bootstrap, conversational elicitation |
| Contextual Relevance | Task B | Nigerian cultural context layer |

---

## Nigerian Cultural Context Layer

Automatically activated when Nigerian linguistic markers are detected in user history.

**What it does:**
- Detects Pidgin English / Nigerian English code-switching using AfriSenti-informed lexicon
- Applies style transfer to generated reviews (Pidgin phrases by sentiment register)
- Boosts culturally-relevant items in recommendation ranking
- References Nigerian food taxonomy (suya, jollof, egusi), literature (Achebe, Adichie), and cultural anchors

**Toggle:** Set `NAIJA_LAYER_ENABLED=false` in `.env` to disable.

---

## Project Structure

```
crackedmind/
├── app/
│   ├── main.py                  # FastAPI app, lifespan, routes
│   ├── config.py                # Pydantic settings
│   ├── models/schemas.py        # Request/response types
│   ├── security/gateway.py      # Injection defense, sanitization
│   ├── core/
│   │   ├── user_profiler.py     # Behavioral fingerprint builder
│   │   ├── rag_retriever.py     # FAISS + BM25 + RRF
│   │   ├── rating_predictor.py  # Regression head
│   │   └── nigerian_layer.py    # Cultural contextualization
│   └── agents/
│       ├── task_a_agent.py      # User modeling pipeline
│       └── task_b_agent.py      # LangGraph recommendation pipeline
├── scripts/
│   └── build_index.py           # Dataset → FAISS index
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## License

MIT
