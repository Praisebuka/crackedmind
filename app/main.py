"""
CRACKEDMIND API — FastAPI entrypoint

Endpoints:
  GET  /health                     → system status
  POST /v1/task-a/simulate-review  → Task A: user modeling + review generation
  POST /v1/task-b/recommend        → Task B: personalized recommendation

Architecture notes:
  - All heavy objects (encoder, FAISS index, rating model) live in app state
    and are instantiated ONCE at startup (lifespan pattern)
  - Anthropic client is async — fully non-blocking LLM calls
  - Security gateway runs before any business logic
"""

import structlog
import anthropic
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.models.schemas import (
    TaskARequest, TaskAResponse,
    TaskBRequest, TaskBResponse,
    HealthResponse,
)
from app.core.rag_retriever import HybridRetriever
from app.core.rating_predictor import RatingPredictor
from app.agents.task_a_agent import run_task_a
from app.agents.task_b_agent import run_task_b

log = structlog.get_logger()
settings = get_settings()

# ─── Shared state ─────────────────────────────────────────────────────────────

class AppState:
    retriever: HybridRetriever = None
    rating_predictor: RatingPredictor = None
    anthropic_client: anthropic.AsyncAnthropic = None


_state = AppState()


# ─── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load all models once. Shutdown: cleanup."""
    log.info("crackedmind_startup", model=settings.model_name)

    # Embedding model + FAISS + BM25
    _state.retriever = HybridRetriever(embedding_model_name=settings.embedding_model)
    _state.retriever.load_item_index(
        items_path=settings.items_path,
        index_path=settings.index_path,
    )

    # Rating regression head
    _state.rating_predictor = RatingPredictor()

    # Anthropic async client
    _state.anthropic_client = anthropic.AsyncAnthropic(
        api_key=settings.anthropic_api_key
    )

    log.info("crackedmind_ready", index_loaded=_state.retriever.index_loaded)
    yield

    log.info("crackedmind_shutdown")


# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="CRACKEDMIND",
    description=(
        "LLM-based user modeling and intelligent recommendation system. "
        "Culturally contextualized for Nigerian users. "
        "Competition: Task A (user modeling) + Task B (recommendation)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ─── Dependencies ─────────────────────────────────────────────────────────────

def get_retriever() -> HybridRetriever:
    if _state.retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not initialized")
    return _state.retriever


def get_rating_predictor() -> RatingPredictor:
    if _state.rating_predictor is None:
        raise HTTPException(status_code=503, detail="Rating predictor not initialized")
    return _state.rating_predictor


def get_anthropic() -> anthropic.AsyncAnthropic:
    if _state.anthropic_client is None:
        raise HTTPException(status_code=503, detail="Anthropic client not initialized")
    return _state.anthropic_client


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health(retriever: HybridRetriever = Depends(get_retriever)):
    return HealthResponse(
        status="ok",
        model=settings.model_name,
        index_loaded=retriever.index_loaded,
        naija_layer=settings.naija_layer_enabled,
    )


@app.post(
    "/v1/task-a/simulate-review",
    response_model=TaskAResponse,
    tags=["Task A — User Modeling"],
    summary="Simulate a user's review for an unseen item",
)
async def simulate_review(
    request: TaskARequest,
    retriever: HybridRetriever = Depends(get_retriever),
    rating_predictor: RatingPredictor = Depends(get_rating_predictor),
    client: anthropic.AsyncAnthropic = Depends(get_anthropic),
):
    """
    Task A endpoint.

    Given a user's review history (persona) and a product,
    generates a simulated review in the user's authentic voice,
    with a predicted star rating.

    - **Predicted rating** is from a regression head (not the LLM) for better RMSE
    - **Generated review** uses RAG over the user's own past reviews for ROUGE alignment
    - **Nigerian voice** is applied automatically when Pidgin/Nigerian markers are detected
    """
    try:
        result = await run_task_a(
            request=request,
            retriever=retriever,
            rating_predictor=rating_predictor,
            anthropic_client=client,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log.error("task_a_error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Task A failed: {str(e)}")


@app.post(
    "/v1/task-b/recommend",
    response_model=TaskBResponse,
    tags=["Task B — Recommendation"],
    summary="Get personalized recommendations for a user",
)
async def recommend(
    request: TaskBRequest,
    retriever: HybridRetriever = Depends(get_retriever),
    rating_predictor: RatingPredictor = Depends(get_rating_predictor),
    client: anthropic.AsyncAnthropic = Depends(get_anthropic),
):
    """
    Task B endpoint.

    Delivers personalized recommendations using a LangGraph agentic pipeline:
    intent analysis → hybrid retrieval → chain-of-thought reranking → cultural adaptation.

    - **Cold-start** users (no history) get content-based bootstrapping
    - **Cross-domain** recommendations work across restaurants, books, and products
    - **Nigerian cultural context** is activated automatically for Nigerian speakers
    - **Multi-turn** conversations are supported via `conversation_history`
    """
    try:
        result = await run_task_b(
            request=request,
            retriever=retriever,
            rating_predictor=rating_predictor,
            anthropic_client=client,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log.error("task_b_error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Task B failed: {str(e)}")
