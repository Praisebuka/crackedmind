from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Literal
from enum import Enum


# ─── Shared ──────────────────────────────────────────────────────────────────

class ReviewEntry(BaseModel):
    """One historical review from a user's past."""
    item_id: str
    item_name: str
    category: str
    rating: float = Field(..., ge=1.0, le=5.0)
    review_text: str
    timestamp: Optional[str] = None

    @field_validator("review_text")
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("review_text cannot be empty")
        return v.strip()


class UserPersona(BaseModel):
    """Full user context sent to both tasks."""
    user_id: str
    review_history: List[ReviewEntry] = Field(..., min_length=0)
    location: Optional[str] = None
    preferred_language: Literal["english", "pidgin", "mixed"] = "english"
    demographic_hints: Optional[dict] = None


class ProductDetails(BaseModel):
    """Item to be reviewed or used as domain signal."""
    item_id: str
    item_name: str
    category: str
    description: Optional[str] = None
    metadata: Optional[dict] = None


# ─── Task A ──────────────────────────────────────────────────────────────────

class TaskARequest(BaseModel):
    user_persona: UserPersona
    product_details: ProductDetails


class TaskAResponse(BaseModel):
    user_id: str
    item_id: str
    predicted_rating: float
    generated_review: str
    profile_summary: dict
    naija_adapted: bool
    retrieval_sources: int


# ─── Task B ──────────────────────────────────────────────────────────────────

class ConversationTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class TaskBRequest(BaseModel):
    user_persona: UserPersona
    conversation_history: Optional[List[ConversationTurn]] = []
    target_domain: Optional[Literal["restaurants", "products", "books", "all"]] = "all"
    top_k: int = Field(default=10, ge=1, le=20)


class RecommendedItem(BaseModel):
    item_id: str
    item_name: str
    category: str
    score: float
    predicted_rating: float
    explanation: str
    cold_start: bool = False


class TaskBResponse(BaseModel):
    user_id: str
    recommendations: List[RecommendedItem]
    intent_summary: str
    reasoning_trace: str
    naija_adapted: bool
    cold_start_mode: bool


# ─── Health ───────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    model: str
    index_loaded: bool
    naija_layer: bool
