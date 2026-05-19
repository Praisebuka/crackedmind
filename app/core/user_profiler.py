"""
User Profile Encoder

Builds a structured behavioral fingerprint from review history.
This is the foundation of both Task A (voice simulation) and Task B (preference modeling).

Profile dimensions:
  - Rating statistics (mean, std, skew)
  - Vocabulary fingerprint (richness, avg review length)
  - Category preferences (top 3 liked/disliked)
  - Tone signals (sentiment curve, negativity willingness)
  - Nigerian linguistic markers (Pidgin detection)
  - Temporal signals (review cadence)
"""

import re
import statistics
from typing import List, Optional
from collections import Counter, defaultdict

from app.models.schemas import UserPersona, ReviewEntry

# Nigerian Pidgin / Nigerian English marker lexicon
# Sourced from NaijaSenti + manual linguistic annotation
_NAIJA_MARKERS = {
    "pidgin_phrases": [
        "e dey", "na wa", "omo", "sha", "abeg", "wetin", "na so", "kai",
        "e don", "wahala", "no be", "dem say", "wey", "make I", "I no fit",
        "how far", "e sweet", "correct correct", "sharp sharp", "better better",
        "e don do", "nawa o", "I no go lie", "e reach", "as e dey", "ehen",
        "na lie", "see me see trouble", "oya", "jare", "jejely", "ginger",
        "sabi", "shine your eye", "packaging", "hammer", "cruise", "pepper dem",
        "no dulling", "boss", "baba", "mama put", "suya", "owambe", "agbero",
        "danfo", "keke", "mainland", "island", "naira", "eko",
    ],
    "code_switch_patterns": [
        r"\b(e\s+dey|na\s+wa|no\s+be)\b",
        r"\b(abeg|wetin|oya|sha)\b",
        r"\b(omo|kai|nawa|ehen)\b",
    ],
}

_COMPILED_NAIJA = [
    re.compile(p, re.IGNORECASE) for p in _NAIJA_MARKERS["code_switch_patterns"]
]


def _detect_naija_markers(text: str) -> dict:
    """Count Nigerian linguistic signals in a review."""
    text_lower = text.lower()
    phrase_hits = sum(1 for phrase in _NAIJA_MARKERS["pidgin_phrases"] if phrase in text_lower)
    pattern_hits = sum(1 for p in _COMPILED_NAIJA if p.search(text))
    total = phrase_hits + pattern_hits
    return {
        "naija_signal_count": total,
        "is_nigerian_speaker": total >= 2,
        "confidence": min(1.0, total / 5.0),
    }


def _vocabulary_richness(texts: List[str]) -> dict:
    """Type-token ratio and average review length."""
    all_words = []
    for t in texts:
        words = re.findall(r"\b\w+\b", t.lower())
        all_words.extend(words)

    if not all_words:
        return {"ttr": 0.0, "avg_length": 0, "total_words": 0}

    ttr = len(set(all_words)) / len(all_words)
    avg_len = int(statistics.mean(len(re.findall(r"\b\w+\b", t)) for t in texts)) if texts else 0
    return {"ttr": round(ttr, 3), "avg_length": avg_len, "total_words": len(all_words)}


def _rating_stats(ratings: List[float]) -> dict:
    if not ratings:
        return {"mean": 3.0, "std": 0.0, "min": 1.0, "max": 5.0, "negativity_rate": 0.0}

    mean = statistics.mean(ratings)
    std = statistics.pstdev(ratings) if len(ratings) > 1 else 0.0
    negativity_rate = sum(1 for r in ratings if r <= 2) / len(ratings)

    return {
        "mean": round(mean, 2),
        "std": round(std, 2),
        "min": min(ratings),
        "max": max(ratings),
        "negativity_rate": round(negativity_rate, 2),
        "positivity_rate": round(sum(1 for r in ratings if r >= 4) / len(ratings), 2),
    }


def _category_preferences(reviews: List[ReviewEntry]) -> dict:
    """Category-level rating breakdown."""
    cat_ratings: dict = defaultdict(list)
    for r in reviews:
        cat_ratings[r.category].append(r.rating)

    preferences = {}
    for cat, ratings in cat_ratings.items():
        preferences[cat] = {
            "mean_rating": round(statistics.mean(ratings), 2),
            "count": len(ratings),
        }

    # Top liked and disliked categories
    sorted_cats = sorted(preferences.items(), key=lambda x: x[1]["mean_rating"], reverse=True)
    return {
        "by_category": preferences,
        "top_liked": [c for c, _ in sorted_cats[:3]],
        "top_disliked": [c for c, _ in sorted_cats[-2:]] if len(sorted_cats) >= 2 else [],
    }


def _tone_signals(texts: List[str]) -> dict:
    """
    Lightweight tone analysis without an external sentiment model.
    Uses lexicon-based positive/negative word counts.
    """
    pos_words = {
        "amazing", "excellent", "great", "fantastic", "love", "wonderful",
        "perfect", "outstanding", "recommend", "delicious", "best", "superb",
        "brilliant", "exceptional", "satisfied", "happy", "pleasant",
    }
    neg_words = {
        "terrible", "awful", "horrible", "worst", "hate", "disgusting",
        "disappointing", "poor", "bad", "rude", "waste", "never", "avoid",
        "overpriced", "undercooked", "dirty", "slow", "cold",
    }

    pos_count, neg_count = 0, 0
    for text in texts:
        words = set(re.findall(r"\b\w+\b", text.lower()))
        pos_count += len(words & pos_words)
        neg_count += len(words & neg_words)

    total = pos_count + neg_count or 1
    return {
        "pos_ratio": round(pos_count / total, 2),
        "neg_ratio": round(neg_count / total, 2),
        "tone_label": "positive" if pos_count > neg_count * 1.5 else
                      "negative" if neg_count > pos_count else "balanced",
    }


def build_user_profile(persona: UserPersona) -> dict:
    """
    Full behavioral fingerprint for a user.
    Returns a serializable dict that feeds into RAG prompts and the rating regressor.
    """
    history = persona.review_history
    texts = [r.review_text for r in history]
    ratings = [r.rating for r in history]

    naija_signals = [_detect_naija_markers(t) for t in texts]
    naija_total = sum(s["naija_signal_count"] for s in naija_signals)
    is_nigerian = naija_total >= 3 or any(s["is_nigerian_speaker"] for s in naija_signals)

    profile = {
        "user_id": persona.user_id,
        "review_count": len(history),
        "rating_stats": _rating_stats(ratings),
        "vocabulary": _vocabulary_richness(texts),
        "category_preferences": _category_preferences(history),
        "tone_signals": _tone_signals(texts),
        "nigerian_linguistic": {
            "is_nigerian_speaker": is_nigerian,
            "total_naija_signals": naija_total,
            "preferred_language": persona.preferred_language,
        },
        "location": persona.location,
        "cold_start": len(history) == 0,
    }

    return profile


def profile_to_prompt_summary(profile: dict) -> str:
    """
    Compact, prompt-friendly summary of the user profile.
    Used as context in LLM prompts for both tasks.
    """
    rs = profile["rating_stats"]
    tone = profile["tone_signals"]["tone_label"]
    cats = profile["category_preferences"].get("top_liked", [])
    vocab = profile["vocabulary"]
    nigerian = profile["nigerian_linguistic"]

    lines = [
        f"User ID: {profile['user_id']}",
        f"Reviews written: {profile['review_count']}",
        f"Average rating given: {rs['mean']} ± {rs['std']} stars",
        f"Negativity rate: {rs['negativity_rate']*100:.0f}% (gives ≤2 stars this often)",
        f"Tone tendency: {tone}",
        f"Avg review length: {vocab['avg_length']} words",
        f"Top categories: {', '.join(cats) if cats else 'unknown'}",
    ]

    if nigerian["is_nigerian_speaker"]:
        lines.append(
            f"Language: Nigerian speaker — uses Pidgin English naturally "
            f"(preference: {nigerian['preferred_language']})"
        )

    if profile.get("location"):
        lines.append(f"Location: {profile['location']}")

    return "\n".join(lines)
