"""
Pydantic schemas for the recommendation system.
Defines all data models used across the API, services, and data layers.
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum
from datetime import datetime


# ─────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────

class InteractionType(str, Enum):
    VIEW = "view"
    CLICK = "click"
    ADD_TO_CART = "add_to_cart"
    PURCHASE = "purchase"
    RATING = "rating"


class Gender(str, Enum):
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"


# ─────────────────────────────────────────────
# Core Domain Models
# ─────────────────────────────────────────────

class Product(BaseModel):
    """Represents a product in the catalog."""
    product_id: str
    name: str
    category: str
    subcategory: str
    brand: str
    price: float
    description: str
    tags: List[str] = []
    created_at: datetime = Field(default_factory=datetime.utcnow)
    avg_rating: float = 0.0
    total_interactions: int = 0


class User(BaseModel):
    """Represents a user in the system."""
    user_id: str
    username: str
    age: int
    gender: Gender
    preferred_categories: List[str] = []
    signup_date: datetime = Field(default_factory=datetime.utcnow)
    interaction_count: int = 0


class Interaction(BaseModel):
    """Represents a user-product interaction event."""
    interaction_id: str
    user_id: str
    product_id: str
    interaction_type: InteractionType
    timestamp: datetime
    rating: Optional[float] = None  # Only for rating interactions
    session_id: Optional[str] = None


# ─────────────────────────────────────────────
# Recommendation & Trending Models
# ─────────────────────────────────────────────

class TrendSignal(BaseModel):
    """Trending signal for a product."""
    product_id: str
    trend_score: float
    velocity: float  # Rate of change in interactions
    window_hours: int  # Time window used for calculation
    interaction_count_window: int  # Interactions in the window
    category: str
    computed_at: datetime = Field(default_factory=datetime.utcnow)


class RecommendationExplanation(BaseModel):
    """Explains why a product was recommended."""
    collaborative_score: float = 0.0
    content_score: float = 0.0
    trend_score: float = 0.0
    cold_start_score: float = 0.0
    final_score: float = 0.0
    reasons: List[str] = []
    trend_contribution_pct: float = 0.0
    personalization_contribution_pct: float = 0.0


class RecommendedProduct(BaseModel):
    """A single recommended product with explanation."""
    product: Product
    score: float
    rank: int
    explanation: RecommendationExplanation


class RecommendationResponse(BaseModel):
    """Full recommendation response for a user."""
    user_id: str
    recommendations: List[RecommendedProduct]
    strategy_used: str  # "hybrid", "cold_start_user", "cold_start_trending"
    total_candidates_evaluated: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ─────────────────────────────────────────────
# API Request/Response Models
# ─────────────────────────────────────────────

class RecommendationRequest(BaseModel):
    """Request for recommendations."""
    user_id: str
    num_recommendations: int = Field(default=10, ge=1, le=50)
    include_trending: bool = True
    trending_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    diversity_factor: float = Field(default=0.2, ge=0.0, le=1.0)
    trending_window_hours: int = Field(default=24, ge=1, le=168)


class TrendingRequest(BaseModel):
    """Request for trending products."""
    category: Optional[str] = None
    window_hours: int = Field(default=24, ge=1, le=168)
    top_n: int = Field(default=20, ge=1, le=100)


class SystemMetrics(BaseModel):
    """System-wide metrics and health."""
    total_users: int
    total_products: int
    total_interactions: int
    model_last_trained: Optional[datetime] = None
    trending_last_computed: Optional[datetime] = None
    avg_recommendation_latency_ms: float = 0.0
    cold_start_users_count: int = 0
    cold_start_products_count: int = 0


class ColdStartInfo(BaseModel):
    """Information about cold-start detection and strategy."""
    is_cold_start: bool
    strategy: str
    reason: str
    fallback_used: bool = False
