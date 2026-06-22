import time
from datetime import datetime
from collections import defaultdict
from fastapi import APIRouter, HTTPException, Query
from typing import Optional

from state import state
from models.schemas import RecommendationRequest, RecommendationResponse

router = APIRouter(tags=["Recommendations"])

@router.post("/recommendations", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """
    Get personalized recommendations for a user.
    
    Handles three scenarios:
    1. **Established users**: Hybrid collaborative + content-based + trending
    2. **Cold-start users**: Preference-guided or popularity-based fallback
    3. **Unknown users**: Global trending + popularity
    """
    start_time = time.time()
    
    # Build trend score lookup
    trend_lookup = {
        ts.product_id: ts.trend_score for ts in state.trend_signals
    }
    
    # Check for cold-start
    user_interactions = defaultdict(list)
    for inter in state.interactions:
        user_interactions[inter.user_id].append(inter)
    
    cold_info = state.cold_start_handler.detect_cold_start_user(
        request.user_id, state.users, user_interactions
    )
    
    if cold_info.is_cold_start:
        # Use cold-start handler
        recommendations, strategy = \
            state.cold_start_handler.recommend_for_cold_start_user(
                request.user_id,
                state.users,
                state.products,
                state.interactions,
                state.trend_signals,
                n=request.num_recommendations,
            )
        total_candidates = len(state.products)
    else:
        # Use hybrid recommendation engine
        recommendations, strategy, total_candidates = state.engine.recommend(
            user_id=request.user_id,
            n=request.num_recommendations,
            trend_scores=trend_lookup if request.include_trending else None,
            trend_weight=request.trending_weight,
            diversity_factor=request.diversity_factor,
        )
    
    latency_ms = (time.time() - start_time) * 1000
    state.request_latencies.append(latency_ms)
    
    return RecommendationResponse(
        user_id=request.user_id,
        recommendations=recommendations,
        strategy_used=strategy,
        total_candidates_evaluated=total_candidates,
    )

@router.get("/recommendations/{user_id}")
async def get_recommendations_simple(
    user_id: str,
    n: int = Query(default=10, ge=1, le=50),
    include_trending: bool = Query(default=True),
    trending_weight: float = Query(default=0.3, ge=0.0, le=1.0),
):
    """Simplified GET endpoint for recommendations."""
    request = RecommendationRequest(
        user_id=user_id,
        num_recommendations=n,
        include_trending=include_trending,
        trending_weight=trending_weight,
    )
    return await get_recommendations(request)

@router.get("/trending")
async def get_trending(
    category: Optional[str] = None,
    window_hours: int = Query(default=24, ge=1, le=168),
    top_n: int = Query(default=20, ge=1, le=100),
):
    """
    Get currently trending products.
    
    Optionally filter by category. Uses exponential time-weighted
    interaction velocity to identify genuine trends while dampening
    short-lived spikes.
    """
    # Recompute if different window requested
    if window_hours != 24:
        reference_time = datetime(2026, 6, 20, 12, 0, 0)
        signals = state.trending_service.compute_trend_scores(
            state.interactions,
            state.products,
            reference_time=reference_time,
            window_hours=window_hours,
        )
    else:
        signals = state.trend_signals
    
    if category:
        signals = state.trending_service.get_trending_by_category(
            signals, category, top_n
        )
    else:
        signals = signals[:top_n]
    
    # Enrich with product details
    product_map = {p.product_id: p for p in state.products}
    enriched = []
    for signal in signals:
        product = product_map.get(signal.product_id)
        if product:
            enriched.append({
                "product": product.model_dump(),
                "trend_signal": signal.model_dump(),
            })
    
    return {
        "trending": enriched,
        "window_hours": window_hours,
        "category": category,
        "count": len(enriched),
    }
