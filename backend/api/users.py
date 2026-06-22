from fastapi import APIRouter, HTTPException, Query
from collections import defaultdict
from datetime import datetime

from state import state

router = APIRouter(tags=["Users"])

@router.get("/users")
async def list_users(
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    """List users."""
    total = len(state.users)
    users = state.users[offset:offset + limit]
    return {
        "users": [u.model_dump() for u in users],
        "total": total,
        "limit": limit,
        "offset": offset,
    }

@router.get("/users/{user_id}")
async def get_user(user_id: str):
    """Get a specific user by ID."""
    user = next(
        (u for u in state.users if u.user_id == user_id), None
    )
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Get user's interaction history
    interactions = [
        i for i in state.interactions if i.user_id == user_id
    ]
    
    # Detect cold start
    user_interactions = defaultdict(list)
    for inter in state.interactions:
        user_interactions[inter.user_id].append(inter)
    
    cold_info = state.cold_start_handler.detect_cold_start_user(
        user_id, state.users, user_interactions
    )
    
    return {
        "user": user.model_dump(),
        "interaction_count": len(interactions),
        "cold_start_info": cold_info.model_dump(),
        "recent_interactions": [
            i.model_dump() for i in sorted(
                interactions, key=lambda x: x.timestamp, reverse=True
            )[:10]
        ],
    }

@router.get("/cold-start/status")
async def get_cold_start_status():
    """Get cold-start analysis for the current user/product population."""
    user_interactions = defaultdict(list)
    product_interactions = defaultdict(list)
    
    for inter in state.interactions:
        user_interactions[inter.user_id].append(inter)
        product_interactions[inter.product_id].append(inter)
    
    cold_users = []
    for user in state.users:
        info = state.cold_start_handler.detect_cold_start_user(
            user.user_id, state.users, user_interactions
        )
        if info.is_cold_start:
            cold_users.append({
                "user_id": user.user_id,
                "strategy": info.strategy,
                "reason": info.reason,
            })
    
    reference_time = datetime(2026, 6, 20, 12, 0, 0)
    cold_products = []
    for product in state.products:
        info = state.cold_start_handler.detect_cold_start_product(
            product, product_interactions, reference_time
        )
        if info.is_cold_start:
            cold_products.append({
                "product_id": product.product_id,
                "name": product.name,
                "strategy": info.strategy,
                "reason": info.reason,
            })
    
    # New products for exploration
    new_products = state.cold_start_handler.get_new_products_for_exploration(
        state.products, product_interactions, reference_time
    )
    
    return {
        "cold_start_users": {
            "count": len(cold_users),
            "users": cold_users[:20],
        },
        "cold_start_products": {
            "count": len(cold_products),
            "products": cold_products[:20],
        },
        "exploration_candidates": [
            {"product_id": p.product_id, "name": p.name, "category": p.category}
            for p in new_products
        ],
    }
