from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from collections import Counter
from state import state
from models.schemas import InteractionType

router = APIRouter(tags=["Products"])

@router.get("/products")
async def list_products(
    category: Optional[str] = None,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    """List products with optional category filtering."""
    products = state.products
    if category:
        products = [p for p in products if p.category == category]
    
    total = len(products)
    products = products[offset:offset + limit]
    
    return {
        "products": [p.model_dump() for p in products],
        "total": total,
        "limit": limit,
        "offset": offset,
    }

@router.get("/products/{product_id}")
async def get_product(product_id: str):
    """Get a specific product by ID."""
    product = next(
        (p for p in state.products if p.product_id == product_id), None
    )
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    
    # Get trend signal for this product
    trend = next(
        (ts for ts in state.trend_signals if ts.product_id == product_id),
        None
    )
    
    # Get interaction stats
    interactions = [
        i for i in state.interactions if i.product_id == product_id
    ]
    
    return {
        "product": product.model_dump(),
        "trend_signal": trend.model_dump() if trend else None,
        "interaction_stats": {
            "total": len(interactions),
            "by_type": {
                t.value: sum(
                    1 for i in interactions if i.interaction_type == t
                )
                for t in InteractionType
            },
        },
    }

@router.get("/categories")
async def list_categories():
    """List all product categories with counts."""
    cat_counts = Counter(p.category for p in state.products)
    return {
        "categories": [
            {"name": cat, "product_count": count}
            for cat, count in sorted(cat_counts.items())
        ]
    }
