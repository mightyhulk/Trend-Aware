"""
FastAPI Application — AI-Driven Trend-Aware Recommendation System.

This is the main application entry point that:
1. Initializes and trains the recommendation models on startup
2. Exposes REST API endpoints for recommendations, trending, and system health
3. Serves the frontend dashboard
4. Provides comprehensive documentation via Swagger UI

Production Design Notes:
- CORS is configured for local development; restrict in production
- All endpoints include proper error handling and validation
- Response models ensure type-safe API contracts
"""

import logging
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from state import state
from data.generator import create_dataset
from services.recommendation_engine import RecommendationEngine
from services.trending_service import TrendingService
from services.cold_start_handler import ColdStartHandler
from services.evolution_manager import SystemEvolutionManager

from api import products, recommendations, users, system

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Application Lifecycle
# ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize models and data on startup."""
    logger.info("Starting AI Recommendation System...")
    
    # Generate synthetic data
    logger.info(" Generating synthetic dataset...")
    state.users, state.products, state.interactions = create_dataset(
        num_users=500,
        num_products=200,
        num_interactions=15000,
        seed=42,
    )
    logger.info(
        f" Generated {len(state.users)} users, "
        f"{len(state.products)} products, "
        f"{len(state.interactions)} interactions"
    )
    
    # Initialize services
    state.trending_service = TrendingService(
        decay_rate=0.1,
        baseline_window_days=14,
        dampening_factor=0.7,
    )
    state.cold_start_handler = ColdStartHandler(
        exploration_boost=0.3,
        popularity_weight=0.5,
    )
    state.evolution_manager = SystemEvolutionManager(
        trend_decay_rate=0.1,
        max_trend_lifetime_hours=72,
        diversity_target=0.7,
    )
    
    # Compute trending signals
    logger.info(" Computing trending signals...")
    reference_time = datetime(2026, 6, 20, 12, 0, 0)
    state.trend_signals = state.trending_service.compute_trend_scores(
        state.interactions,
        state.products,
        reference_time=reference_time,
        window_hours=24,
    )
    logger.info(
        f" Computed trends for {len(state.trend_signals)} products"
    )
    
    # Train recommendation engine
    logger.info(" Training recommendation engine (Hybrid SVD + TF-IDF)...")
    state.engine = RecommendationEngine(
        n_latent_factors=50,
        collaborative_weight=0.6,
        min_interactions_for_cf=3,
    )
    state.training_metrics = state.engine.train(
        state.users, state.products, state.interactions
    )
    state.evolution_manager.last_retrain_time = datetime.utcnow()
    logger.info(" Training complete!")
    logger.info(f" Metrics: {state.training_metrics}")
    
    state.startup_time = datetime.utcnow()
    logger.info(" System ready to serve recommendations!")
    
    yield
    
    logger.info(" Shutting down recommendation system")


# ─────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────

app = FastAPI(
    title="AI-Driven Trend-Aware Recommendation System",
    description=(
        "A hybrid recommendation system combining collaborative filtering (SVD), "
        "content-based filtering (TF-IDF), and real-time trending signals with "
        "cold-start handling, explainability, and system evolution management."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# API Routers
# ─────────────────────────────────────────────
app.include_router(system.router, prefix="/api")
app.include_router(recommendations.router, prefix="/api")
app.include_router(products.router, prefix="/api")
app.include_router(users.router, prefix="/api")

# ─────────────────────────────────────────────
# Static Files — Frontend Dashboard
# ─────────────────────────────────────────────
import os
from pathlib import Path
from fastapi.responses import FileResponse

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"

@app.get("/")
async def serve_frontend():
    """Serve the frontend dashboard."""
    return FileResponse(FRONTEND_DIR / "index.html")

if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
