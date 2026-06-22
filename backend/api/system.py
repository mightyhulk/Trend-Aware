from fastapi import APIRouter
from datetime import datetime

from state import state

router = APIRouter(tags=["System"])

@router.get("/health")
async def health_check():
    """System health check endpoint."""
    return {
        "status": "healthy",
        "uptime_seconds": (
            datetime.utcnow() - state.startup_time
        ).total_seconds() if state.startup_time else 0,
        "users_loaded": len(state.users),
        "products_loaded": len(state.products),
        "interactions_loaded": len(state.interactions),
        "model_trained": state.engine.is_trained if state.engine else False,
    }

@router.get("/system/metrics")
async def get_system_metrics():
    """Get system-wide metrics and health status."""
    reference_time = datetime(2026, 6, 20, 12, 0, 0)
    
    report = state.evolution_manager.get_system_health_report(
        state.users,
        state.products,
        state.interactions,
        state.trend_signals,
        reference_time=reference_time,
    )
    
    # Add training metrics
    report["training_metrics"] = state.training_metrics
    
    # Add latency stats
    if state.request_latencies:
        report["performance"] = {
            "avg_latency_ms": round(
                sum(state.request_latencies) / len(state.request_latencies), 2
            ),
            "max_latency_ms": round(max(state.request_latencies), 2),
            "total_requests": len(state.request_latencies),
        }
    
    return report

@router.get("/system/evolution")
async def get_evolution_status():
    """
    Get system evolution status including:
    - Retrain schedule
    - Trend decay projections
    - Over-amplification warnings
    - Diversity health
    """
    reference_time = datetime(2026, 6, 20, 12, 0, 0)
    
    # Check retrain need
    should_retrain, retrain_reason = state.evolution_manager.should_retrain(
        current_time=reference_time,
        new_interactions_count=len(state.interactions),
    )
    
    # Detect over-amplification
    over_amplified = state.evolution_manager.detect_over_amplification(
        state.trend_signals,
        state.interactions,
        reference_time=reference_time,
    )
    
    # Get top trend's decay projection
    top_trend = state.trend_signals[0] if state.trend_signals else None
    decay_projection = None
    if top_trend:
        decay_projection = state.evolution_manager.compute_trend_decay_schedule(
            top_trend, state.trending_service
        ) if hasattr(state.evolution_manager, 'compute_trend_decay_schedule') else None
    
    return {
        "retrain": {
            "should_retrain": should_retrain,
            "reason": retrain_reason,
            "last_retrain": state.evolution_manager.last_retrain_time.isoformat()
                if state.evolution_manager.last_retrain_time else None,
        },
        "over_amplification_warnings": over_amplified,
        "trend_lifecycle": {
            "max_trend_lifetime_hours": state.evolution_manager.max_trend_lifetime_hours,
            "decay_rate": state.evolution_manager.trend_decay_rate,
            "active_trends_above_50": sum(
                1 for ts in state.trend_signals if ts.trend_score > 50
            ),
        },
    }
