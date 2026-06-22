"""
Trending Signal Service.

Computes real-time trending scores for products based on time-weighted
interaction velocity. Uses exponential decay to ensure recent interactions
have more influence, while preventing over-amplification of short-lived trends.

Key Concepts:
- Interaction Velocity: Rate of interactions per unit time
- Exponential Decay: weight = e^(-λ × Δt) where Δt is time since interaction
- Trend Score: Aggregated weighted interactions normalized by baseline
- Trend Dampening: Prevents viral loops by applying logarithmic scaling
"""

import math
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

from models.schemas import (
    Interaction, Product, TrendSignal, InteractionType
)


# Interaction type weights — purchases are 5x more significant than views
INTERACTION_WEIGHTS: Dict[InteractionType, float] = {
    InteractionType.VIEW: 1.0,
    InteractionType.CLICK: 2.0,
    InteractionType.ADD_TO_CART: 3.5,
    InteractionType.PURCHASE: 5.0,
    InteractionType.RATING: 2.5,
}

# Default decay rate (λ) — higher = faster decay
DEFAULT_DECAY_RATE = 0.1  # per hour


class TrendingService:
    """
    Computes and manages trending signals for products.
    
    The trending score is calculated as:
    
        trend_score = log(1 + Σ(w_i × e^(-λ × Δt_i))) / baseline
    
    Where:
    - w_i = interaction type weight
    - λ = decay rate (configurable)
    - Δt_i = hours since interaction
    - baseline = average interaction rate over a longer window
    
    The log(1 + x) transformation prevents viral amplification while
    still capturing genuine surges in interest.
    """

    def __init__(
        self,
        decay_rate: float = DEFAULT_DECAY_RATE,
        baseline_window_days: int = 14,
        dampening_factor: float = 0.7,
    ):
        """
        Initialize the trending service.
        
        Args:
            decay_rate: Exponential decay rate (λ). Higher = faster decay.
            baseline_window_days: Days to use for computing baseline activity.
            dampening_factor: Dampening to prevent over-amplification (0-1).
                Lower values = more dampening.
        """
        self.decay_rate = decay_rate
        self.baseline_window_days = baseline_window_days
        self.dampening_factor = dampening_factor

    def compute_trend_scores(
        self,
        interactions: List[Interaction],
        products: List[Product],
        reference_time: Optional[datetime] = None,
        window_hours: int = 24,
    ) -> List[TrendSignal]:
        """
        Compute trending scores for all products.
        
        Args:
            interactions: All interaction events
            products: Product catalog
            reference_time: Time to compute trends relative to (default: now)
            window_hours: Window in hours for trend computation
            
        Returns:
            List of TrendSignal objects, sorted by trend_score descending
        """
        if reference_time is None:
            reference_time = datetime.utcnow()
        
        product_map = {p.product_id: p for p in products}
        
        # ── Step 1: Compute weighted interaction scores in the trend window ──
        trend_window_start = reference_time - timedelta(hours=window_hours)
        baseline_window_start = reference_time - timedelta(
            days=self.baseline_window_days
        )
        
        # Group interactions by product
        product_interactions: Dict[str, List[Interaction]] = defaultdict(list)
        product_baseline_interactions: Dict[str, List[Interaction]] = defaultdict(list)
        
        for interaction in interactions:
            if interaction.timestamp >= trend_window_start:
                product_interactions[interaction.product_id].append(interaction)
            if interaction.timestamp >= baseline_window_start:
                product_baseline_interactions[interaction.product_id].append(
                    interaction
                )
        
        # ── Step 2: Calculate trend scores ──
        trend_signals = []
        
        for product in products:
            pid = product.product_id
            recent_interactions = product_interactions.get(pid, [])
            baseline_interactions = product_baseline_interactions.get(pid, [])
            
            # Compute time-weighted score for the trend window
            weighted_score = 0.0
            for interaction in recent_interactions:
                dt_hours = max(
                    (reference_time - interaction.timestamp).total_seconds() / 3600,
                    0.01  # Prevent division by zero
                )
                weight = INTERACTION_WEIGHTS.get(
                    interaction.interaction_type, 1.0
                )
                decay = math.exp(-self.decay_rate * dt_hours)
                weighted_score += weight * decay
            
            # Compute baseline rate (interactions per hour over baseline window)
            baseline_hours = self.baseline_window_days * 24
            baseline_total = sum(
                INTERACTION_WEIGHTS.get(i.interaction_type, 1.0)
                for i in baseline_interactions
            )
            baseline_rate = baseline_total / baseline_hours if baseline_hours > 0 else 0.001
            
            # Trend score = ratio of current velocity to baseline
            current_velocity = weighted_score / window_hours if window_hours > 0 else 0
            
            if baseline_rate > 0:
                raw_trend = current_velocity / baseline_rate
            else:
                raw_trend = current_velocity * 10  # New product bonus
            
            # Apply dampening: log(1 + x) prevents exponential blowup
            trend_score = math.log(1 + raw_trend) * self.dampening_factor
            
            # Normalize to 0-100 scale (will be renormalized after all products)
            trend_signals.append(TrendSignal(
                product_id=pid,
                trend_score=round(trend_score, 4),
                velocity=round(current_velocity, 4),
                window_hours=window_hours,
                interaction_count_window=len(recent_interactions),
                category=product.category,
                computed_at=reference_time,
            ))
        
        # ── Step 3: Normalize scores to 0-100 ──
        max_score = max(
            (ts.trend_score for ts in trend_signals), default=1.0
        )
        if max_score > 0:
            for ts in trend_signals:
                ts.trend_score = round((ts.trend_score / max_score) * 100, 2)
        
        # Sort by trend score descending
        trend_signals.sort(key=lambda x: x.trend_score, reverse=True)
        
        return trend_signals

    def get_trending_by_category(
        self,
        trend_signals: List[TrendSignal],
        category: str,
        top_n: int = 10,
    ) -> List[TrendSignal]:
        """Get top trending products in a specific category."""
        category_trends = [
            ts for ts in trend_signals if ts.category == category
        ]
        return category_trends[:top_n]

    def detect_trend_anomalies(
        self,
        trend_signals: List[TrendSignal],
        threshold: float = 70.0,
    ) -> List[TrendSignal]:
        """
        Detect products with abnormally high trend scores.
        
        This is used by the System Evolution module to flag potential
        over-amplification that may need dampening.
        """
        return [ts for ts in trend_signals if ts.trend_score >= threshold]

    def compute_trend_decay_schedule(
        self,
        current_score: float,
        hours_ahead: int = 48,
    ) -> List[Tuple[int, float]]:
        """
        Project how a trend score will decay over time if no new interactions occur.
        
        Useful for understanding trend lifecycle and preventing over-amplification.
        
        Returns:
            List of (hours_from_now, projected_score) tuples
        """
        schedule = []
        for h in range(0, hours_ahead + 1, 1):
            decayed = current_score * math.exp(-self.decay_rate * h)
            schedule.append((h, round(decayed, 2)))
        return schedule
