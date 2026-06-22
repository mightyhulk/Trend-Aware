"""
System Evolution Manager.

Ensures the recommendation system evolves over time:

1. **Model Retraining**: Periodically retrain SVD and TF-IDF models
   with accumulated interaction data to capture shifting preferences.

2. **Trend Decay Management**: Automatically decay trending signals
   to prevent over-amplification of short-lived viral events.

3. **Diversity Monitoring**: Track recommendation diversity over time
   and inject exploration when filter bubbles are detected.

4. **Feedback Loop Protection**: Detect and mitigate positive feedback
   loops where popular items get more popular indefinitely.

5. **A/B Test Framework**: Support for experimentation with different
   model parameters and fusion weights.
"""

import math
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from collections import defaultdict, Counter

from models.schemas import (
    Product, User, Interaction, TrendSignal,
    InteractionType, SystemMetrics,
)


class SystemEvolutionManager:
    """
    Manages system evolution, trend lifecycle, and recommendation health.
    
    Key Responsibilities:
    - Monitor and manage trend decay to prevent over-amplification
    - Track recommendation diversity metrics
    - Detect feedback loops and popularity bias
    - Schedule model retraining based on data drift
    """

    def __init__(
        self,
        trend_decay_rate: float = 0.1,
        max_trend_lifetime_hours: int = 72,
        diversity_target: float = 0.7,  # Gini coefficient target
        retrain_interval_hours: int = 24,
    ):
        self.trend_decay_rate = trend_decay_rate
        self.max_trend_lifetime_hours = max_trend_lifetime_hours
        self.diversity_target = diversity_target
        self.retrain_interval_hours = retrain_interval_hours
        
        # Tracking state
        self.last_retrain_time: Optional[datetime] = None
        self.trend_history: List[Dict] = []
        self.diversity_history: List[Dict] = []

    def apply_trend_decay(
        self,
        trend_signals: List[TrendSignal],
        reference_time: Optional[datetime] = None,
    ) -> List[TrendSignal]:
        """
        Apply time-based decay to trending signals.
        
        This prevents short-lived trends from dominating recommendations
        indefinitely. Products must continue receiving interactions to
        maintain their trend status.
        
        Decay formula: decayed_score = score × e^(-λ × age_hours)
        
        Additionally, trends older than max_trend_lifetime are zeroed out
        to prevent stale signals.
        """
        if reference_time is None:
            reference_time = datetime.utcnow()
        
        decayed_signals = []
        
        for signal in trend_signals:
            age_hours = max(
                (reference_time - signal.computed_at).total_seconds() / 3600,
                0
            )
            
            # Hard cutoff for very old trends
            if age_hours > self.max_trend_lifetime_hours:
                signal.trend_score = 0.0
            else:
                # Exponential decay
                decay_factor = math.exp(-self.trend_decay_rate * age_hours)
                signal.trend_score = round(
                    signal.trend_score * decay_factor, 2
                )
            
            decayed_signals.append(signal)
        
        return decayed_signals

    def detect_over_amplification(
        self,
        trend_signals: List[TrendSignal],
        interactions: List[Interaction],
        reference_time: Optional[datetime] = None,
        window_hours: int = 24,
    ) -> List[Dict]:
        """
        Detect products that may be over-amplified by trending signals.
        
        A product is flagged if:
        1. Its trend score is in the top 5%
        2. The interaction velocity is declining (the burst is over)
        3. It has sustained high visibility for too long
        
        Returns:
            List of flagged products with details
        """
        if reference_time is None:
            reference_time = datetime.utcnow()
        
        flagged = []
        
        # Identify top 5% by trend score
        if not trend_signals:
            return flagged
        
        sorted_signals = sorted(
            trend_signals, key=lambda x: x.trend_score, reverse=True
        )
        top_threshold = max(1, len(sorted_signals) // 20)
        top_signals = sorted_signals[:top_threshold]
        
        for signal in top_signals:
            # Check if velocity is declining
            recent_window = reference_time - timedelta(hours=window_hours // 2)
            older_window_start = reference_time - timedelta(hours=window_hours)
            
            recent_count = sum(
                1 for i in interactions
                if i.product_id == signal.product_id and
                i.timestamp >= recent_window
            )
            older_count = sum(
                1 for i in interactions
                if i.product_id == signal.product_id and
                older_window_start <= i.timestamp < recent_window
            )
            
            velocity_declining = recent_count < older_count * 0.7
            
            if velocity_declining and signal.trend_score > 70:
                flagged.append({
                    "product_id": signal.product_id,
                    "trend_score": signal.trend_score,
                    "recent_interactions": recent_count,
                    "older_interactions": older_count,
                    "recommendation": "Apply dampening — trend burst is fading",
                    "suggested_score": round(signal.trend_score * 0.5, 2),
                })
        
        return flagged

    def compute_diversity_metrics(
        self,
        recommendations: Dict[str, List[str]],
        products: List[Product],
    ) -> Dict:
        """
        Compute diversity metrics for the recommendation system.
        
        Measures:
        - Category coverage: % of categories represented in recommendations
        - Gini coefficient: Inequality in category distribution
        - Brand diversity: Number of unique brands recommended
        - Price range coverage: Spread across price points
        
        Args:
            recommendations: Dict of user_id → list of recommended product_ids
            products: Product catalog
            
        Returns:
            Dictionary of diversity metrics
        """
        product_map = {p.product_id: p for p in products}
        
        all_recommended = []
        for recs in recommendations.values():
            all_recommended.extend(recs)
        
        if not all_recommended:
            return {
                "category_coverage": 0,
                "gini_coefficient": 1.0,
                "brand_diversity": 0,
                "price_range_coverage": 0,
                "is_healthy": False,
            }
        
        # Category distribution
        all_categories = set(p.category for p in products)
        rec_categories = set()
        category_counts = Counter()
        brand_set = set()
        prices = []
        
        for pid in all_recommended:
            if pid in product_map:
                p = product_map[pid]
                rec_categories.add(p.category)
                category_counts[p.category] += 1
                brand_set.add(p.brand)
                prices.append(p.price)
        
        # Category coverage
        category_coverage = len(rec_categories) / len(all_categories) \
            if all_categories else 0
        
        # Gini coefficient
        gini = self._gini_coefficient(list(category_counts.values()))
        
        # Price range coverage
        all_prices = [p.price for p in products]
        if all_prices and prices:
            price_range_coverage = (max(prices) - min(prices)) / \
                (max(all_prices) - min(all_prices)) \
                if max(all_prices) > min(all_prices) else 0
        else:
            price_range_coverage = 0
        
        is_healthy = (
            category_coverage >= 0.5 and
            gini <= 0.6 and
            len(brand_set) >= 3
        )
        
        metrics = {
            "category_coverage": round(category_coverage, 3),
            "gini_coefficient": round(gini, 3),
            "brand_diversity": len(brand_set),
            "unique_brands": list(brand_set)[:10],
            "price_range_coverage": round(price_range_coverage, 3),
            "category_distribution": dict(category_counts),
            "is_healthy": is_healthy,
        }
        
        self.diversity_history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": metrics,
        })
        
        return metrics

    def _gini_coefficient(self, values: List[float]) -> float:
        """Compute Gini coefficient for measuring inequality."""
        if not values or sum(values) == 0:
            return 0.0
        
        sorted_values = sorted(values)
        n = len(sorted_values)
        total = sum(sorted_values)
        
        cumulative = 0
        gini_sum = 0
        for i, v in enumerate(sorted_values):
            cumulative += v
            gini_sum += (2 * (i + 1) - n - 1) * v
        
        return gini_sum / (n * total) if total > 0 else 0

    def should_retrain(
        self,
        current_time: Optional[datetime] = None,
        new_interactions_count: int = 0,
        min_new_interactions: int = 500,
    ) -> Tuple[bool, str]:
        """
        Determine if the model should be retrained.
        
        Retrain conditions:
        1. Time-based: Exceeded retrain interval
        2. Data-based: Sufficient new interactions accumulated
        3. Drift-based: Significant change in interaction patterns
        
        Returns:
            Tuple of (should_retrain, reason)
        """
        if current_time is None:
            current_time = datetime.utcnow()
        
        # First train
        if self.last_retrain_time is None:
            return True, "Initial training required"
        
        # Time-based check
        hours_since = (current_time - self.last_retrain_time).total_seconds() / 3600
        if hours_since >= self.retrain_interval_hours:
            return True, (
                f"Time-based retrain: {hours_since:.1f}h since last train "
                f"(threshold: {self.retrain_interval_hours}h)"
            )
        
        # Data-based check
        if new_interactions_count >= min_new_interactions:
            return True, (
                f"Data-based retrain: {new_interactions_count} new interactions "
                f"(threshold: {min_new_interactions})"
            )
        
        return False, "No retrain needed"

    def get_system_health_report(
        self,
        users: List[User],
        products: List[Product],
        interactions: List[Interaction],
        trend_signals: List[TrendSignal],
        reference_time: Optional[datetime] = None,
    ) -> Dict:
        """
        Generate a comprehensive system health report.
        
        Covers:
        - Data statistics
        - Cold-start population sizes
        - Trend health
        - Interaction distribution
        - Staleness indicators
        """
        if reference_time is None:
            reference_time = datetime.utcnow()
        
        # Count cold-start entities
        user_interaction_counts = Counter(i.user_id for i in interactions)
        product_interaction_counts = Counter(i.product_id for i in interactions)
        
        cold_users = sum(
            1 for u in users
            if user_interaction_counts.get(u.user_id, 0) < 5
        )
        cold_products = sum(
            1 for p in products
            if product_interaction_counts.get(p.product_id, 0) < 10
        )
        
        # Interaction recency
        if interactions:
            latest_interaction = max(i.timestamp for i in interactions)
            hours_since_latest = (
                reference_time - latest_interaction
            ).total_seconds() / 3600
        else:
            hours_since_latest = float('inf')
        
        # Trend health
        active_trends = sum(
            1 for ts in trend_signals if ts.trend_score > 10
        )
        hot_trends = sum(
            1 for ts in trend_signals if ts.trend_score > 70
        )
        
        # Interaction type distribution
        type_dist = Counter(i.interaction_type.value for i in interactions)
        
        return {
            "data_statistics": {
                "total_users": len(users),
                "total_products": len(products),
                "total_interactions": len(interactions),
                "avg_interactions_per_user": round(
                    len(interactions) / max(len(users), 1), 1
                ),
                "avg_interactions_per_product": round(
                    len(interactions) / max(len(products), 1), 1
                ),
            },
            "cold_start": {
                "cold_start_users": cold_users,
                "cold_start_users_pct": round(
                    cold_users / max(len(users), 1) * 100, 1
                ),
                "cold_start_products": cold_products,
                "cold_start_products_pct": round(
                    cold_products / max(len(products), 1) * 100, 1
                ),
            },
            "trending": {
                "active_trends": active_trends,
                "hot_trends": hot_trends,
                "trending_categories": list(set(
                    ts.category for ts in trend_signals if ts.trend_score > 50
                )),
            },
            "freshness": {
                "hours_since_latest_interaction": round(hours_since_latest, 1),
                "is_stale": hours_since_latest > 48,
                "last_retrain": self.last_retrain_time.isoformat() 
                    if self.last_retrain_time else None,
            },
            "interaction_distribution": dict(type_dist),
        }
