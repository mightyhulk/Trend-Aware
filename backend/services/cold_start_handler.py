"""
Cold-Start Handler.

Addresses the cold-start problem for both new users and new products:

New Users (no interaction history):
1. Popularity-based fallback: Recommend globally popular items
2. Category trending: Surface items trending in diverse categories
3. Onboarding preferences: Use declared category preferences if available
4. Demographic similarity: Match with similar demographic cohorts

New Products (no interaction data):
1. Content similarity: Find similar products with existing interactions
2. Category popularity proxy: Inherit category-level engagement metrics
3. Exploration boost: Temporarily boost new product visibility
4. Quality signals: Use brand reputation and price positioning

The cold-start strategy transitions smoothly to the hybrid approach
as the user/product accumulates more interactions.
"""

import math
from typing import List, Dict, Optional, Set, Tuple
from collections import defaultdict
from datetime import datetime, timedelta

from models.schemas import (
    Product, User, Interaction, InteractionType,
    RecommendedProduct, RecommendationExplanation,
    ColdStartInfo, TrendSignal,
)


# Minimum interactions before a user is no longer "cold"
COLD_START_USER_THRESHOLD = 5
# Minimum total interactions on a product before it's no longer "cold"
COLD_START_PRODUCT_THRESHOLD = 10
# How many hours a product is considered "new" for exploration boost
NEW_PRODUCT_WINDOW_HOURS = 72


class ColdStartHandler:
    """
    Handles cold-start scenarios for users and products.
    """

    def __init__(
        self,
        exploration_boost: float = 0.3,
        popularity_weight: float = 0.5,
        diversity_bonus: float = 0.15,
    ):
        self.exploration_boost = exploration_boost
        self.popularity_weight = popularity_weight
        self.diversity_bonus = diversity_bonus

    def detect_cold_start_user(
        self,
        user_id: str,
        users: List[User],
        user_interactions: Dict[str, List[Interaction]],
    ) -> ColdStartInfo:
        """Detect if a user is in cold-start state."""
        user = next((u for u in users if u.user_id == user_id), None)
        
        if user is None:
            return ColdStartInfo(
                is_cold_start=True,
                strategy="unknown_user",
                reason="User not found in system — treating as fully cold start",
                fallback_used=True,
            )
        
        interaction_count = len(user_interactions.get(user_id, []))
        
        if interaction_count == 0:
            if user.preferred_categories:
                return ColdStartInfo(
                    is_cold_start=True,
                    strategy="preference_guided",
                    reason=f"New user with declared preferences: "
                           f"{', '.join(user.preferred_categories)}",
                )
            return ColdStartInfo(
                is_cold_start=True,
                strategy="global_popularity",
                reason="New user with no interactions or preferences",
                fallback_used=True,
            )
        
        if interaction_count < COLD_START_USER_THRESHOLD:
            return ColdStartInfo(
                is_cold_start=True,
                strategy="partial_cold_start",
                reason=f"User has only {interaction_count} interactions "
                       f"(threshold: {COLD_START_USER_THRESHOLD})",
            )
        
        return ColdStartInfo(
            is_cold_start=False,
            strategy="none",
            reason="User has sufficient interaction history",
        )

    def detect_cold_start_product(
        self,
        product: Product,
        product_interactions: Dict[str, List[Interaction]],
        reference_time: Optional[datetime] = None,
    ) -> ColdStartInfo:
        """Detect if a product is in cold-start state."""
        if reference_time is None:
            reference_time = datetime.utcnow()
        
        interaction_count = len(
            product_interactions.get(product.product_id, [])
        )
        
        hours_since_creation = max(
            (reference_time - product.created_at).total_seconds() / 3600, 0.1
        )
        
        if interaction_count < COLD_START_PRODUCT_THRESHOLD:
            if hours_since_creation < NEW_PRODUCT_WINDOW_HOURS:
                return ColdStartInfo(
                    is_cold_start=True,
                    strategy="new_product_exploration",
                    reason=f"New product (created {hours_since_creation:.0f}h ago) "
                           f"with only {interaction_count} interactions",
                )
            return ColdStartInfo(
                is_cold_start=True,
                strategy="underperforming_product",
                reason=f"Product has only {interaction_count} interactions "
                       f"after {hours_since_creation:.0f}h",
            )
        
        return ColdStartInfo(
            is_cold_start=False,
            strategy="none",
            reason="Product has sufficient interaction data",
        )

    def recommend_for_cold_start_user(
        self,
        user_id: str,
        users: List[User],
        products: List[Product],
        interactions: List[Interaction],
        trend_signals: List[TrendSignal],
        n: int = 10,
    ) -> Tuple[List[RecommendedProduct], str]:
        """
        Generate recommendations for a cold-start user.
        
        Returns:
            Tuple of (recommendations, strategy_used)
        """
        user = next((u for u in users if u.user_id == user_id), None)
        user_interactions_map: Dict[str, List[Interaction]] = defaultdict(list)
        for inter in interactions:
            user_interactions_map[inter.user_id].append(inter)
        
        cold_info = self.detect_cold_start_user(
            user_id, users, user_interactions_map
        )
        
        if not cold_info.is_cold_start:
            return [], "not_cold_start"
        
        # ── Compute product popularity scores ──
        product_popularity = self._compute_product_popularity(
            products, interactions
        )
        
        # ── Build trend score lookup ──
        trend_lookup = {ts.product_id: ts.trend_score for ts in trend_signals}
        
        # ── Strategy: Preference-Guided ──
        if cold_info.strategy == "preference_guided" and user is not None:
            return self._preference_guided_recommend(
                user, products, product_popularity, trend_lookup, n
            ), "cold_start_preference_guided"
        
        # ── Strategy: Partial Cold Start ──
        if cold_info.strategy == "partial_cold_start":
            return self._partial_cold_start_recommend(
                user_id, users, products, interactions,
                product_popularity, trend_lookup, n
            ), "cold_start_partial"
        
        # ── Strategy: Global Popularity ──
        return self._global_popularity_recommend(
            products, product_popularity, trend_lookup, n
        ), "cold_start_global_popularity"

    def _compute_product_popularity(
        self,
        products: List[Product],
        interactions: List[Interaction],
    ) -> Dict[str, float]:
        """Compute normalized popularity score for each product."""
        interaction_counts: Dict[str, float] = defaultdict(float)
        
        for inter in interactions:
            weight = {
                InteractionType.VIEW: 1.0,
                InteractionType.CLICK: 2.0,
                InteractionType.ADD_TO_CART: 3.5,
                InteractionType.PURCHASE: 5.0,
                InteractionType.RATING: 2.5,
            }.get(inter.interaction_type, 1.0)
            interaction_counts[inter.product_id] += weight
        
        max_pop = max(interaction_counts.values()) if interaction_counts else 1
        return {
            pid: count / max_pop
            for pid, count in interaction_counts.items()
        }

    def _preference_guided_recommend(
        self,
        user: User,
        products: List[Product],
        popularity: Dict[str, float],
        trends: Dict[str, float],
        n: int,
    ) -> List[RecommendedProduct]:
        """Recommend based on declared category preferences + popularity."""
        candidates = []
        
        for product in products:
            pop_score = popularity.get(product.product_id, 0)
            trend_score = trends.get(product.product_id, 0) / 100  # Normalize
            
            # Category match bonus
            cat_match = 1.0 if product.category in user.preferred_categories else 0.3
            
            # Final score
            score = (
                0.4 * cat_match +
                0.3 * pop_score +
                0.2 * trend_score +
                0.1 * (product.avg_rating / 5.0)
            )
            
            reasons = []
            if product.category in user.preferred_categories:
                reasons.append(
                    f"Matches your interest in {product.category}"
                )
            if pop_score > 0.5:
                reasons.append("Popular among all users")
            if trend_score > 0.3:
                reasons.append(
                    f"Trending in {product.category}"
                )
            if not reasons:
                reasons.append("Recommended for discovery")
            
            candidates.append(RecommendedProduct(
                product=product,
                score=round(score, 4),
                rank=0,
                explanation=RecommendationExplanation(
                    cold_start_score=round(score, 4),
                    trend_score=round(trend_score, 4),
                    final_score=round(score, 4),
                    reasons=reasons,
                    trend_contribution_pct=round(trend_score * 100 / max(score, 0.01), 1),
                    personalization_contribution_pct=round(cat_match * 100 / max(score, 0.01), 1),
                ),
            ))
        
        candidates.sort(key=lambda x: x.score, reverse=True)
        
        # Apply diversity
        candidates = self._ensure_category_diversity(candidates, n)
        
        for i, rec in enumerate(candidates[:n]):
            rec.rank = i + 1
        
        return candidates[:n]

    def _partial_cold_start_recommend(
        self,
        user_id: str,
        users: List[User],
        products: List[Product],
        interactions: List[Interaction],
        popularity: Dict[str, float],
        trends: Dict[str, float],
        n: int,
    ) -> List[RecommendedProduct]:
        """
        For users with 1-4 interactions: use those interactions to infer
        preferences, combined with popularity and trends.
        """
        user_ints = [i for i in interactions if i.user_id == user_id]
        
        # Infer preferred categories from sparse history
        cat_weights: Dict[str, float] = defaultdict(float)
        interacted_products: Set[str] = set()
        product_map = {p.product_id: p for p in products}
        
        for inter in user_ints:
            interacted_products.add(inter.product_id)
            if inter.product_id in product_map:
                cat = product_map[inter.product_id].category
                cat_weights[cat] += 1
        
        candidates = []
        for product in products:
            if product.product_id in interacted_products:
                continue
            
            pop_score = popularity.get(product.product_id, 0)
            trend_score = trends.get(product.product_id, 0) / 100
            
            # Inferred category preference
            cat_score = cat_weights.get(product.category, 0)
            max_cat = max(cat_weights.values()) if cat_weights else 1
            cat_normalized = cat_score / max_cat if max_cat > 0 else 0
            
            score = (
                0.35 * cat_normalized +
                0.3 * pop_score +
                0.25 * trend_score +
                0.1 * (product.avg_rating / 5.0)
            )
            
            reasons = []
            if cat_normalized > 0.5:
                reasons.append(
                    f"Based on your recent interest in {product.category}"
                )
            if pop_score > 0.5:
                reasons.append("Popular choice among users")
            if trend_score > 0.3:
                reasons.append(f"Currently trending")
            if not reasons:
                reasons.append("Suggested for you to explore")
            
            candidates.append(RecommendedProduct(
                product=product,
                score=round(score, 4),
                rank=0,
                explanation=RecommendationExplanation(
                    cold_start_score=round(score, 4),
                    trend_score=round(trend_score, 4),
                    final_score=round(score, 4),
                    reasons=reasons,
                    trend_contribution_pct=round(trend_score * 100 / max(score, 0.01), 1),
                    personalization_contribution_pct=round(
                        cat_normalized * 100 / max(score, 0.01), 1
                    ),
                ),
            ))
        
        candidates.sort(key=lambda x: x.score, reverse=True)
        candidates = self._ensure_category_diversity(candidates, n)
        
        for i, rec in enumerate(candidates[:n]):
            rec.rank = i + 1
        
        return candidates[:n]

    def _global_popularity_recommend(
        self,
        products: List[Product],
        popularity: Dict[str, float],
        trends: Dict[str, float],
        n: int,
    ) -> List[RecommendedProduct]:
        """Pure popularity + trending based recommendations for unknown users."""
        candidates = []
        
        for product in products:
            pop_score = popularity.get(product.product_id, 0)
            trend_score = trends.get(product.product_id, 0) / 100
            
            score = (
                0.45 * pop_score +
                0.35 * trend_score +
                0.2 * (product.avg_rating / 5.0)
            )
            
            reasons = []
            if pop_score > 0.5:
                reasons.append("Highly popular among users")
            if trend_score > 0.3:
                reasons.append(f"Trending now in {product.category}")
            reasons.append("Recommended for new users")
            
            candidates.append(RecommendedProduct(
                product=product,
                score=round(score, 4),
                rank=0,
                explanation=RecommendationExplanation(
                    cold_start_score=round(score, 4),
                    trend_score=round(trend_score, 4),
                    final_score=round(score, 4),
                    reasons=reasons,
                    trend_contribution_pct=round(trend_score * 100 / max(score, 0.01), 1),
                    personalization_contribution_pct=0,
                ),
            ))
        
        candidates.sort(key=lambda x: x.score, reverse=True)
        candidates = self._ensure_category_diversity(candidates, n)
        
        for i, rec in enumerate(candidates[:n]):
            rec.rank = i + 1
        
        return candidates[:n]

    def _ensure_category_diversity(
        self,
        candidates: List[RecommendedProduct],
        n: int,
    ) -> List[RecommendedProduct]:
        """
        Ensure recommendations span multiple categories.
        Uses a round-robin approach by category, picking top items from each.
        """
        if len(candidates) <= n:
            return candidates
        
        by_category: Dict[str, List[RecommendedProduct]] = defaultdict(list)
        for c in candidates:
            by_category[c.product.category].append(c)
        
        result = []
        categories = list(by_category.keys())
        cat_idx = 0
        cat_pointers = {cat: 0 for cat in categories}
        
        while len(result) < n and categories:
            cat = categories[cat_idx % len(categories)]
            if cat_pointers[cat] < len(by_category[cat]):
                result.append(by_category[cat][cat_pointers[cat]])
                cat_pointers[cat] += 1
                cat_idx += 1
            else:
                categories.remove(cat)
                if categories:
                    cat_idx = cat_idx % len(categories)
        
        return result

    def get_new_products_for_exploration(
        self,
        products: List[Product],
        product_interactions: Dict[str, List[Interaction]],
        reference_time: Optional[datetime] = None,
        n: int = 5,
    ) -> List[Product]:
        """
        Identify new products that should receive an exploration boost.
        
        These are products created recently with insufficient interaction data,
        which benefit from increased visibility to collect initial signals.
        """
        if reference_time is None:
            reference_time = datetime.utcnow()
        
        new_products = []
        for product in products:
            cold_info = self.detect_cold_start_product(
                product, product_interactions, reference_time
            )
            if cold_info.is_cold_start and \
               cold_info.strategy == "new_product_exploration":
                new_products.append(product)
        
        # Sort by recency
        new_products.sort(key=lambda p: p.created_at, reverse=True)
        return new_products[:n]
