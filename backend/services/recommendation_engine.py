"""
Recommendation Engine — Hybrid Collaborative + Content-Based Filtering.

Implements a production-grade hybrid recommendation system:

1. **Collaborative Filtering (SVD)**:
   - Builds a user-item interaction matrix
   - Applies Truncated SVD for dimensionality reduction
   - Predicts user preferences from latent factor space
   
2. **Content-Based Filtering (TF-IDF + Cosine Similarity)**:
   - Creates product feature vectors from descriptions, tags, categories
   - Computes cosine similarity between user profile and product features
   
3. **Hybrid Fusion**:
   - Weighted combination: α × collaborative + (1-α) × content-based
   - α is adaptive based on user interaction history density

Design Justification:
- **Why Hybrid?** Pure collaborative filtering fails on cold-start; 
  pure content-based creates filter bubbles. Hybrid gets the best of both.
- **Why SVD?** Handles sparse matrices well, scales to millions of users,
  and captures latent patterns that explicit features miss.
- **Why TF-IDF?** Lightweight, interpretable, and effective for product 
  feature extraction without requiring deep learning infrastructure.
"""

import math
import numpy as np
from typing import List, Dict, Optional, Tuple, Set
from collections import defaultdict
from datetime import datetime, timedelta

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

from models.schemas import (
    Product, User, Interaction, InteractionType,
    RecommendedProduct, RecommendationExplanation,
)


# Interaction weights for building the user-item matrix
INTERACTION_VALUE = {
    InteractionType.VIEW: 1.0,
    InteractionType.CLICK: 2.0,
    InteractionType.ADD_TO_CART: 3.5,
    InteractionType.PURCHASE: 5.0,
    InteractionType.RATING: 3.0,  # Base value; actual rating adds to this
}


class RecommendationEngine:
    """
    Hybrid recommendation engine combining collaborative filtering (SVD)
    and content-based filtering (TF-IDF) with configurable fusion weights.
    """

    def __init__(
        self,
        n_latent_factors: int = 50,
        collaborative_weight: float = 0.6,
        min_interactions_for_cf: int = 3,
    ):
        """
        Initialize the recommendation engine.
        
        Args:
            n_latent_factors: Number of latent factors for SVD.
            collaborative_weight: Weight for collaborative scores (α).
                Content-based weight = 1 - α.
            min_interactions_for_cf: Minimum interactions needed to use
                collaborative filtering. Below this, content-based dominates.
        """
        self.n_latent_factors = n_latent_factors
        self.collaborative_weight = collaborative_weight
        self.min_interactions_for_cf = min_interactions_for_cf
        
        # State — populated during training
        self.user_item_matrix: Optional[np.ndarray] = None
        self.user_factors: Optional[np.ndarray] = None
        self.item_factors: Optional[np.ndarray] = None
        self.user_index: Dict[str, int] = {}
        self.item_index: Dict[str, int] = {}
        self.reverse_item_index: Dict[int, str] = {}
        self.product_features: Optional[np.ndarray] = None
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.products_map: Dict[str, Product] = {}
        self.user_interactions: Dict[str, List[Interaction]] = defaultdict(list)
        self.is_trained = False

    def train(
        self,
        users: List[User],
        products: List[Product],
        interactions: List[Interaction],
    ) -> Dict[str, any]:
        """
        Train both collaborative and content-based models.
        
        Returns:
            Training metrics dictionary
        """
        self.products_map = {p.product_id: p for p in products}
        
        # Index users and items
        self.user_index = {u.user_id: idx for idx, u in enumerate(users)}
        self.item_index = {p.product_id: idx for idx, p in enumerate(products)}
        self.reverse_item_index = {v: k for k, v in self.item_index.items()}
        
        # Group interactions by user
        self.user_interactions = defaultdict(list)
        for interaction in interactions:
            self.user_interactions[interaction.user_id].append(interaction)
        
        # ── Train Collaborative Filtering (SVD) ──
        cf_metrics = self._train_collaborative(users, products, interactions)
        
        # ── Train Content-Based (TF-IDF) ──
        cb_metrics = self._train_content_based(products)
        
        self.is_trained = True
        
        return {
            "collaborative": cf_metrics,
            "content_based": cb_metrics,
            "total_users": len(users),
            "total_products": len(products),
            "total_interactions": len(interactions),
            "trained_at": datetime.utcnow().isoformat(),
        }

    def _train_collaborative(
        self,
        users: List[User],
        products: List[Product],
        interactions: List[Interaction],
    ) -> Dict:
        """Build user-item matrix and compute SVD decomposition."""
        n_users = len(users)
        n_items = len(products)
        
        # Build sparse user-item matrix
        rows, cols, data = [], [], []
        
        for interaction in interactions:
            if interaction.user_id in self.user_index and \
               interaction.product_id in self.item_index:
                row = self.user_index[interaction.user_id]
                col = self.item_index[interaction.product_id]
                value = INTERACTION_VALUE.get(
                    interaction.interaction_type, 1.0
                )
                # For ratings, use the actual rating value
                if interaction.interaction_type == InteractionType.RATING \
                   and interaction.rating is not None:
                    value = interaction.rating
                
                rows.append(row)
                cols.append(col)
                data.append(value)
        
        # Create sparse matrix and aggregate duplicate entries
        sparse_matrix = csr_matrix(
            (data, (rows, cols)),
            shape=(n_users, n_items),
        )
        
        # Store dense matrix for predictions
        self.user_item_matrix = sparse_matrix.toarray()
        
        # Apply SVD
        n_components = min(self.n_latent_factors, min(n_users, n_items) - 1)
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        
        self.user_factors = svd.fit_transform(sparse_matrix)
        self.item_factors = svd.components_.T  # n_items × n_components
        
        explained_variance = svd.explained_variance_ratio_.sum()
        
        return {
            "matrix_shape": (n_users, n_items),
            "n_components": n_components,
            "explained_variance": round(explained_variance, 4),
            "sparsity": round(
                1 - len(data) / (n_users * n_items), 4
            ),
        }

    def _train_content_based(self, products: List[Product]) -> Dict:
        """Build TF-IDF feature vectors from product metadata."""
        # Create text representations of products
        product_texts = []
        for product in products:
            text = (
                f"{product.category} {product.subcategory} {product.brand} "
                f"{product.description} {' '.join(product.tags)}"
            )
            product_texts.append(text)
        
        # Fit TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=500,
            stop_words="english",
            ngram_range=(1, 2),
        )
        self.product_features = self.tfidf_vectorizer.fit_transform(
            product_texts
        ).toarray()
        
        return {
            "n_features": self.product_features.shape[1],
            "vocabulary_size": len(self.tfidf_vectorizer.vocabulary_),
        }

    def get_collaborative_scores(
        self, user_id: str
    ) -> Dict[str, float]:
        """
        Get collaborative filtering scores for all products for a user.
        
        Uses the latent factor space: score = user_factors · item_factors.T
        """
        if user_id not in self.user_index:
            return {}
        
        user_idx = self.user_index[user_id]
        user_vector = self.user_factors[user_idx]
        
        # Predict scores for all items
        scores = user_vector @ self.item_factors.T
        
        # Normalize to 0-1
        min_score = scores.min()
        max_score = scores.max()
        if max_score > min_score:
            scores = (scores - min_score) / (max_score - min_score)
        else:
            scores = np.zeros_like(scores)
        
        return {
            self.reverse_item_index[idx]: float(scores[idx])
            for idx in range(len(scores))
        }

    def get_content_based_scores(
        self, user_id: str
    ) -> Dict[str, float]:
        """
        Get content-based scores by building a user profile from
        their interaction history and computing similarity to all products.
        """
        user_ints = self.user_interactions.get(user_id, [])
        if not user_ints:
            return {}
        
        # Build user profile as weighted average of interacted product features
        profile = np.zeros(self.product_features.shape[1])
        total_weight = 0
        
        for interaction in user_ints:
            if interaction.product_id in self.item_index:
                idx = self.item_index[interaction.product_id]
                weight = INTERACTION_VALUE.get(
                    interaction.interaction_type, 1.0
                )
                profile += weight * self.product_features[idx]
                total_weight += weight
        
        if total_weight > 0:
            profile /= total_weight
        
        # Compute cosine similarity between user profile and all products
        profile_2d = profile.reshape(1, -1)
        similarities = cosine_similarity(
            profile_2d, self.product_features
        ).flatten()
        
        # Normalize to 0-1
        min_sim = similarities.min()
        max_sim = similarities.max()
        if max_sim > min_sim:
            similarities = (similarities - min_sim) / (max_sim - min_sim)
        else:
            similarities = np.zeros_like(similarities)
        
        return {
            self.reverse_item_index[idx]: float(similarities[idx])
            for idx in range(len(similarities))
        }

    def recommend(
        self,
        user_id: str,
        n: int = 10,
        exclude_interacted: bool = True,
        trend_scores: Optional[Dict[str, float]] = None,
        trend_weight: float = 0.3,
        diversity_factor: float = 0.2,
    ) -> Tuple[List[RecommendedProduct], str, int]:
        """
        Generate hybrid recommendations for a user.
        
        The final score is computed as:
        
            final = (1 - β) × [α × collaborative + (1-α) × content] + β × trend
        
        Where:
        - α = collaborative_weight (adaptive based on interaction density)
        - β = trend_weight
        - Diversity injection is applied post-scoring
        
        Args:
            user_id: User to recommend for
            n: Number of recommendations
            exclude_interacted: Whether to filter out already-interacted items
            trend_scores: Dict of product_id → trend_score (0-100)
            trend_weight: Weight for trending signal in final score
            diversity_factor: How much to boost category diversity (0-1)
            
        Returns:
            Tuple of (recommendations, strategy_used, total_candidates)
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before recommending")
        
        user_ints = self.user_interactions.get(user_id, [])
        interacted_ids: Set[str] = {i.product_id for i in user_ints}
        
        # ── Determine strategy ──
        has_enough_history = len(user_ints) >= self.min_interactions_for_cf
        
        if user_id not in self.user_index:
            strategy = "cold_start_user"
        elif not has_enough_history:
            strategy = "content_dominant"
        else:
            strategy = "hybrid"
        
        # ── Get base scores ──
        cf_scores = {}
        cb_scores = {}
        
        if strategy == "hybrid":
            cf_scores = self.get_collaborative_scores(user_id)
            cb_scores = self.get_content_based_scores(user_id)
            alpha = self.collaborative_weight
        elif strategy == "content_dominant":
            cb_scores = self.get_content_based_scores(user_id)
            alpha = 0.1  # Almost entirely content-based
            cf_scores = self.get_collaborative_scores(user_id) \
                if user_id in self.user_index else {}
        else:
            # Cold start — will be handled by ColdStartHandler
            alpha = 0.0
        
        # ── Normalize trend scores to 0-1 ──
        normalized_trends: Dict[str, float] = {}
        if trend_scores:
            max_trend = max(trend_scores.values()) if trend_scores else 1
            normalized_trends = {
                pid: score / max_trend if max_trend > 0 else 0
                for pid, score in trend_scores.items()
            }
        
        # ── Compute final scores ──
        candidates: List[RecommendedProduct] = []
        
        for product_id, product in self.products_map.items():
            if exclude_interacted and product_id in interacted_ids:
                continue
            
            cf_score = cf_scores.get(product_id, 0.0)
            cb_score = cb_scores.get(product_id, 0.0)
            trend_score = normalized_trends.get(product_id, 0.0)
            
            # Hybrid fusion
            personalized_score = alpha * cf_score + (1 - alpha) * cb_score
            final_score = (
                (1 - trend_weight) * personalized_score +
                trend_weight * trend_score
            )
            
            # Compute contribution percentages
            total_contribution = (
                (1 - trend_weight) * personalized_score +
                trend_weight * trend_score
            )
            
            if total_contribution > 0:
                trend_pct = (trend_weight * trend_score / total_contribution) * 100
                personal_pct = 100 - trend_pct
            else:
                trend_pct = 0
                personal_pct = 0
            
            # Build explanation
            reasons = []
            if cf_score > 0.5:
                reasons.append(
                    "Recommended based on similar users' preferences"
                )
            if cb_score > 0.5:
                # Find what user has interacted with in this category
                similar_cats = [
                    self.products_map[iid].name 
                    for iid in list(interacted_ids)[:3]
                    if iid in self.products_map and
                    self.products_map[iid].category == product.category
                ]
                if similar_cats:
                    reasons.append(
                        f"Similar to items you've shown interest in: "
                        f"{', '.join(similar_cats[:2])}"
                    )
                else:
                    reasons.append(
                        f"Matches your interest in {product.category}"
                    )
            if trend_score > 0.3:
                reasons.append(
                    f"Currently trending in {product.category} "
                    f"(trend score: {trend_score * 100:.0f})"
                )
            if not reasons:
                reasons.append("Recommended for you based on overall preferences")
            
            explanation = RecommendationExplanation(
                collaborative_score=round(cf_score, 4),
                content_score=round(cb_score, 4),
                trend_score=round(trend_score, 4),
                final_score=round(final_score, 4),
                reasons=reasons,
                trend_contribution_pct=round(trend_pct, 1),
                personalization_contribution_pct=round(personal_pct, 1),
            )
            
            candidates.append(RecommendedProduct(
                product=product,
                score=round(final_score, 4),
                rank=0,  # Will be set after sorting
                explanation=explanation,
            ))
        
        # ── Apply diversity injection ──
        if diversity_factor > 0:
            candidates = self._apply_diversity(candidates, diversity_factor, n)
        
        # Sort by score and assign ranks
        candidates.sort(key=lambda x: x.score, reverse=True)
        top_n = candidates[:n]
        for i, rec in enumerate(top_n):
            rec.rank = i + 1
        
        return top_n, strategy, len(candidates)

    def _apply_diversity(
        self,
        candidates: List[RecommendedProduct],
        diversity_factor: float,
        n: int,
    ) -> List[RecommendedProduct]:
        """
        Apply category diversity to prevent filter bubbles.
        
        Uses Maximal Marginal Relevance (MMR) inspired approach:
        - Penalizes candidates from categories already well-represented
        - Boosts underrepresented categories proportionally
        """
        category_counts: Dict[str, int] = defaultdict(int)
        
        # Sort candidates by score first
        candidates.sort(key=lambda x: x.score, reverse=True)
        
        for candidate in candidates:
            cat = candidate.product.category
            count = category_counts[cat]
            
            # Apply diminishing returns penalty for repeated categories
            if count > 0:
                penalty = diversity_factor * math.log(1 + count) * 0.1
                candidate.score = max(0, candidate.score - penalty)
                candidate.explanation.final_score = candidate.score
            
            category_counts[cat] += 1
        
        return candidates
