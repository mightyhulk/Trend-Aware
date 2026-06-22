"""
Synthetic Data Generator for the Recommendation System.

Generates realistic e-commerce data including:
- 500 users with varied demographics and preferences
- 200 products across 8 categories with realistic attributes
- 15,000+ interactions with temporal patterns that create trending signals

The generator creates time-distributed interactions to simulate:
- Organic browsing patterns
- Trending bursts (sudden spikes in specific product interactions)
- Seasonal preferences
- New user / new product scenarios for cold-start testing
"""

import random
import uuid
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import math

from models.schemas import (
    Product, User, Interaction, InteractionType, Gender
)


# ─────────────────────────────────────────────
# Product Catalog Templates
# ─────────────────────────────────────────────

CATEGORIES = {
    "Electronics": {
        "subcategories": ["Smartphones", "Laptops", "Headphones", "Tablets", "Smartwatches"],
        "brands": ["TechNova", "PixelPro", "SoundWave", "ZenithTech", "VoltEdge"],
        "tags_pool": ["wireless", "bluetooth", "4K", "OLED", "fast-charging", "AI-powered",
                      "noise-cancelling", "lightweight", "gaming", "professional"],
        "price_range": (49.99, 1999.99),
    },
    "Fashion": {
        "subcategories": ["T-Shirts", "Jeans", "Sneakers", "Jackets", "Accessories"],
        "brands": ["UrbanThread", "StyleCraft", "ModaVista", "PrimeFit", "LuxeWear"],
        "tags_pool": ["cotton", "slim-fit", "casual", "formal", "sustainable", "premium",
                      "limited-edition", "vintage", "streetwear", "designer"],
        "price_range": (19.99, 499.99),
    },
    "Home & Kitchen": {
        "subcategories": ["Cookware", "Furniture", "Decor", "Appliances", "Storage"],
        "brands": ["HomeHaven", "KitchenCraft", "CozyNest", "ModernLiving", "GreenHome"],
        "tags_pool": ["stainless-steel", "eco-friendly", "minimalist", "smart-home",
                      "space-saving", "handcrafted", "modern", "rustic", "ergonomic"],
        "price_range": (14.99, 899.99),
    },
    "Books": {
        "subcategories": ["Fiction", "Non-Fiction", "Tech", "Self-Help", "Academic"],
        "brands": ["PageTurner", "MindSpark", "InkWell", "BrightReads", "DeepDive"],
        "tags_pool": ["bestseller", "award-winning", "classic", "new-release", "hardcover",
                      "audiobook", "illustrated", "series", "standalone", "anthology"],
        "price_range": (8.99, 59.99),
    },
    "Sports & Fitness": {
        "subcategories": ["Yoga", "Running", "Gym Equipment", "Outdoor", "Supplements"],
        "brands": ["FlexForce", "RunElite", "GymPro", "TrailBlaze", "VitalFuel"],
        "tags_pool": ["lightweight", "durable", "waterproof", "portable", "professional-grade",
                      "beginner-friendly", "advanced", "organic", "high-protein", "recovery"],
        "price_range": (12.99, 599.99),
    },
    "Beauty & Personal Care": {
        "subcategories": ["Skincare", "Makeup", "Haircare", "Fragrance", "Grooming"],
        "brands": ["GlowLab", "PureSkin", "VelvetTouch", "AuraScent", "FreshEdge"],
        "tags_pool": ["organic", "vegan", "cruelty-free", "dermatologist-tested", "SPF",
                      "anti-aging", "hydrating", "mattifying", "long-lasting", "sensitive-skin"],
        "price_range": (9.99, 199.99),
    },
    "Toys & Games": {
        "subcategories": ["Board Games", "Puzzles", "Action Figures", "Educational", "Outdoor Toys"],
        "brands": ["FunFactory", "BrainBox", "PlayPeak", "WonderWorld", "KidCraft"],
        "tags_pool": ["ages-3+", "ages-8+", "family", "strategy", "cooperative",
                      "STEM", "creative", "collectible", "interactive", "award-winning"],
        "price_range": (9.99, 149.99),
    },
    "Grocery & Gourmet": {
        "subcategories": ["Snacks", "Beverages", "Organic", "International", "Baking"],
        "brands": ["FreshBite", "PurePantry", "GlobalTaste", "NatureNest", "BakeJoy"],
        "tags_pool": ["organic", "gluten-free", "non-GMO", "vegan", "keto-friendly",
                      "artisanal", "fair-trade", "sugar-free", "high-fiber", "superfood"],
        "price_range": (3.99, 79.99),
    },
}

PRODUCT_NAME_TEMPLATES = {
    "Electronics": [
        "{brand} {sub} Pro Max", "{brand} {sub} Ultra", "{brand} {sub} Lite",
        "{brand} {sub} X1", "{brand} {sub} Elite Series",
    ],
    "Fashion": [
        "{brand} Classic {sub}", "{brand} Premium {sub}", "{brand} Street {sub}",
        "{brand} Urban {sub}", "{brand} Signature {sub}",
    ],
    "Home & Kitchen": [
        "{brand} {sub} Set", "{brand} Premium {sub}", "{brand} Smart {sub}",
        "{brand} Essential {sub}", "{brand} Deluxe {sub}",
    ],
    "Books": [
        "The Art of {tag}", "Mastering {tag}", "Journey Through {tag}",
        "Essential Guide to {tag}", "Deep Dive into {tag}",
    ],
    "Sports & Fitness": [
        "{brand} {sub} Pro", "{brand} {sub} Starter Kit", "{brand} {sub} Bundle",
        "{brand} Elite {sub}", "{brand} {sub} Essentials",
    ],
    "Beauty & Personal Care": [
        "{brand} {sub} Collection", "{brand} {sub} Essentials", "{brand} Radiance {sub}",
        "{brand} {sub} Luxury Set", "{brand} Pure {sub}",
    ],
    "Toys & Games": [
        "{brand} {sub} Adventure", "{brand} {sub} Challenge", "{brand} Super {sub}",
        "{brand} {sub} Deluxe", "{brand} {sub} Explorer",
    ],
    "Grocery & Gourmet": [
        "{brand} {sub} Selection", "{brand} Premium {sub}", "{brand} {sub} Pack",
        "{brand} Artisan {sub}", "{brand} {sub} Variety Box",
    ],
}


def _generate_product_id(idx: int) -> str:
    """Generate a deterministic product ID."""
    return f"prod_{idx:04d}"


def _generate_user_id(idx: int) -> str:
    """Generate a deterministic user ID."""
    return f"user_{idx:04d}"


def _generate_interaction_id() -> str:
    """Generate a unique interaction ID."""
    return uuid.uuid4().hex[:16]


class DataGenerator:
    """
    Generates synthetic e-commerce data for the recommendation system.
    
    Uses a seeded random generator for reproducibility while creating
    realistic data distributions including temporal trending patterns.
    """

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.products: List[Product] = []
        self.users: List[User] = []
        self.interactions: List[Interaction] = []
        self.now = datetime(2026, 6, 20, 12, 0, 0)  # Fixed reference time

    def generate_all(
        self,
        num_users: int = 500,
        num_products: int = 200,
        num_interactions: int = 15000,
        num_trending_bursts: int = 8,
    ) -> Tuple[List[User], List[Product], List[Interaction]]:
        """
        Generate the complete synthetic dataset.
        
        Args:
            num_users: Number of users to generate
            num_products: Number of products to generate
            num_interactions: Base number of interactions
            num_trending_bursts: Number of products to have trending bursts
            
        Returns:
            Tuple of (users, products, interactions)
        """
        self.products = self._generate_products(num_products)
        self.users = self._generate_users(num_users)
        self.interactions = self._generate_interactions(
            num_interactions, num_trending_bursts
        )
        return self.users, self.products, self.interactions

    def _generate_products(self, count: int) -> List[Product]:
        """Generate a catalog of products across all categories."""
        products = []
        categories = list(CATEGORIES.keys())
        
        # Distribute products roughly evenly, with slight variation
        per_category = count // len(categories)
        remainder = count % len(categories)
        
        idx = 0
        for cat_idx, category in enumerate(categories):
            cat_info = CATEGORIES[category]
            n = per_category + (1 if cat_idx < remainder else 0)
            
            for j in range(n):
                subcategory = self.rng.choice(cat_info["subcategories"])
                brand = self.rng.choice(cat_info["brands"])
                tags = self.rng.sample(
                    cat_info["tags_pool"],
                    k=self.rng.randint(2, min(5, len(cat_info["tags_pool"])))
                )
                
                # Generate product name
                template = self.rng.choice(PRODUCT_NAME_TEMPLATES[category])
                name = template.format(
                    brand=brand,
                    sub=subcategory,
                    tag=self.rng.choice(tags).replace("-", " ").title()
                )
                
                # Price with realistic distribution (log-normal within range)
                price_min, price_max = cat_info["price_range"]
                price = round(
                    price_min + (price_max - price_min) * self.rng.betavariate(2, 5),
                    2
                )
                
                # Some products are "new" (created recently) for cold-start testing
                if j >= n - 2:  # Last 2 per category are new
                    created_at = self.now - timedelta(hours=self.rng.randint(1, 24))
                else:
                    created_at = self.now - timedelta(
                        days=self.rng.randint(30, 365)
                    )
                
                description = (
                    f"Premium {subcategory.lower()} from {brand}. "
                    f"Features: {', '.join(tags[:3])}. "
                    f"Perfect for those looking for quality {category.lower()} products."
                )
                
                products.append(Product(
                    product_id=_generate_product_id(idx),
                    name=name,
                    category=category,
                    subcategory=subcategory,
                    brand=brand,
                    price=price,
                    description=description,
                    tags=tags,
                    created_at=created_at,
                    avg_rating=round(self.rng.uniform(2.5, 5.0), 1),
                ))
                idx += 1
        
        return products

    def _generate_users(self, count: int) -> List[User]:
        """Generate users with diverse demographics and preferences."""
        users = []
        first_names = [
            "Alex", "Jordan", "Sam", "Morgan", "Casey", "Riley", "Taylor",
            "Quinn", "Avery", "Drew", "Blake", "Cameron", "Dakota", "Emery",
            "Finley", "Harper", "Kai", "Logan", "Micah", "Noel", "Peyton",
            "Reese", "Sage", "Tatum", "Val", "Wren", "Zion", "Arden",
            "Brook", "Charlie", "Devon", "Ellis", "Frankie", "Gray", "Haven",
            "Indigo", "Jules", "Kit", "Lane", "Milan",
        ]
        
        categories = list(CATEGORIES.keys())
        
        new_user_count = max(2, count // 25)  # ~4% are new users
        for i in range(count):
            # Some users are "new" (for cold-start testing)
            if i >= count - new_user_count:
                signup_date = self.now - timedelta(minutes=self.rng.randint(5, 120))
                interaction_count = 0
            else:
                signup_date = self.now - timedelta(
                    days=self.rng.randint(30, 365)
                )
                interaction_count = self.rng.randint(5, 200)
            
            # Users prefer 1-3 categories
            num_preferred = self.rng.randint(1, 3)
            preferred_cats = self.rng.sample(categories, num_preferred)
            
            name = self.rng.choice(first_names)
            suffix = hashlib.md5(f"{name}{i}".encode()).hexdigest()[:4]
            
            users.append(User(
                user_id=_generate_user_id(i),
                username=f"{name.lower()}_{suffix}",
                age=self.rng.randint(18, 65),
                gender=self.rng.choice(list(Gender)),
                preferred_categories=preferred_cats,
                signup_date=signup_date,
                interaction_count=interaction_count,
            ))
        
        return users

    def _generate_interactions(
        self, count: int, num_trending_bursts: int
    ) -> List[Interaction]:
        """
        Generate time-distributed interactions with trending bursts.
        
        Creates a realistic interaction timeline spanning 30 days, with
        specific products receiving "trending bursts" — sudden spikes in
        interactions within the last 24-48 hours.
        """
        interactions = []
        
        # Separate new users and established users
        established_users = [u for u in self.users if u.interaction_count > 0]
        
        # Select products for trending bursts
        eligible_for_trending = [
            p.product_id for p in self.products
            if (self.now - p.created_at).days > 7
        ]
        burst_count = min(num_trending_bursts, len(eligible_for_trending))
        trending_product_ids = self.rng.sample(
            eligible_for_trending, burst_count
        ) if burst_count > 0 else []
        
        # Interaction type weights (view is most common, purchase is rarest)
        type_weights = {
            InteractionType.VIEW: 0.45,
            InteractionType.CLICK: 0.25,
            InteractionType.ADD_TO_CART: 0.15,
            InteractionType.PURCHASE: 0.08,
            InteractionType.RATING: 0.07,
        }
        types = list(type_weights.keys())
        weights = list(type_weights.values())
        
        # ── Generate base organic interactions ──
        for _ in range(count):
            user = self.rng.choice(established_users)
            
            # User more likely to interact with preferred categories
            if self.rng.random() < 0.65:
                preferred_products = [
                    p for p in self.products
                    if p.category in user.preferred_categories
                ]
                if preferred_products:
                    product = self.rng.choice(preferred_products)
                else:
                    product = self.rng.choice(self.products)
            else:
                product = self.rng.choice(self.products)
            
            # Time distribution: more recent interactions are more common
            # Using exponential distribution to simulate recency bias
            hours_ago = int(abs(self.rng.expovariate(1 / 168)))  # ~7 day mean
            hours_ago = min(hours_ago, 30 * 24)  # Cap at 30 days
            timestamp = self.now - timedelta(hours=hours_ago)
            
            interaction_type = self.rng.choices(types, weights=weights, k=1)[0]
            
            rating = None
            if interaction_type == InteractionType.RATING:
                rating = round(self.rng.triangular(1, 5, 4), 1)
            
            interactions.append(Interaction(
                interaction_id=_generate_interaction_id(),
                user_id=user.user_id,
                product_id=product.product_id,
                interaction_type=interaction_type,
                timestamp=timestamp,
                rating=rating,
                session_id=uuid.uuid4().hex[:8],
            ))
        
        # ── Generate trending bursts ──
        # These create noticeable spikes for specific products in the last 24h
        for prod_id in trending_product_ids:
            burst_size = self.rng.randint(80, 200)
            for _ in range(burst_size):
                user = self.rng.choice(established_users)
                hours_ago = self.rng.expovariate(1 / 6)  # Concentrated in last ~6h
                hours_ago = min(hours_ago, 24)
                timestamp = self.now - timedelta(hours=hours_ago)
                
                interaction_type = self.rng.choices(types, weights=weights, k=1)[0]
                rating = None
                if interaction_type == InteractionType.RATING:
                    rating = round(self.rng.triangular(3, 5, 4.5), 1)
                
                interactions.append(Interaction(
                    interaction_id=_generate_interaction_id(),
                    user_id=user.user_id,
                    product_id=prod_id,
                    interaction_type=interaction_type,
                    timestamp=timestamp,
                    rating=rating,
                    session_id=uuid.uuid4().hex[:8],
                ))
        
        # Sort by timestamp
        interactions.sort(key=lambda x: x.timestamp)
        
        return interactions


def create_dataset(
    num_users: int = 500,
    num_products: int = 200,
    num_interactions: int = 15000,
    seed: int = 42,
) -> Tuple[List[User], List[Product], List[Interaction]]:
    """
    Convenience function to generate the complete dataset.
    
    Returns:
        Tuple of (users, products, interactions)
    """
    generator = DataGenerator(seed=seed)
    return generator.generate_all(
        num_users=num_users,
        num_products=num_products,
        num_interactions=num_interactions,
    )
