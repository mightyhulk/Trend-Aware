# AI-Driven Trend-Aware Product Recommendation System

A production-grade hybrid recommendation system that combines **personalized recommendations** with **real-time trending signals**, ensuring users see items that are both relevant to them and currently popular.

## Architecture

```
┌─────────────────────────────────────────────────┐
│           Frontend Dashboard (HTML/JS)          |
└──────────────────────┬──────────────────────────┘
                       │ REST API (FastAPI)
┌──────────────────────▼──────────────────────────┐
│              Backend Services                   │
│  ┌──────────────┐  ┌─────────────────────────┐  │
│  │   Hybrid     │  │ Trending Signal Service │  │
│  │ Rec Engine   │  │ (Exp. Decay Velocity)   │  │
│  │ (SVD+TF-IDF) │  └─────────────────────────┘  │
│  └──────────────┘  ┌─────────────────────────┐  │
│  ┌──────────────┐  │ System Evolution Manager│  │
│  │ Cold-Start   │  │ (Decay, Diversity, Anti-│  │
│  │ Handler      │  │  Amplification)         │  │
│  └──────────────┘  └─────────────────────────┘  │
│  ┌───────────────────────────────────────────┐  │
│  │       Explainability Module               │  │
│  └───────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
```

## Quick Start

```bash
# 1. Create virtual environment and install dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Run the server
cd backend
uvicorn main:app --reload --port 8000

# 3. Open in browser
# Dashboard: http://localhost:8000
# API Docs:  http://localhost:8000/docs
```

## Core Requirements Addressed

### 1. Recommendation Engine (Hybrid)

- **Collaborative Filtering (SVD)**: Truncated SVD on user-item interaction matrix captures latent preference patterns
- **Content-Based Filtering (TF-IDF)**: Product features vectorized via TF-IDF with cosine similarity
- **Hybrid Fusion**: `final = (1-β) × [α × CF + (1-α) × CB] + β × trending`
- **Why Hybrid?** Pure CF fails on cold-start; pure CB creates filter bubbles. Hybrid adapts α based on interaction density.

### 2. Trending Signal Integration

- **Exponential Time-Decay**: `weight = e^(-λ × Δt)` — recent interactions matter more
- **Interaction Velocity**: Weighted rate of views/clicks/purchases per hour
- **Trend Score**: `log(1 + velocity/baseline) × dampening` — prevents viral amplification
- **Configurable**: Window (1h-7d), decay rate, and blend weight are all tunable

### 3. Cold-Start Handling

| Scenario                   | Strategy                                                |
| -------------------------- | ------------------------------------------------------- |
| New user with preferences  | Category-popularity blend favoring declared interests   |
| New user, no preferences   | Global popularity + trending + category diversity       |
| User with 1-4 interactions | Inferred preferences + popularity fallback              |
| New product (<72h)         | Exploration boost + content similarity to popular items |

### 4. Explainability

Every recommendation includes:

- **Score breakdown**: Collaborative, Content-based, Trending scores
- **Human-readable reasons**: "Similar to items you liked", "Trending in Electronics"
- **Contribution percentages**: Visual bar showing personalization vs. trend influence

### 5. System Evolution

- **Trend Decay**: Exponential decay prevents stale trends; hard cutoff at 72h
- **Over-Amplification Detection**: Flags products with declining velocity but high scores
- **Diversity Monitoring**: Gini coefficient tracking, category coverage metrics
- **Retrain Scheduling**: Time-based (24h) and data-based (500+ new interactions) triggers
- **Feedback Loop Protection**: Diminishing returns penalty for over-represented categories

## Data

Synthetic but realistic e-commerce dataset:

- **500 users** with demographics and category preferences
- **200 products** across 8 categories (Electronics, Fashion, Books, etc.)
- **16,000+ interactions** with temporal patterns and trending bursts
- Cold-start users (last 20) and new products (last 2 per category) built-in

## Testing

```bash
# Run all tests (64 tests)
python -m pytest backend/tests/ -v

# Unit + Integration tests
python -m pytest backend/tests/test_comprehensive.py -v  # 47 tests

# API tests
python -m pytest backend/tests/test_api.py -v  # 17 tests
```

## Project Structure

```
Trend-Aware/
├── backend/
│   ├── main.py                         # FastAPI application entry point
│   ├── state.py                        # Application state management
│   ├── api/                            # API endpoints (routers)
│   │   ├── products.py
│   │   ├── recommendations.py
│   │   ├── system.py
│   │   └── users.py
│   ├── models/
│   │   └── schemas.py                  # Pydantic data models
│   ├── services/
│   │   ├── recommendation_engine.py    # Hybrid SVD + TF-IDF engine
│   │   ├── trending_service.py         # Trend signal computation
│   │   ├── cold_start_handler.py       # Cold-start strategies
│   │   └── evolution_manager.py        # System evolution & health
│   ├── data/
│   │   └── generator.py               # Synthetic data generation
│   └── tests/
│       ├── test_comprehensive.py       # 47 unit/integration tests
│       └── test_api.py                 # 17 API endpoint tests
├── frontend/
│   ├── index.html                      # Dashboard UI
│   ├── style.css                       # Dark-mode premium styles
│   └── app.js                          # Frontend logic
├── requirements.txt
└── README.md
```

## API Endpoints

| Method | Endpoint                           | Description                      |
| ------ | ---------------------------------- | -------------------------------- |
| GET    | `/api/health`                    | System health check              |
| POST   | `/api/recommendations`           | Get personalized recommendations |
| GET    | `/api/recommendations/{user_id}` | Simple recommendation endpoint   |
| GET    | `/api/trending`                  | Trending products (filterable)   |
| GET    | `/api/products`                  | List products                    |
| GET    | `/api/products/{product_id}`     | Get product details              |
| GET    | `/api/categories`                | List all product categories      |
| GET    | `/api/users`                     | List users                       |
| GET    | `/api/users/{user_id}`           | User details + cold-start info   |
| GET    | `/api/system/metrics`            | System health report             |
| GET    | `/api/system/evolution`          | Trend lifecycle & retrain status |
| GET    | `/api/cold-start/status`         | Cold-start population analysis   |

## Design Decisions

1. **SVD over Neural CF**: For this scale (~500 users, ~200 products), SVD provides excellent quality with sub-millisecond inference. Neural methods (NCF, autoencoders) add complexity without proportional benefit.
2. **Log dampening on trends**: `log(1 + x)` prevents exponential amplification of viral products while still capturing genuine surges.
3. **MMR-inspired diversity**: Maximal Marginal Relevance approach penalizes repeated categories, preventing filter bubbles even when the model has strong category preferences.
4. **Graceful cold-start degradation**: The system smoothly transitions from popularity-based → preference-guided → partial hybrid → full hybrid as user history grows.
