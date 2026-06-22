from typing import List, Dict, Any, Optional

class AppState:
    """Container for application state."""
    def __init__(self):
        self.users = []
        self.products = []
        self.interactions = []
        self.engine = None
        self.trending_service = None
        self.cold_start_handler = None
        self.evolution_manager = None
        self.trend_signals = []
        self.training_metrics = {}
        self.startup_time = None
        self.request_latencies = []

state = AppState()
