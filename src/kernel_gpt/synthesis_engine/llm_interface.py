from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict, Any
import hashlib
import json
from .strategy import Strategy, HeuristicStrategySelector
from .llm_clients import generate_strategy


@dataclass
class LLMArchitect:
    """Simple wrapper that would call an LLM to design a strategy.

    This implementation uses a cached heuristic fallback so it runs without API keys.
    """

    cache: Dict[str, Strategy] = None

    def __post_init__(self):
        if self.cache is None:
            self.cache = {}

    def design_strategy(self, spec, gpu) -> Strategy:
        key = hashlib.sha1(json.dumps({
            "spec": spec.name,
            "gpu": getattr(gpu, "name", str(gpu))
        }).encode()).hexdigest()
        if key in self.cache:
            return self.cache[key]
        # Try to get a structured strategy from an LLM (OpenAI or Anthropic).
        # If unavailable or the call fails, fall back to the heuristic selector.
        try:
            strat = generate_strategy(spec, gpu) or HeuristicStrategySelector.select(spec, gpu)
        except Exception:
            strat = HeuristicStrategySelector.select(spec, gpu)
        self.cache[key] = strat
        return strat
