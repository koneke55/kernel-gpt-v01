from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict, Any
import hashlib
import json
from .strategy import Strategy, HeuristicStrategySelector


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
        # TODO: integrate with OpenAI or Anthropic for structured JSON responses
        strat = HeuristicStrategySelector.select(spec, gpu)
        self.cache[key] = strat
        return strat
