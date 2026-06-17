from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass(frozen=True)
class Strategy:
    block_size: int
    tile_sizes: Dict[str, int]
    use_shared_mem: bool
    use_tensor_cores: bool
    num_stages: int = 2
    thread_layout: str = "bxk"
    optimization_notes: str = ""


class HeuristicStrategySelector:
    """Fallback strategy selector used if LLM is unavailable."""

    @staticmethod
    def select(spec, gpu) -> Strategy:
        # very simple heuristic: choose tile sizes based on dtype
        dtype = spec.inputs[0].dtype
        if getattr(dtype, "name", str(dtype)).lower().startswith("fp16"):
            tile = {"M": 128, "N": 128, "K": 32}
            use_tc = True
        else:
            tile = {"M": 64, "N": 64, "K": 16}
            use_tc = False
        return Strategy(block_size=256, tile_sizes=tile, use_shared_mem=True, use_tensor_cores=use_tc)
