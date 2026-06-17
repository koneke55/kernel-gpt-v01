from __future__ import annotations
from dataclasses import dataclass
from typing import List


@dataclass
class NumericalChecker:
    """Check numerical stability given spec and operations."""

    @staticmethod
    def check_dtype_suitability(spec) -> List[str]:
        issues = []
        for t in list(spec.inputs) + list(spec.outputs):
            if t.dtype.name == "FP8" and spec.compute_pattern.name in ("reduce",):
                issues.append("FP8 reductions may be unstable; consider FP16 or accumulation in FP32")
        return issues
