from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Tuple


@dataclass
class MemoryEstimator:
    """Estimate shared memory and rough register usage from kernel code."""

    @staticmethod
    def parse_shared_bytes(code: str) -> int:
        # parse patterns like __shared__ float buf[1024];
        total = 0
        for m in re.finditer(r"__shared__\s+\w+\s+(\w+)\s*\[\s*(\d+)\s*\]", code):
            sz = int(m.group(2))
            # assume 4 bytes per element as heuristic
            total += sz * 4
        return total

    @staticmethod
    def estimate_registers(code: str) -> int:
        # crude heuristic: count local variable declarations
        decls = re.findall(r"\bfloat\b|\bint\b|\bdouble\b", code)
        return max(32, len(decls) * 4)
