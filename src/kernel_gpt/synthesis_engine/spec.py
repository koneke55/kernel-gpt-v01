from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Any
from enum import Enum


class DType(str, Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    FP8 = "fp8"
    INT8 = "int8"


@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: Tuple[int, ...]
    dtype: DType
    layout: str = "row_major"


@dataclass(frozen=True)
class ComputePattern:
    name: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class KernelSpec:
    name: str
    inputs: Tuple[TensorSpec, ...]
    outputs: Tuple[TensorSpec, ...]
    compute_pattern: ComputePattern
    constraints: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> Tuple[bool, List[str]]:
        """Validate shape compatibility and basic constraints.

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues: List[str] = []
        # simple shape checks: matching inner dims for matmul
        if self.compute_pattern.name == "matmul":
            a, b = self.inputs
            if len(a.shape) != 2 or len(b.shape) != 2:
                issues.append("matmul expects 2D tensors")
            else:
                if a.shape[1] != b.shape[0]:
                    issues.append("matmul inner dimensions mismatch")
        return (len(issues) == 0, issues)
