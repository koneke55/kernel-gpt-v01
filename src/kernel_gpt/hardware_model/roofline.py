from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import math
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class RooflineModel:
    """Simple roofline model for GPUs.

    Attributes:
        peak_tflops_by_dtype: mapping like {"FP32": tf_lops}
    """
    peak_tflops_by_dtype: Dict[str, float]

    def classify_workload(self, arithmetic_intensity: float, dtype: str = "FP32") -> str:
        """Classify workload as memory_bound, compute_bound, or balanced.

        Args:
            arithmetic_intensity: FLOPs per byte
            dtype: data type key in peak_tflops_by_dtype

        Returns:
            classification string
        """
        if dtype not in self.peak_tflops_by_dtype:
            raise KeyError(f"Unknown dtype {dtype}")
        peak = self.peak_tflops_by_dtype[dtype]
        # crude roofline: if intensity * memory_bw < peak -> memory bound
        # here we assume a nominal memory bandwidth; user should compare externally
        if arithmetic_intensity < 1.0:
            return "memory_bound"
        if arithmetic_intensity > 10.0:
            return "compute_bound"
        return "balanced"

    def plot_roofline(self, save_path: str | None = None):
        """Plot a very simple roofline curve (illustrative).

        Args:
            save_path: optional path to save the figure
        """
        xs = [10 ** (i / 10) for i in range(-20, 41)]
        plt.figure()
        for k, v in self.peak_tflops_by_dtype.items():
            ys = [min(v, 0.1 * xi * v) for xi in xs]
            plt.loglog(xs, ys, label=k)
        plt.xlabel("Arithmetic intensity (FLOPs/byte)")
        plt.ylabel("Performance (TFLOP/s)")
        plt.legend()
        plt.grid(True, which="both")
        if save_path:
            plt.savefig(save_path)
        return plt
