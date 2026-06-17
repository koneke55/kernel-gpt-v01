from __future__ import annotations
from typing import Dict
import shutil
import subprocess


class Profiler:
    """Wrapper to call `ncu` (NVIDIA Nsight) and parse CSV output. Optional dependency."""

    @staticmethod
    def profile_ncu(kernel_path: str, spec) -> Dict[str, float]:
        if not shutil.which("ncu"):
            return {"note": "ncu not available"}
        cmd = ["ncu", "--target-processes", "all", "--csv", kernel_path]
        out = subprocess.run(cmd, capture_output=True, text=True)
        # crude parse: return raw output size
        return {"stdout_len": len(out.stdout)}
