from dataclasses import dataclass
from .base import GPUArchitecture


@dataclass(frozen=True)
class AMD_MI300X(GPUArchitecture):
    name: str = "AMD MI300X"

    @property
    def sm_count(self) -> int:
        return 144

    @property
    def clock_ghz(self) -> float:
        return 2.2

    @property
    def memory_bw_gbps(self) -> float:
        return 3840.0

    @property
    def l2_cache_size(self) -> int:
        return 128 * 1024 * 1024

    @property
    def shared_mem_per_sm(self) -> int:
        return 256 * 1024

    @property
    def max_registers_per_sm(self) -> int:
        return 256 * 1024

    @property
    def warp_size(self) -> int:
        return 64

    @property
    def tensor_core_specs(self):
        return {"matrix": {"max_tflops_fp16": 2500.0}}
