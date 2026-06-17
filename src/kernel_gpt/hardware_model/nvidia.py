from dataclasses import dataclass
from .base import GPUArchitecture


@dataclass(frozen=True)
class NVIDIA_H100(GPUArchitecture):
    name: str = "NVIDIA H100"

    @property
    def sm_count(self) -> int:
        return 132  # Hopper H100 SMs

    @property
    def clock_ghz(self) -> float:
        return 1.41

    @property
    def memory_bw_gbps(self) -> float:
        return 3860.0  # HBM3 effective GB/s (approx)

    @property
    def l2_cache_size(self) -> int:
        return 94 * 1024 * 1024  # 94 MB

    @property
    def shared_mem_per_sm(self) -> int:
        return 256 * 1024

    @property
    def max_registers_per_sm(self) -> int:
        return 256 * 1024

    @property
    def warp_size(self) -> int:
        return 32

    @property
    def tensor_core_specs(self):
        return {"wmma": {"fmha": True, "max_tflops_fp16": 3090.0}}


@dataclass(frozen=True)
class NVIDIA_A100(GPUArchitecture):
    name: str = "NVIDIA A100"

    @property
    def sm_count(self) -> int:
        return 108

    @property
    def clock_ghz(self) -> float:
        return 1.41

    @property
    def memory_bw_gbps(self) -> float:
        return 1555.0

    @property
    def l2_cache_size(self) -> int:
        return 40 * 1024 * 1024

    @property
    def shared_mem_per_sm(self) -> int:
        return 160 * 1024

    @property
    def max_registers_per_sm(self) -> int:
        return 256 * 1024

    @property
    def warp_size(self) -> int:
        return 32

    @property
    def tensor_core_specs(self):
        return {"wmma": {"max_tflops_fp16": 3120.0}}


@dataclass(frozen=True)
class NVIDIA_B200(GPUArchitecture):
    name: str = "NVIDIA B200"

    @property
    def sm_count(self) -> int:
        return 90

    @property
    def clock_ghz(self) -> float:
        return 1.5

    @property
    def memory_bw_gbps(self) -> float:
        return 2000.0

    @property
    def l2_cache_size(self) -> int:
        return 48 * 1024 * 1024

    @property
    def shared_mem_per_sm(self) -> int:
        return 192 * 1024

    @property
    def max_registers_per_sm(self) -> int:
        return 256 * 1024

    @property
    def warp_size(self) -> int:
        return 32

    @property
    def tensor_core_specs(self):
        return {"wmma": {"max_tflops_fp16": 2200.0}}
