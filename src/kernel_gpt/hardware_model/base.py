from __future__ import annotations
import abc
from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class GPUArchitecture(abc.ABC):
    """Abstract GPU architecture description.

    Args:
        name: Human-readable GPU name.
    """
    name: str

    @property
    @abc.abstractmethod
    def sm_count(self) -> int:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def clock_ghz(self) -> float:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def memory_bw_gbps(self) -> float:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def l2_cache_size(self) -> int:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def shared_mem_per_sm(self) -> int:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def max_registers_per_sm(self) -> int:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def warp_size(self) -> int:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def tensor_core_specs(self) -> Dict[str, Any]:
        raise NotImplementedError
