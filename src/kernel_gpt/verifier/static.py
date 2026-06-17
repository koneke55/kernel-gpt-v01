from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple


class Severity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class Verdict:
    passed: bool
    issues: List[str]
    severity: Severity = Severity.INFO


class StaticVerifier:
    """Static verifier that performs basic checks on generated kernel code and specs."""

    def check_shape_compatibility(self, spec) -> Verdict:
        ok, issues = spec.validate()
        return Verdict(passed=ok, issues=issues, severity=(Severity.ERROR if not ok else Severity.INFO))

    def check_memory_budget(self, kernel_code: str, spec, constraints) -> Verdict:
        # Very basic heuristic: count occurrences of shared memory markers
        issues = []
        shared_bytes = kernel_code.count("__shared__") * 1024  # crude
        max_sh = constraints.get("max_shared_mem", 48 * 1024)
        if shared_bytes > max_sh:
            issues.append(f"Estimated shared memory {shared_bytes} > {max_sh}")
        return Verdict(passed=(len(issues) == 0), issues=issues, severity=(Severity.ERROR if issues else Severity.INFO))

    def check_race_conditions(self, kernel_code: str) -> Verdict:
        # static text checks for missing __syncthreads
        issues = []
        if "__syncthreads()" not in kernel_code and "atomicAdd" in kernel_code:
            issues.append("Potential race: atomicAdd present without barrier hints")
        return Verdict(passed=(len(issues) == 0), issues=issues, severity=(Severity.WARNING if issues else Severity.INFO))
