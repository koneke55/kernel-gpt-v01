from __future__ import annotations
from typing import Any
from ..synthesis_engine.llm_interface import LLMArchitect
from ..synthesis_engine.generator import CUDAGenerator, TritonGenerator
from ..verifier.static import StaticVerifier


def synthesize(spec, gpu_name: str = "H100", backend: str = "cuda") -> str:
    """Synthesize a kernel for the given spec and backend.

    Args:
        spec: KernelSpec or path to spec module
        gpu_name: target GPU name
        backend: 'cuda' or 'triton'

    Returns:
        generated code string
    """
    architect = LLMArchitect()
    strategy = architect.design_strategy(spec, gpu_name)
    templates_dir = "templates/cuda" if backend == "cuda" else "templates/triton"
    gen = CUDAGenerator(templates_dir=templates_dir) if backend == "cuda" else TritonGenerator(templates_dir=templates_dir)
    return gen.generate(spec, strategy)


def verify(kernel_code: str, spec) -> Any:
    v = StaticVerifier()
    return v.check_race_conditions(kernel_code)


def benchmark(kernel_code: str, spec) -> Any:
    # simplified wrapper for examples
    from ..benchmark.runner import BenchmarkRunner
    runner = BenchmarkRunner()
    module = runner.compile(kernel_code, name=spec.name)
    return runner.run(module, (), ())
