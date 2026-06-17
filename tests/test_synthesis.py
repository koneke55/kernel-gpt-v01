from kernel_gpt.synthesis_engine.spec import TensorSpec, KernelSpec, ComputePattern, DType
from kernel_gpt.synthesis_engine.strategy import HeuristicStrategySelector


def test_spec_validation_matmul():
    a = TensorSpec("A", (16, 32), DType.FP32)
    b = TensorSpec("B", (32, 8), DType.FP32)
    out = TensorSpec("C", (16, 8), DType.FP32)
    ks = KernelSpec("matmul_test", (a, b), (out,), ComputePattern("matmul"))
    ok, issues = ks.validate()
    assert ok


def test_heuristic_selector():
    a = TensorSpec("A", (16, 32), DType.FP16)
    b = TensorSpec("B", (32, 8), DType.FP16)
    ks = KernelSpec("matmul_test", (a, b), (a,), ComputePattern("matmul"))
    strat = HeuristicStrategySelector.select(ks, None)
    assert strat.block_size > 0
