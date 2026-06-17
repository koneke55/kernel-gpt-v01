from kernel_gpt.verifier.static import StaticVerifier
from kernel_gpt.synthesis_engine.spec import TensorSpec, KernelSpec, ComputePattern, DType


def test_shape_check():
    a = TensorSpec("A", (16, 32), DType.FP32)
    b = TensorSpec("B", (31, 8), DType.FP32)
    ks = KernelSpec("matmul_bad", (a, b), (a,), ComputePattern("matmul"))
    verifier = StaticVerifier()
    v = verifier.check_shape_compatibility(ks)
    assert not v.passed
