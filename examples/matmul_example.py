"""End-to-end example: spec -> strategy -> kernel -> verify -> benchmark -> report"""
from kernel_gpt.synthesis_engine.spec import TensorSpec, KernelSpec, ComputePattern, DType
from kernel_gpt.interface.api import synthesize, verify, benchmark
from kernel_gpt.hardware_model.nvidia import NVIDIA_H100


def main():
    A = TensorSpec("A", (64, 128), DType.FP32)
    B = TensorSpec("B", (128, 64), DType.FP32)
    C = TensorSpec("C", (64, 64), DType.FP32)
    ks = KernelSpec("matmul_example", (A, B), (C,), ComputePattern("matmul"))
    gpu = NVIDIA_H100()
    code = synthesize(ks, gpu_name=gpu.name, backend="cuda")
    print("Generated code length:", len(code))
    v = verify(code, ks)
    print("Verify:", v)
    res = benchmark(code, ks)
    print("Benchmark result:", res)


if __name__ == "__main__":
    main()
