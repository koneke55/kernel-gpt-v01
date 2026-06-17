from kernel_gpt.hardware_model.nvidia import NVIDIA_H100
from kernel_gpt.hardware_model.roofline import RooflineModel


def test_h100_properties():
    gpu = NVIDIA_H100()
    assert gpu.warp_size == 32
    assert gpu.sm_count > 0


def test_roofline_classify():
    r = RooflineModel({"FP32": 1000.0})
    assert r.classify_workload(0.1) == "memory_bound"
    assert r.classify_workload(20.0) == "compute_bound"
