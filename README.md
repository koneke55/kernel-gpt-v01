kernel-gpt: Hardware-aware GPU Kernel Synthesis
===============================================

Quickstart
---------

Install dependencies (prefer a virtualenv):

```bash
pip install -e .
```

Run the CLI help:

```bash
python -m kernel_gpt
```
 
Design
------

kernel-gpt follows the "LLM-as-Architect + Hardware-Guided Synthesis" philosophy: an LLM proposes high-level strategies and a deterministic generator emits CUDA/Triton kernels. Static verifiers check budgets before execution. See docs/architecture.md for details.

TODOs and research extensions
- Cross-vendor portability (CUDA <-> ROCm)
- Sparse kernel support and block-sparse tiling
- Dynamic shape handling and JIT-backed kernels
- LLM fine-tuning on CUDA/Triton kernel corpora

