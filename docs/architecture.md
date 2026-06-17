# kernel-gpt Architecture

kernel-gpt follows a two-stage pipeline:

1. LLM-as-Architect: LLM proposes a Strategy (tiling, parallelization, memory use)
2. Deterministic Generator: uses templates to emit exact CUDA/Triton code

Static verification prevents running obviously invalid kernels. Benchmarking measures performance and reports.

TODO: flesh out diagrams and cross-vendor notes.
