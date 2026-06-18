kernel-gpt: Hardware-aware GPU Kernel Synthesis
===============================================

Overview
--------

kernel-gpt is a research prototype for hardware-aware GPU kernel synthesis. An LLM proposes high-level strategies and a deterministic generator emits CUDA/Triton kernels; static verifiers check resource budgets before execution. See [docs/architecture.md](docs/architecture.md#L1) for design details.

Requirements
------------

- Python 3.10+
- A CUDA-enabled system for running generated kernels (for full benchmarking)
- PyTorch with a CUDA build if you intend to run GPU benchmarks

Install
-------

Prefer a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

If you only need to read the code or run non-GPU parts, a CPU-only PyTorch is fine.

Quickstart / Examples
---------------------

Run the end-to-end matmul example:

```bash
python examples/matmul_example.py
```

Or use the package entrypoint (shows CLI help):

```bash
python -m kernel_gpt
```

Project layout
--------------

- `src/` — core package `kernel_gpt`
- `examples/` — runnable demos (matmul, attention, comparisons)
- `models/` — small reference models/strategies
- `docs/` — design and architecture docs
- `tests/` — unit and integration tests (pytest)

Running tests
-------------

Run the test suite with:

```bash
pytest -q
```

Note: some tests may require a GPU-enabled PyTorch and will be skipped on CPU-only environments.

Contributing
------------

Contributions and bug reports are welcome. Suggested starter tasks:

- Improve docs in `docs/`
- Add ROCm backend tests for cross-vendor portability
- Add more microbenchmarks in `examples/`

When opening a PR, include a short description, the test strategy, and any hardware/driver versions used.

Roadmap / Research TODOs
------------------------
- Cross-vendor portability (CUDA <-> ROCm)
- Sparse kernel support and block-sparse tiling
- Dynamic shape handling and JIT-backed kernels
- LLM fine-tuning on CUDA/Triton kernel corpora

License & Contact
-----------------

This repository is released under the terms of the included `LICENSE` file. For questions, contact the author in `pyproject.toml`.


