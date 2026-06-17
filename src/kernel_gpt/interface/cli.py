from __future__ import annotations
import argparse
from ..synthesis_engine.spec import KernelSpec
from ..synthesis_engine.llm_interface import LLMArchitect
from ..synthesis_engine.generator import CUDAGenerator
from ..hardware_model.nvidia import NVIDIA_H100
from ..verifier.static import StaticVerifier


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kernel-gpt")
    sub = parser.add_subparsers(dest="cmd")

    p_syn = sub.add_parser("synthesize")
    p_syn.add_argument("--spec", required=True)
    p_syn.add_argument("--gpu", default="H100")
    p_syn.set_defaults(func=cmd_synthesize)

    p_ver = sub.add_parser("verify")
    p_ver.add_argument("--kernel", required=True)
    p_ver.set_defaults(func=cmd_verify)

    p_bench = sub.add_parser("benchmark")
    p_bench.add_argument("--kernel", required=True)
    p_bench.set_defaults(func=cmd_benchmark)

    return parser


def cmd_synthesize(args):
    print("Synthesize: loading spec from", args.spec)
    # For simplicity we import the spec file as a module
    import importlib.util
    spec_path = args.spec
    spec_mod = importlib.util.spec_from_file_location("user_spec", spec_path)
    mod = importlib.util.module_from_spec(spec_mod)
    spec_mod.loader.exec_module(mod)  # type: ignore
    ks: KernelSpec = getattr(mod, "kernel_spec")
    gpu = NVIDIA_H100()
    architect = LLMArchitect()
    strategy = architect.design_strategy(ks, gpu)
    gen = CUDAGenerator(templates_dir="templates/cuda")
    code = gen.generate(ks, strategy)
    out_path = ks.name + ".cu"
    with open(out_path, "w") as f:
        f.write(code)
    print("Wrote", out_path)


def cmd_verify(args):
    verifier = StaticVerifier()
    with open(args.kernel) as f:
        code = f.read()
    # dummy spec loader
    print(verifier.check_race_conditions(code))


def cmd_benchmark(args):
    print("Benchmarking is not fully implemented in CLI")
