from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import os
from jinja2 import Environment, FileSystemLoader


@dataclass
class CodeGenerator:
    templates_dir: str

    def _env(self):
        return Environment(loader=FileSystemLoader(self.templates_dir), trim_blocks=True, lstrip_blocks=True)


class CUDAGenerator(CodeGenerator):
    """Generate CUDA code from templates using Jinja2."""

    def generate(self, spec, strategy) -> str:
        env = self._env()
        tpl = env.get_template("matmul.cuh.jinja")
        return tpl.render(spec=spec, strategy=strategy)


class TritonGenerator(CodeGenerator):
    def generate(self, spec, strategy) -> str:
        env = self._env()
        tpl = env.get_template("matmul.py.jinja")
        return tpl.render(spec=spec, strategy=strategy)
