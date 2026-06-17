from __future__ import annotations
from typing import Dict, Any
import json


class Reporter:
    """Generate simple markdown and JSON reports for benchmarks."""

    @staticmethod
    def to_markdown(name: str, results: Dict[str, Any]) -> str:
        md = f"# Benchmark report: {name}\n\n"
        for k, v in results.items():
            md += f"- **{k}**: {v}\n"
        return md

    @staticmethod
    def to_json(results: Dict[str, Any], path: str):
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
