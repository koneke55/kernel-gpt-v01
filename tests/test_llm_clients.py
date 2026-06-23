from __future__ import annotations
import types

from kernel_gpt.synthesis_engine import llm_clients
from kernel_gpt.synthesis_engine.strategy import Strategy


class DummyInput:
    def __init__(self, dtype_name="fp32"):
        self.dtype = types.SimpleNamespace(name=dtype_name)


class DummySpec:
    def __init__(self):
        self.name = "dummy"
        self.inputs = [DummyInput()]
        self.ops = []


class DummyGPU:
    def __init__(self):
        self.name = "H100"


async def _fake_openai_ok(prompt: str) -> str:
    return '{"block_size":256,"tile_sizes":{"M":128,"N":128,"K":32},"use_shared_mem":true,"use_tensor_cores":false,"num_stages":2,"thread_layout":"bxk","optimization_notes":"ok"}'


async def _fake_anthropic_none(prompt: str):
    return None


def test_generate_strategy_opens_ok(monkeypatch):
    # Patch async provider functions
    monkeypatch.setattr(llm_clients, "_call_openai", _fake_openai_ok)
    monkeypatch.setattr(llm_clients, "_call_anthropic", _fake_anthropic_none)

    spec = DummySpec()
    gpu = DummyGPU()
    strat = llm_clients.generate_strategy(spec, gpu)
    assert isinstance(strat, Strategy)
    assert strat.block_size == 256
    assert strat.tile_sizes.get("M") == 128
