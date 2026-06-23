from __future__ import annotations
import os
import asyncio
import json
from typing import Optional
import httpx

from .strategy import Strategy


async def _post_with_retries(url: str, headers: dict, payload: dict, retries: int = 3, backoff: float = 0.8) -> Optional[dict]:
    for attempt in range(1, retries + 1):
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                r = await client.post(url, headers=headers, json=payload)
            if r.status_code >= 200 and r.status_code < 300:
                return r.json()
            # non-2xx: treat as transient
            await asyncio.sleep(backoff * attempt)
        except Exception:
            await asyncio.sleep(backoff * attempt)
    return None


async def _call_openai(prompt: str) -> Optional[str]:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return None
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    payload = {
        "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 800,
    }
    resp = await _post_with_retries(url, headers, payload)
    if not resp:
        return None
    # extract text
    try:
        return resp["choices"][0]["message"]["content"]
    except Exception:
        return None


async def _call_anthropic(prompt: str) -> Optional[str]:
    key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY") or os.getenv("ANTHROPIC_KEY")
    if not key:
        return None
    url = "https://api.anthropic.com/v1/messages"
    headers = {"Content-Type": "application/json", "x-api-key": key}
    payload = {
        "model": os.getenv("ANTHROPIC_MODEL", "claude-2.1"),
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 800,
    }
    resp = await _post_with_retries(url, headers, payload)
    if not resp:
        return None
    try:
        # Anthropic may return different shapes; try common ones
        if isinstance(resp.get("content"), list):
            return resp["content"][0].get("text")
        if isinstance(resp.get("completion"), str):
            return resp.get("completion")
        if isinstance(resp.get("choices"), list):
            return resp["choices"][0].get("message", {}).get("content")
    except Exception:
        return None
    return None


def _make_prompt_for_strategy(spec, gpu) -> str:
    # Ask LLM to return a compact JSON matching Strategy fields
    return (
        "You are a hardware-aware GPU kernel synthesis assistant. "
        "Given the following kernel spec and GPU description, output a JSON object with the exact keys: "
        "block_size (int), tile_sizes (object of strings->ints), use_shared_mem (bool), use_tensor_cores (bool), num_stages (int, optional), thread_layout (string, optional), optimization_notes (string, optional).\n\n"
        f"Spec name: {getattr(spec, 'name', str(spec))}\n"
        f"Spec inputs: {getattr(spec, 'inputs', [])}\n"
        f"Spec ops: {getattr(spec, 'ops', [])}\n"
        f"GPU: {getattr(gpu, 'name', str(gpu))}\n"
        "Return only valid JSON, no explanation."
    )


async def generate_strategy_async(spec, gpu) -> Optional[Strategy]:
    prompt = _make_prompt_for_strategy(spec, gpu)

    # Try OpenAI and Anthropic in parallel (pipelining across providers)
    tasks = [asyncio.create_task(_call_openai(prompt)), asyncio.create_task(_call_anthropic(prompt))]
    done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    text = None
    for t in done:
        text = t.result()
        if text:
            break

    # cancel remaining
    for t in tasks:
        if not t.done():
            t.cancel()

    if not text:
        return None

    # parse JSON from LLM output
    try:
        # Some LLMs may include code fences, strip common wrappers
        cleaned = text.strip()
        if cleaned.startswith('```'):
            # remove fence
            parts = cleaned.split('\n')
            if parts[0].startswith('```'):
                cleaned = '\n'.join(parts[1:])
                if cleaned.endswith('```'):
                    cleaned = '\n'.join(cleaned.split('\n')[:-1])
        obj = json.loads(cleaned)
        # construct Strategy
        return Strategy(
            block_size=int(obj.get('block_size')),
            tile_sizes={k: int(v) for k, v in obj.get('tile_sizes', {}).items()},
            use_shared_mem=bool(obj.get('use_shared_mem', True)),
            use_tensor_cores=bool(obj.get('use_tensor_cores', False)),
            num_stages=int(obj.get('num_stages', 2)),
            thread_layout=str(obj.get('thread_layout', 'bxk')),
            optimization_notes=str(obj.get('optimization_notes', '')),
        )
    except Exception:
        return None


def generate_strategy(spec, gpu) -> Optional[Strategy]:
    try:
        return asyncio.run(generate_strategy_async(spec, gpu))
    except Exception:
        return None
