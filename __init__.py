"""
PolyKV — Shared Asymmetrically-Compressed KV Cache Pool for Multi-Agent LLM Inference.

GitHub: https://github.com/ishan1410/PolyKV
DOI:    https://doi.org/10.5281/zenodo.19686730

Quick start:
    import polykv

    pool = polykv.compress(model, tokenizer, document)
    agents = polykv.create_agents(pool, model, tokenizer, n=5)
    responses = [a.generate(query) for a, query in zip(agents, queries)]
    print(pool.memory_summary(n_agents=5))
"""

from .pool import SharedKVPool
from .agents import PooledAgent

__version__ = "0.1.0"
__author__ = "Ishan Patel"
__doi__ = "10.5281/zenodo.19686730"

__all__ = [
    "SharedKVPool",
    "PooledAgent",
    "compress",
    "create_agents",
    "__version__",
]


def compress(model, tokenizer, document: str) -> SharedKVPool:
    """
    Build a SharedKVPool from a document and a loaded model.

    Args:
        model:      Any HuggingFace model (any arch, CUDA/MPS/CPU).
        tokenizer:  Corresponding tokenizer.
        document:   The shared context document (string).

    Returns:
        SharedKVPool — ready to inject into agents.

    Example:
        pool = polykv.compress(model, tokenizer, my_document)
    """
    return SharedKVPool.from_context(model, tokenizer, document)


def create_agents(pool: SharedKVPool, model, tokenizer, n: int = 1) -> list:
    """
    Create N PooledAgents sharing a single pool.

    Args:
        pool:       SharedKVPool built with polykv.compress().
        model:      Same model used to build the pool.
        tokenizer:  Corresponding tokenizer.
        n:          Number of agents to create.

    Returns:
        List of PooledAgent instances.

    Example:
        agents = polykv.create_agents(pool, model, tokenizer, n=10)
        responses = [a.generate(q) for a, q in zip(agents, queries)]
    """
    return [
        PooledAgent(f"agent_{i}", pool, model, tokenizer)
        for i in range(n)
    ]
