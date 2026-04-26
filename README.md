# PolyKV

**Shared asymmetrically-compressed KV cache pool for multi-agent LLM inference.**

Instead of each agent holding its own full-precision KV cache of a shared document, PolyKV compresses it once and shares it across all agents — achieving **O(1) memory complexity** in agent count for document context.

[![PyPI](https://img.shields.io/pypi/v/polykv-llm)](https://pypi.org/project/polykv-llm/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19686730.svg)](https://doi.org/10.5281/zenodo.19686730)

---

## Why PolyKV

In multi-agent inference, N agents reading the same document naively require N full-precision KV caches. PolyKV writes the cache **once**, compressed, and injects it into each agent at inference time.

| Configuration | KV Cache Memory |
|---|---|
| 3 agents, no sharing | 1.011 GB |
| 3 agents with PolyKV | 0.116 GB |
| 15 agents, no sharing | 19.798 GB |
| 15 agents with PolyKV | 0.454 GB |

- **2.91× compression ratio** — stable across all model sizes and context lengths
- **+1.59% perplexity degradation** at 2K tokens, improving to **+0.57%** at 4K tokens
- **Mean BERTScore F1: 0.957–0.970** across agent counts
- Validated on Llama-3-8B-Instruct (32 layers, GQA) and SmolLM2-1.7B-Instruct
- Works with any HuggingFace model on CUDA, MPS, or CPU

---

## Installation

```bash
pip install polykv-llm
```

**Requirements:** Python ≥ 3.9, PyTorch ≥ 2.1, Transformers ≥ 4.40

---

## Quickstart

```python
import polykv
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

document = "Your shared context document..."
queries = ["Agent A query", "Agent B query", "Agent C query"]

# Build the shared compressed pool — runs once
pool = polykv.compress(model, tokenizer, document)

# Spin up N agents sharing the same pool
agents = polykv.create_agents(pool, model, tokenizer, n=3)

# Each agent generates independently
responses = [agent.generate(query) for agent, query in zip(agents, queries)]

print(pool.memory_summary(n_agents=3))
```

---

## How It Works

Compression is **asymmetric** — Keys and Values have different sensitivity profiles:

- **Keys → int8 (q8 0):** Key errors amplify exponentially through softmax, so they receive higher precision.
- **Values → 3-bit TurboQuant MSE:** FWHT rotation spreads outlier energy across dimensions, then Lloyd-Max quantization maps each coordinate to the nearest centroid. Value errors scale linearly and are safe to compress aggressively.

At inference time, each `PooledAgent` decompresses the KV tensors layer-by-layer, moves them to the correct device, and injects them into a fresh `DynamicCache` — no contention between agents, no per-agent copy of the compressed data.

---

## API Reference

### `polykv.compress(model, tokenizer, document) → SharedKVPool`

Prefills the model on `document`, extracts the KV cache, compresses it asymmetrically, and offloads to CPU. Call this once per document.

### `polykv.create_agents(pool, model, tokenizer, n=1) → List[PooledAgent]`

Creates `n` agents sharing the same pool. Each agent is independent — they can generate concurrently.

### `agent.generate(query, max_tokens=200) → str`

Generates a response using the shared compressed KV pool as document context.

### `pool.memory_summary(n_agents) → str`

Prints a human-readable breakdown of memory savings for `n_agents`.

---

## Reproduce Results

```bash
git clone https://github.com/ishan1410/PolyKV.git
cd PolyKV
pip install -r requirements.txt
python experiment.py
```

Full results across all test configurations: see [RESULTS.md](RESULTS.md).

---

## Citation

```bibtex
@software{polykv2026,
  author  = {Patel, Ishan and Joshi, Ishan},
  title   = {PolyKV: A Shared Asymmetrically-Compressed KV Cache Pool for Multi-Agent LLM Inference},
  year    = {2026},
  doi     = {10.5281/zenodo.19686730},
  url     = {https://github.com/ishan1410/PolyKV}
}
```

---

## License

MIT
