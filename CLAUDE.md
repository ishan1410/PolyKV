# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

PolyKV is a research library implementing **shared asymmetric KV cache compression for multi-agent LLM inference**. The core idea: instead of N agents each holding their own full-precision KV cache of a shared document, they all read from a single compressed pool — achieving O(1) memory complexity in agent count for document context.

DOI: [10.5281/zenodo.19686730](https://doi.org/10.5281/zenodo.19686730)

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Install as package (editable)
pip install -e .

# Run the Phase 0 reproduction experiment (requires Llama-3-8B-Instruct access + GPU)
python experiment.py

# Test TurboQuant standalone (no GPU required)
python -m polykv.turboquant
```

There is no test suite or linter configured. `pyproject.toml` defines `pytest`, `black`, and `isort` as optional dev dependencies but they have no config and no tests exist yet.

## Architecture

### Data Flow

```
document
   │
   ▼
SharedKVPool.from_context(model, tokenizer, document)
   │  prefills model, extracts past_key_values, offloads to CPU
   │
   ▼
pool.encode(raw_kv_cache)
   │  AsymmetricKVCompressor:
   │    K → int8 linear quantization (q8_0, per-tensor scale)
   │    V → TurboQuantMSE 3-bit (FWHT rotation + Lloyd-Max centroids)
   │
   ▼
SharedKVPool (blocks: Dict[layer_idx → CompressedKVBlock])
   │
   ├─► PooledAgent("agent_0", pool, model, tokenizer)
   ├─► PooledAgent("agent_1", pool, model, tokenizer)
   └─► PooledAgent("agent_N", pool, model, tokenizer)
          │
          ▼
       agent.generate(query)
          │  pool.get_kv_for_layer(i) → decompress K, V per layer
          │  injects into DynamicCache → model.generate(past_key_values=cache)
```

### Key Files

- **`polykv/pool.py`** — `SharedKVPool` and `CompressedKVBlock`. Central data structure. `from_context()` is the recommended entry point; `encode()` / `get_kv_for_layer()` are the read/write API.
- **`polykv/compress.py`** — `AsymmetricKVCompressor`. Keys use int8 linear quant; Values use TurboQuant MSE. The asymmetry is intentional: K errors amplify exponentially through softmax, V errors scale linearly.
- **`polykv/turboquant.py`** — `TurboQuantMSE`. FWHT rotation spreads outlier energy across dimensions before Lloyd-Max 3-bit quantization. `dim` must be a power of 2. Inverse uses unnormalized FWHT divided by `d` (exploiting H·H = d·I).
- **`polykv/agents.py`** — `PooledAgent`. Thin wrapper: decodes KV layer-by-layer from the pool, moves tensors to the correct layer device, injects into `DynamicCache`, calls `model.generate`.
- **`polykv/backends/_arch.py`** — Architecture-agnostic helpers (`get_num_layers`, `get_layer_device`, `get_first_device`). Supports Llama, Mistral, Qwen2, Gemma, Phi-3, Falcon, GPT-2/Neo/J/NeoX, OPT, BLOOM, Phi-2 via a try-each-pattern approach.
- **`experiment.py`** — Full validation script. Uses raw imports (`from pool import SharedKVPool`) so must be run from the repo root, not as a module.

### High-Level API (`polykv/__init__.py`)

```python
pool   = polykv.compress(model, tokenizer, document)
agents = polykv.create_agents(pool, model, tokenizer, n=5)
responses = [a.generate(q) for a, q in zip(agents, queries)]
print(pool.memory_summary(n_agents=5))
```

### Compression Details

- **K (keys):** per-tensor int8 (`scale = abs_max / 127`), stored as `torch.int8` + float32 scale
- **V (values):** 3-bit TurboQuant MSE — normalize to unit sphere → FWHT rotate → map to nearest Lloyd-Max centroid from `[-2.152, -1.344, -0.756, -0.245, 0.245, 0.756, 1.344, 2.152]`. Stored as `torch.int8` indices + float32 norms.
- **Compression ratio:** Stable at **2.91x** across all tested model sizes and context lengths.
- **Quality:** Mean BERTScore F1 ~0.957 vs full-precision baseline; PPL delta +1.59% at short context, improving to +0.57% at long context.

### Device Handling

The pool is always stored on CPU. `PooledAgent.generate()` moves each layer's K/V to the device that layer lives on (via `get_layer_device`), which matters for multi-GPU `device_map="auto"` setups. This is done inside `DynamicCache.update()` just before inference.

## Validated Configurations

Primary validation: `meta-llama/Meta-Llama-3-8B-Instruct` (4-bit NF4, bfloat16 KV) on Kaggle T4 x2. Also validated on `HuggingFaceTB/SmolLM2-1.7B-Instruct`. The `experiment.py` script hard-codes `device_map="auto"` with `max_memory={0: "13GiB", 1: "13GiB"}` and uses `BitsAndBytesConfig` for 4-bit loading — this requires a CUDA environment with bitsandbytes installed (not in `requirements.txt`).
