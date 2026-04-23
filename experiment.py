import torch
import numpy as np
import math
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.cache_utils import DynamicCache
from pool import SharedKVPool
from agents import PooledAgent
from bert_score import score as bert_score

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"

from datasets import load_dataset
wikitext = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
DOCUMENT = " ".join(wikitext["text"])[:8000]

AGENT_QUERIES = [
    "Agent A: What specific names, places, or dates are mentioned in this text?",
    "Agent B: What is the main subject described in the opening of this passage?",
    "Agent C: Describe any events or processes explained in this text.",
]

def run_experiment():
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        max_memory={0: "13GiB", 1: "13GiB"},
    )
    model.eval()
    primary_device = next(model.parameters()).device  # reliable with device_map="auto"

    for i in range(torch.cuda.device_count()):
        used = torch.cuda.memory_allocated(i) / 1e9
        print(f"  GPU {i} used: {used:.2f} GB")

    print("\n── PRE-RUN DOCUMENT INFO ──")
    tokenized_doc = tokenizer(DOCUMENT, return_tensors="pt")
    doc_token_count = tokenized_doc.input_ids.shape[1]
    print(f"Document token count: {doc_token_count}")

    # ── STEP 1: Prefill ──
    print("Prefilling document context...")
    doc_tokens = tokenizer.encode(DOCUMENT, return_tensors="pt")

    split_idx = int(doc_tokens.shape[1] * 0.7)
    context_tokens = doc_tokens[:, :split_idx].to(primary_device)
    target_tokens  = doc_tokens[:, split_idx:].to(primary_device)

    with torch.no_grad():
        prefill_out = model(context_tokens, use_cache=True)
        raw_cache = prefill_out.past_key_values

    # ── DynamicCache extraction (transformers version safe) ──
    if hasattr(raw_cache, "key_cache"):
        raw_kv_cache = [
            (raw_cache.key_cache[i], raw_cache.value_cache[i])
            for i in range(len(raw_cache.key_cache))
        ]
    else:
        raw_kv_cache = [(layer[0], layer[1]) for layer in raw_cache]

    del prefill_out
    torch.cuda.empty_cache()

    kv_layers = len(raw_kv_cache)
    k_shape   = raw_kv_cache[0][0].shape
    print(f"KV cache: {kv_layers} layers, K shape: {k_shape}")

    # ── STEP 2: Build SharedKVPool ──
    print("Building SharedKVPool...")
    head_dim = raw_kv_cache[0][0].shape[-1]
    
    raw_kv_cache = [
        (k.cpu().float(), v.cpu().float())
        for k, v in raw_kv_cache
    ]
    torch.cuda.empty_cache()

    pool = SharedKVPool(head_dim=head_dim)
    pool.encode(raw_kv_cache)

    ratio      = pool.get_compression_ratio()
    n_agents   = 3
    full_kv_gb = sum(k.element_size() * k.nelement() + v.element_size() * v.nelement()
                     for k, v in raw_kv_cache) / 1e9
    comp_kv_gb = full_kv_gb / ratio
    saved_gb   = full_kv_gb * n_agents - comp_kv_gb
    saved_pct  = saved_gb / (full_kv_gb * n_agents) * 100

    print(f"Pool encoded: {kv_layers} layers, {ratio:.2f}x compression")
    print(f"Compression ratio: {ratio:.2f}x")
    print(f"\n── KV CACHE MEMORY ──")
    print(f"Full precision KV (1 agent):     {full_kv_gb:.3f} GB")
    print(f"Compressed pool (1 shared):      {comp_kv_gb:.3f} GB")
    print(f"Compression ratio:               {ratio:.2f}x")
    print(f"{n_agents} agents WITHOUT sharing:        {full_kv_gb * n_agents:.3f} GB")
    print(f"{n_agents} agents WITH PolyKV pool:       {comp_kv_gb:.3f} GB")
    print(f"Total memory saved ({n_agents} agents):   {saved_gb:.3f} GB  ({saved_pct:.1f}% reduction)")

    # ── STEP 3: Run agents from SHARED COMPRESSED pool ──
    print("\nRunning agents from SHARED COMPRESSED pool...")
    pooled_agents = [
        PooledAgent(f"agent_{i}", pool, model, tokenizer)
        for i in range(n_agents)
    ]
    pooled_outputs = []
    for agent, query in zip(pooled_agents, AGENT_QUERIES):
        print(f"\n{agent.agent_id} querying...")
        response = agent.generate(query)
        pooled_outputs.append(response)
        print(f"Response: {response[:200]}...")

    # ── STEP 4: Baseline (full precision KV, moved to GPU on demand) ──
    print("\nRunning agents with FULL PRECISION KV (baseline)...")
    baseline_outputs = []
    for query in AGENT_QUERIES:
        input_ids = tokenizer.encode(query, return_tensors="pt").to(primary_device)
        with torch.no_grad():
            cache_clone = DynamicCache()
            for i, (k, v) in enumerate(raw_kv_cache):
                layer_device = model.model.layers[i].self_attn.q_proj.weight.device
                cache_clone.update(
                    k.to(torch.bfloat16).to(layer_device),
                    v.to(torch.bfloat16).to(layer_device),
                    i
                )
                
            seq_len = cache_clone.get_seq_length()
            attention_mask = torch.ones(
                1, seq_len + input_ids.shape[1],
                dtype=torch.long, device=primary_device
            )
            cache_position = torch.arange(
                seq_len, seq_len + input_ids.shape[1],
                device=primary_device
            )
            out = model.generate(
                input_ids,
                attention_mask=attention_mask,
                cache_position=cache_position,
                past_key_values=cache_clone,
                max_new_tokens=200,
                do_sample=False,
            )
        response = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
        baseline_outputs.append(response)
        print(f"Baseline response: {response[:200]}...")
        del cache_clone, out  # clear before next iteration
        torch.cuda.empty_cache()

    # ── STEP 5: Perplexity ──
    print("\nMeasuring Perplexity...")
    with torch.no_grad():
        # Baseline PPL
        cache_clone_ppl = DynamicCache()
        for i, (k, v) in enumerate(raw_kv_cache):
            layer_device = model.model.layers[i].self_attn.q_proj.weight.device
            cache_clone_ppl.update(
                k.to(torch.bfloat16).to(layer_device),
                v.to(torch.bfloat16).to(layer_device),
                i
            )
            
        seq_len_ppl = cache_clone_ppl.get_seq_length()
        attn_mask_ppl = torch.ones(
            1, seq_len_ppl + target_tokens.shape[1],
            dtype=torch.long, device=primary_device
        )
        cache_pos_ppl = torch.arange(
            seq_len_ppl, seq_len_ppl + target_tokens.shape[1],
            device=primary_device
        )
        baseline_out = model(
            input_ids=target_tokens,
            labels=target_tokens,
            past_key_values=cache_clone_ppl,
            attention_mask=attn_mask_ppl,
            cache_position=cache_pos_ppl,
        )
        baseline_ppl = math.exp(baseline_out.loss.item())
        del cache_clone_ppl, baseline_out  # clear before loading compressed
        torch.cuda.empty_cache()

        # Compressed PPL — correct dtype AND device
        cache_comp = DynamicCache()
        for layer_idx in range(len(model.model.layers)):
            layer_device = model.model.layers[layer_idx].self_attn.q_proj.weight.device
            k, v = pool.get_kv_for_layer(layer_idx)
            cache_comp.update(
                k.to(model.dtype).to(layer_device),
                v.to(model.dtype).to(layer_device),
                layer_idx
            )

        comp_out = model(
            input_ids=target_tokens,
            labels=target_tokens,
            past_key_values=cache_comp,
            attention_mask=attn_mask_ppl,
            cache_position=cache_pos_ppl,
        )
        compressed_ppl = math.exp(comp_out.loss.item())
        ppl_delta = (compressed_ppl - baseline_ppl) / baseline_ppl * 100
        del cache_comp, comp_out  # clean up after compressed PPL
        torch.cuda.empty_cache()

    # ── STEP 6: Free model, run BERTScore ──
    print("\nFreeing model memory for BERTScore...")
    del model
    del pooled_agents
    del raw_kv_cache
    del pool
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        print(f"GPU {i}: {free/1e9:.2f} GB free after cleanup")

    print("Measuring BERTScore...")
    P, R, F1 = bert_score(
        cands=pooled_outputs,
        refs=baseline_outputs,
        lang="en",
        model_type="roberta-large",
        device="cuda:0",
        verbose=False,
    )

    # ── RESULTS ──
    print(f"\n── RESULTS ──")
    print(f"Model: {MODEL_NAME}")
    print(f"Compression ratio: {ratio:.2f}x memory saved")
    print(f"Memory: {n_agents} agents share 1 pool vs {n_agents} full-precision copies")
    print(f"Baseline PPL: {baseline_ppl:.3f} | Compressed PPL: {compressed_ppl:.3f} | Delta: {ppl_delta:.2f}%")
    print(f"\n── BERTSCORE (semantic quality) ──")
    for i, (p, r, f1) in enumerate(zip(P, R, F1)):
        status = "✓ Good" if f1.item() >= 0.92 else "✗ Degraded"
        print(f"Agent {i}: BERTScore P={p.item():.4f} R={r.item():.4f} F1={f1.item():.4f}  ({status})")
    print(f"Mean BERTScore F1: {F1.mean().item():.4f}")

    return pooled_outputs, baseline_outputs

if __name__ == "__main__":
    run_experiment()
