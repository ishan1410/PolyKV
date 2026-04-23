import math
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.cache_utils import DynamicCache
from pool import SharedKVPool
from agents import PooledAgent
from datasets import load_dataset

MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"

wikitext = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
DOCUMENT = " ".join(wikitext["text"])[:8000]

AGENT_QUERIES = [
    "Agent A: What specific names, places, or dates are mentioned in this text?",
    "Agent B: What is the main subject described in the opening of this passage?",
    "Agent C: Describe any events or processes explained in this text.",
]

def build_cache(kv_pairs, model):
    cache = DynamicCache()
    for layer_idx, (k, v) in enumerate(kv_pairs):
        layer_device = model.model.layers[layer_idx].self_attn.q_proj.weight.device
        cache.update(k.to(layer_device), v.to(layer_device), layer_idx)
    return cache

def run_experiment():
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto"
    )
    model.eval()
    DEVICE = model.model.layers[0].self_attn.q_proj.weight.device
    print(f"Model loaded | dtype: {model.dtype} | device: {DEVICE}")
    for i in range(torch.cuda.device_count()):
        used = torch.cuda.memory_allocated(i) / 1e9
        print(f"  GPU {i} used: {used:.2f} GB")

    print("\n── PRE-RUN DOCUMENT INFO ──")
    doc_token_count = tokenizer(DOCUMENT, return_tensors="pt").input_ids.shape[1]
    print(f"Document token count: {doc_token_count}")

    print("Prefilling document context...")
    doc_tokens = tokenizer.encode(DOCUMENT, return_tensors="pt").to(DEVICE)
    # Fixed indexing from user provided snippet
    split_idx = int(doc_tokens.shape[1] * 0.7)
    context_tokens = doc_tokens[:, :split_idx]
    target_tokens  = doc_tokens[:, split_idx:]

    with torch.no_grad():
        prefill_out = model(context_tokens, use_cache=True)

    cache_obj = prefill_out.past_key_values
    kv_list = [
        (cache_obj.layers[i].keys, cache_obj.layers[i].values)
        for i in range(len(cache_obj.layers))
    ]
    print(f"KV cache: {len(kv_list)} layers, K shape: {kv_list[0][0].shape}")

    kv_list_cpu = [(k.cpu(), v.cpu()) for k, v in kv_list]

    print("Building SharedKVPool...")
    head_dim = kv_list_cpu[0][0].shape[-1]
    pool = SharedKVPool(head_dim=head_dim, device="cpu")
    pool.encode(kv_list_cpu)
    print(f"Compression ratio: {pool.get_compression_ratio():.2f}x")

    print("\nRunning agents from SHARED COMPRESSED pool...")
    pooled_agents = [PooledAgent(f"agent_{i}", pool, model, tokenizer) for i in range(3)]
    pooled_outputs = []
    for agent, query in zip(pooled_agents, AGENT_QUERIES):
        print(f"\n{agent.agent_id} querying...")
        response = agent.generate(query)
        pooled_outputs.append(response)
        print(f"Response: {response[:200]}...")

    print("\nRunning agents with FULL PRECISION KV (baseline)...")
    baseline_outputs = []
    for query in AGENT_QUERIES:
        input_ids = tokenizer.encode(query, return_tensors="pt").to(DEVICE)
        cache_clone = build_cache(
            [(k.clone().to(model.dtype), v.clone().to(model.dtype)) for k, v in kv_list_cpu],
            model
        )
        seq_len = cache_clone.get_seq_length()
        attention_mask = torch.ones(1, seq_len + input_ids.shape[1], dtype=torch.long, device=DEVICE)
        cache_position = torch.arange(seq_len, seq_len + input_ids.shape[1], device=DEVICE)
        with torch.no_grad():
            out = model.generate(
                input_ids,
                attention_mask=attention_mask,
                cache_position=cache_position,
                past_key_values=cache_clone,
                max_new_tokens=200,
                do_sample=False
            )
        response = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
        baseline_outputs.append(response)
        print(f"Baseline response: {response[:200]}...")

    print("\nMeasuring Perplexity...")
    with torch.no_grad():
        cache_ppl = build_cache(
            [(k.clone().to(model.dtype), v.clone().to(model.dtype)) for k, v in kv_list_cpu],
            model
        )
        seq_len_ppl   = cache_ppl.get_seq_length()
        attn_mask_ppl = torch.ones(1, seq_len_ppl + target_tokens.shape[1], dtype=torch.long, device=DEVICE)
        cache_pos_ppl = torch.arange(seq_len_ppl, seq_len_ppl + target_tokens.shape[1], device=DEVICE)
        baseline_out  = model(
            input_ids=target_tokens, labels=target_tokens,
            past_key_values=cache_ppl,
            attention_mask=attn_mask_ppl, cache_position=cache_pos_ppl
        )
        baseline_ppl = math.exp(baseline_out.loss.item())

        injected_kv = []
        for i in range(len(model.model.layers)):
             k, v = pool.get_kv_for_layer(i)
             injected_kv.append((k, v))
             
        cache_comp = build_cache(injected_kv, model)
        comp_out   = model(
            input_ids=target_tokens, labels=target_tokens,
            past_key_values=cache_comp,
            attention_mask=attn_mask_ppl, cache_position=cache_pos_ppl
        )
        compressed_ppl = math.exp(comp_out.loss.item())
        ppl_delta = (compressed_ppl - baseline_ppl) / baseline_ppl * 100

    print("\n── RESULTS ──")
    print(f"Model: {MODEL_NAME}")
    print(f"Compression ratio: {pool.get_compression_ratio():.2f}x memory saved")
    print(f"Memory: 3 agents share 1 pool vs 3 full-precision copies")
    print(f"Baseline PPL: {baseline_ppl:.3f} | Compressed PPL: {compressed_ppl:.3f} | Delta: {ppl_delta:.2f}%")

    for i, (pooled, baseline) in enumerate(zip(pooled_outputs, baseline_outputs)):
        if pooled.strip() == baseline.strip():
            overlap = 1.0
        else:
            pt = set(pooled.lower().split())
            bt = set(baseline.lower().split())
            overlap = len(pt & bt) / max(len(bt), 1)
        print(f"Agent {i}: token overlap = {overlap:.3f} ({'✓ Good' if overlap >= 0.9 else '✗ Degraded'})")

    return pooled_outputs, baseline_outputs

if __name__ == "__main__":
    run_experiment()
