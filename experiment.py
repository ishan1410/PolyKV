import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from pool import SharedKVPool
from agents import PooledAgent

MODEL_NAME = "HuggingFaceTB/SmolLM2-1.7B-Instruct"  # swapped to 1.7B

DOCUMENT = """
Title: The Technical Contributions and Limitations of the Apollo 11 Mission

Abstract:
The Apollo 11 mission represents one of the most significant achievements in human history, marking the first time humans set foot on another celestial body. The main argument of this document is that the success of the Apollo program relied not merely on astronaut bravery, but on an unprecedented integration of novel aerospace computer systems and orbital mechanics techniques. Launched on July 16, 1969, the mission was the culmination of an ambitious endeavor to land a man on the Moon and return him safely to Earth before the end of the decade. The spacecraft was crewed by three astronauts: Armstrong, Collins, and Aldrin. After a journey of over 240,000 miles, the crew achieved orbit, and the delicate descent phase began.

The key technical contributions mentioned in this account focus extensively on the Apollo Guidance Computer (AGC). This computer pioneered the use of integrated circuits, offering unprecedented computational power in a lightweight, compact package entirely critical for spaceflight. Another vital technical contribution was the development of the Lunar Orbit Rendezvous (LOR) strategy. Instead of sending a massive, un-stageable rocket directly to the lunar surface and back, LOR mandated that a specialized, lightweight Lunar Module detach, land, and then reconnect with the orbiting Command Module. This modular methodology drastically reduced the total mass required to escape Earth's gravity, serving as a fundamental architectural blueprint for all deep space missions that followed.

However, the historical success often overshadows the profound difficulties inherent in early spaceflight. What limitations or weaknesses does the document acknowledge? The document acknowledges that the primary limitation of the Apollo 11 architecture was the extreme scarcity of automated redundancies during the landing sequence. Specifically, as the Eagle approached the Sea of Tranquility, the targeting radar was overwhelmed by processing interrupts, famously causing the 1202 program alarm. This weakness required Armstrong to take manual control of the spacecraft, burning precious propellant to manually steer clear of a boulder field. This highlighted a stark vulnerability: a heavy reliance on finite, manual human intervention in split-second, life-or-death decision pathways where computational safety networks were insufficiently robust.

Furthermore, fuel capacity presented a strict operational limitation. The descent propulsion system had an agonizingly small margin of error. When the Eagle finally touched down, Mission Control noted they had mere seconds of hover time left before an emergency abort protocol would have been automatically triggered. This fragility illustrated the incredible risks accepted by the crew. Despite these limitations, the astronauts successfully completed their surface activities. They deployed seismometers, retroreflectors, and gathered substantial lunar samples. Following these rigorous surface tasks, the rendezvous and trans-Earth injection were executed properly. Ultimately, while the Apollo 11 mission established the pinnacle of 1960s technological capability, it also revealed the significant computational limits of the era's avionics, paving the way for the fault-tolerant software architectures that govern modern aerospace design today.

In conclusion, revisiting the Apollo 11 mission from an engineering perspective underscores how far computational paradigms have evolved. The AGC operated with a clock speed of barely one megahertz and possessed only a few kilobytes of memory. By modern standards, integrating orbital mechanics, telemetry parsing, and human-in-the-loop landing systems on such constrained hardware seems almost entirely impossible. Yet, the mission engineers compensated for these hardware deficits with tightly optimized, priority-scheduled software structures that could gracefully shed lower-priority tasks when overloaded, as evidenced by the 1202 alarm handling. This architectural foresight represents a profound achievement in systems software engineering. While the primary argument surrounding Apollo often centers on national prestige or geopolitical maneuvering, the raw technical contributions—ranging from fly-by-wire controls to mathematical optimization in deep-space environments—remain the most durable legacy of the mission. Acknowledging the extreme limitations they operated under only magnifies the scale of their accomplishment.
"""

# --- Queries for agents to answer about the document ---
AGENT_QUERIES = [
    "Agent A: Summarize the main argument of this document in 3 sentences.",
    "Agent B: What are the key technical contributions mentioned?",
    "Agent C: What limitations or weaknesses does the document acknowledge?",
    "Agent D: What specific hardware constraints does the document mention?",
    "Agent E: What year did the event described in this document take place?"
]

def run_experiment():
    import math
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16
    )
    model.eval()

    # ── STEP 1: Prefill document, capture KV cache ──
    print("Prefilling document context...")
    doc_tokens = tokenizer.encode(DOCUMENT, return_tensors="pt")
    
    # Split document for PPL measurement 
    split_idx = int(doc_tokens.shape[1] * 0.7)
    context_tokens = doc_tokens[:, :split_idx]
    target_tokens = doc_tokens[:, split_idx:]
    
    with torch.no_grad():
        prefill_out = model(context_tokens, use_cache=True)
    raw_kv_cache = prefill_out.past_key_values

    # ── STEP 2: Build shared compressed pool (encode ONCE) ──
    print("Building SharedKVPool...")
    head_dim = raw_kv_cache[0][0].shape[-1]
    pool = SharedKVPool(head_dim=head_dim)
    pool.encode(raw_kv_cache)
    print(f"Compression ratio: {pool.get_compression_ratio():.2f}x")

    # ── STEP 3: Run 5 agents from shared pool ──
    print("\nRunning agents from SHARED COMPRESSED pool...")
    pooled_agents = [
        PooledAgent(f"agent_{i}", pool, model, tokenizer)
        for i in range(5)
    ]
    pooled_outputs = []
    for agent, query in zip(pooled_agents, AGENT_QUERIES):
        print(f"\n{agent.agent_id} querying...")
        response = agent.generate(query)
        pooled_outputs.append(response)
        print(f"Response: {response[:200]}...")

    # ── STEP 4: Run same agents with FULL PRECISION KV (baseline) ──
    print("\nRunning agents with FULL PRECISION KV (baseline)...")
    baseline_outputs = []
    from transformers.cache_utils import DynamicCache
    for query in AGENT_QUERIES:
        input_ids = tokenizer.encode(query, return_tensors="pt")
        with torch.no_grad():
            cache_clone = DynamicCache.from_legacy_cache(tuple((k.clone(), v.clone()) for k, v in raw_kv_cache))
            seq_len = cache_clone.get_seq_length()
            attention_mask = torch.ones(1, seq_len + input_ids.shape[1], dtype=torch.long, device=input_ids.device)
            cache_position = torch.arange(seq_len, seq_len + input_ids.shape[1], device=input_ids.device)
            out = model.generate(
                input_ids,
                attention_mask=attention_mask,
                cache_position=cache_position,
                past_key_values=cache_clone,
                max_new_tokens=200,
                do_sample=False
            )
        response = tokenizer.decode(
            out[0][input_ids.shape[1]:], skip_special_tokens=True
        )
        baseline_outputs.append(response)
        print(f"Baseline response: {response[:200]}...")

    # ── STEP 5: Perplexity Measurement ──
    print("\nMeasuring Perplexity...")
    with torch.no_grad():
        # Baseline PPL
        cache_clone_ppl = DynamicCache.from_legacy_cache(tuple((k.clone(), v.clone()) for k, v in raw_kv_cache))
        seq_len_ppl = cache_clone_ppl.get_seq_length()
        attn_mask_ppl = torch.ones(1, seq_len_ppl + target_tokens.shape[1], dtype=torch.long, device=model.device)
        cache_pos_ppl = torch.arange(seq_len_ppl, seq_len_ppl + target_tokens.shape[1], device=model.device)
        baseline_out = model(
            input_ids=target_tokens, 
            labels=target_tokens, 
            past_key_values=cache_clone_ppl,
            attention_mask=attn_mask_ppl,
            cache_position=cache_pos_ppl
        )
        baseline_ppl = math.exp(baseline_out.loss.item())

        # Compressed PPL
        injected_kv = []
        for layer_idx in range(len(model.model.layers)):
            k, v = pool.get_kv_for_layer(layer_idx)
            injected_kv.append((k.to(model.dtype), v.to(model.dtype)))
        cache_comp = DynamicCache.from_legacy_cache(tuple(injected_kv))
        
        comp_out = model(
            input_ids=target_tokens, 
            labels=target_tokens, 
            past_key_values=cache_comp,
            attention_mask=attn_mask_ppl,
            cache_position=cache_pos_ppl
        )
        compressed_ppl = math.exp(comp_out.loss.item())
        ppl_delta = (compressed_ppl - baseline_ppl) / baseline_ppl * 100

    # ── STEP 6: Measure quality ──
    print("\n── RESULTS ──")
    print(f"Compression ratio: {pool.get_compression_ratio():.2f}x memory saved")
    print(f"Memory: {5} agents share 1 pool vs {5} full-precision copies")
    print(f"Baseline PPL: {baseline_ppl:.3f} | Compressed PPL: {compressed_ppl:.3f} | Delta: {ppl_delta:.2f}%")
    
    # Simple semantic similarity check
    for i, (pooled, baseline) in enumerate(
        zip(pooled_outputs, baseline_outputs)
    ):
        # Token overlap as rough quality proxy
        if pooled.strip() == baseline.strip():
            overlap = 1.0
        else:
            pooled_tokens = set(pooled.lower().split())
            baseline_tokens = set(baseline.lower().split())
            overlap = len(pooled_tokens & baseline_tokens) / \
                      max(len(baseline_tokens), 1)
        print(f"Agent {i}: token overlap with baseline = {overlap:.3f} "
              f"({'✓ Good' if overlap > 0.5 else '✗ Degraded'})")

    return pooled_outputs, baseline_outputs

if __name__ == "__main__":
    run_experiment()
