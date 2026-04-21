# PolyKV Phase 0 Results

## Research Question
Do multiple agents sharing a single asymmetrically-compressed KV pool 
produce quality output comparable to full-precision per-agent KV caches?

## Method
- Model: HuggingFaceTB/SmolLM2-1.7B-Instruct
- Document: ~600 token Apollo 11 technical passage
- Compression: K at q8_0 (8-bit), V at TurboQuant MSE 3-bit (FWHT + Lloyd-Max)
- Agents: 3 agents sharing 1 pool vs 3 agents with full-precision KV

## Results
- Compression ratio: 2.91x memory reduction
- Baseline PPL: 14.085
- Compressed PPL: 14.159  
- PPL delta: 0.53% (well within 5% threshold)
- Agent 0 token overlap: 0.912
- Agent 1 token overlap: 1.000
- Agent 2 token overlap: 1.000

## Conclusion
Shared asymmetric TurboQuant-compressed KV pool preserves output quality
across multiple concurrent agents at 2.91x memory reduction.
This combination (shared + compressed + asymmetric + multi-reader) has not
been previously implemented or empirically validated.

## TurboQuant Implementation
- FWHT: recursive butterfly transform, normalized by 1/sqrt(d)
- Inverse FWHT: unnormalized butterfly / d (exploiting H*H = d*I property)  
- Lloyd-Max centroids (3-bit, N(0,1)): [-2.152, -1.344, -0.756, -0.245, 
  0.245, 0.756, 1.344, 2.152]
- K quantization: per-tensor int8 linear quantization
- V quantization: TurboQuant MSE with FWHT rotation
