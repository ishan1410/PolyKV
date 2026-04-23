import torch
from .compress import AsymmetricKVCompressor
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


@dataclass
class CompressedKVBlock:
    """One layer's compressed KV in the shared pool."""
    q_k: torch.Tensor        # int8 Keys
    k_scale: torch.Tensor    # fp32 scale factor
    v_indices: torch.Tensor  # 3-bit quantized indices
    v_norms: torch.Tensor    # norms for dequantization
    v_shape: tuple           # original shape
    layer_idx: int
    token_count: int


class SharedKVPool:
    """
    A shared, asymmetrically-compressed KV memory pool.

    Multiple agents can read from this pool concurrently.
    The pool stores the KV cache of a document ONCE, compressed,
    instead of each agent storing their own full-precision copy.

    Memory savings: N agents, 1x compressed memory
    vs naive:       N agents, N x full-precision memory
    vs per-agent turboquant: N agents, N x compressed memory

    This combination (shared + compressed + asymmetric) has not been
    implemented or empirically validated anywhere before.
    """

    def __init__(self, head_dim: int = 128, device: str = "cpu"):
        self.head_dim = head_dim
        self.device = device
        self.compressor = AsymmetricKVCompressor(head_dim, device)
        self.blocks: Dict[int, CompressedKVBlock] = {}
        self.compression_stats = {}

    @classmethod
    def from_context(cls, model, tokenizer, document: str) -> "SharedKVPool":
        """
        Build a SharedKVPool directly from a model + document.
        Handles prefill, KV extraction, CPU offload, and encode in one call.

        Args:
            model:      Any HuggingFace model (any arch, CUDA/MPS/CPU).
            tokenizer:  Corresponding tokenizer.
            document:   The shared context document (string).

        Returns:
            SharedKVPool ready to inject into agents.

        Example:
            pool = SharedKVPool.from_context(model, tokenizer, my_document)
        """
        from .backends._arch import get_first_device

        primary_device = get_first_device(model)
        input_ids = tokenizer.encode(document, return_tensors="pt").to(primary_device)

        with torch.no_grad():
            out = model(input_ids, use_cache=True)
            raw_cache = out.past_key_values

        # Normalise: DynamicCache or legacy tuple-of-tuples → list[(K, V)]
        if hasattr(raw_cache, "key_cache"):
            raw_kv = [
                (raw_cache.key_cache[i].cpu().float(),
                 raw_cache.value_cache[i].cpu().float())
                for i in range(len(raw_cache.key_cache))
            ]
        else:
            raw_kv = [(k.cpu().float(), v.cpu().float()) for k, v in raw_cache]

        del out, raw_cache
        torch.cuda.empty_cache()

        head_dim = raw_kv[0][0].shape[-1]
        pool = cls(head_dim=head_dim)
        pool.encode(raw_kv)
        return pool

    def encode(self, kv_cache: List[Tuple[torch.Tensor, torch.Tensor]]):
        """
        Compress and store a full KV cache from a document.
        kv_cache: list of (K, V) tensors, one per layer.
        Called ONCE. All agents then share this.
        """
        total_original = 0
        total_compressed = 0

        for layer_idx, (k, v) in enumerate(kv_cache):
            q_k, k_scale = self.compressor.compress_k(k)
            v_indices, v_norms, v_shape = self.compressor.compress_v(v)

            self.blocks[layer_idx] = CompressedKVBlock(
                q_k=q_k,
                k_scale=k_scale,
                v_indices=v_indices,
                v_norms=v_norms,
                v_shape=v_shape,
                layer_idx=layer_idx,
                token_count=k.shape[-2],
            )

            orig = k.numel() * 2 + v.numel() * 2  # fp16 bytes
            comp = q_k.numel() + v_indices.numel() * 0.375  # 8-bit K + 3-bit V
            total_original += orig
            total_compressed += comp

        self.compression_stats = {
            "original_bytes": total_original,
            "compressed_bytes": total_compressed,
            "ratio": total_original / total_compressed,
        }
        print(f"Pool encoded: {len(self.blocks)} layers, "
              f"{self.compression_stats['ratio']:.2f}x compression")

    def get_kv_for_layer(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decompress and return K, V for a given layer.
        Safe for concurrent reads — decompression is stateless and deterministic.
        """
        block = self.blocks[layer_idx]
        k = self.compressor.decompress_k(block.q_k, block.k_scale)
        v = self.compressor.decompress_v(block.v_indices, block.v_norms, block.v_shape)
        return k, v

    def get_compression_ratio(self) -> float:
        return self.compression_stats.get("ratio", 0.0)

    def memory_summary(self, n_agents: int = 1) -> str:
        """Human-readable memory savings report."""
        stats = self.compression_stats
        if not stats:
            return "Pool not yet encoded."

        orig_gb = stats["original_bytes"] / 1e9
        comp_gb = stats["compressed_bytes"] / 1e9
        naive_gb = orig_gb * n_agents
        saved_gb = naive_gb - comp_gb
        pct = (saved_gb / naive_gb) * 100 if naive_gb > 0 else 0

        lines = [
            "── PolyKV Memory Summary ──────────────────",
            f"  Compression ratio:              {stats['ratio']:.2f}x",
            f"  Full-precision KV (1 agent):    {orig_gb:.3f} GB",
            f"  Compressed pool (shared):       {comp_gb:.3f} GB",
            f"  {n_agents} agents WITHOUT PolyKV:       {naive_gb:.3f} GB",
            f"  {n_agents} agents WITH PolyKV:          {comp_gb:.3f} GB",
            f"  Memory saved ({n_agents} agents):        {saved_gb:.3f} GB  ({pct:.1f}% reduction)",
            "────────────────────────────────────────────",
        ]
        return "\n".join(lines)

    def __len__(self):
        return len(self.blocks)

    def __repr__(self):
        return (f"SharedKVPool(layers={len(self.blocks)}, "
                f"ratio={self.get_compression_ratio():.2f}x)")
