import torch
from compress import AsymmetricKVCompressor
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

@dataclass
class CompressedKVBlock:
    """One layer's compressed KV in the shared pool."""
    # Keys: 8-bit protected
    q_k: torch.Tensor        # int8
    k_scale: torch.Tensor    # fp32 scale factor
    # Values: 3-bit TurboQuant compressed  
    v_indices: torch.Tensor  # quantized indices
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
        self.blocks: Dict[int, CompressedKVBlock] = {}  # layer_idx -> block
        self.reader_count = 0  # track concurrent agent reads
        self.compression_stats = {}

    def encode(self, kv_cache: List[Tuple[torch.Tensor, torch.Tensor]]):
        """
        Compress and store a full KV cache from a document.
        kv_cache: list of (K, V) tensors, one per layer
        Called ONCE. All agents then share this.
        """
        total_original = 0
        total_compressed = 0

        for layer_idx, (k, v) in enumerate(kv_cache):
            # Compress K at 8-bit
            q_k, k_scale = self.compressor.compress_k(k)
            # Compress V at 3-bit TurboQuant MSE
            v_indices, v_norms, v_shape = self.compressor.compress_v(v)

            self.blocks[layer_idx] = CompressedKVBlock(
                q_k=q_k,
                k_scale=k_scale,
                v_indices=v_indices,
                v_norms=v_norms,
                v_shape=v_shape,
                layer_idx=layer_idx,
                token_count=k.shape[-2]
            )

            # Track compression ratio
            orig = k.numel() * 2 + v.numel() * 2  # fp16 bytes
            comp = q_k.numel() + v_indices.numel() * 0.375  # 8-bit K + 3-bit V
            total_original += orig
            total_compressed += comp

        self.compression_stats = {
            "original_bytes": total_original,
            "compressed_bytes": total_compressed,
            "ratio": total_original / total_compressed
        }
        print(f"Pool encoded: {len(self.blocks)} layers, "
              f"{self.compression_stats['ratio']:.2f}x compression")

    def get_kv_for_layer(self, layer_idx: int
                         ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decompress and return K, V for a given layer.
        Called by each agent independently — safe for concurrent reads
        because decompression is stateless and deterministic.
        """
        block = self.blocks[layer_idx]
        k = self.compressor.decompress_k(block.q_k, block.k_scale)
        v = self.compressor.decompress_v(
            block.v_indices, block.v_norms, block.v_shape
        )
        return k, v

    def get_compression_ratio(self) -> float:
        return self.compression_stats.get("ratio", 0.0)
