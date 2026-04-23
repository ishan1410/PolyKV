import numpy as np
import torch
from .turboquant import TurboQuantMSE

class AsymmetricKVCompressor:
    """
    Compresses KV tensors asymmetrically:
    - Keys (K): 8-bit linear quantization (q8_0) — protected
    - Values (V): 3-bit TurboQuant MSE with FWHT rotation — compressed
    
    Why asymmetric: K errors feed through softmax (exponential amplification).
    V errors scale linearly and vanish for near-zero attention weights.
    """

    def __init__(self, head_dim: int, device: str = "cpu"):
        self.head_dim = head_dim
        self.device = device
        # TurboQuant MSE quantizer for V — 3 bits per coordinate
        self.v_quantizer = TurboQuantMSE(dim=head_dim, bits=3, device=device)

    def compress_k(self, k: torch.Tensor):
        scale = k.abs().max() / 127.0
        scale = scale.clamp(min=1e-8)
        q_k = (k / scale).round().clamp(-128, 127).to(torch.int8)
        return q_k, scale

    def decompress_k(self, q_k: torch.int8, scale: torch.Tensor) -> torch.Tensor:
        return (q_k.float() * scale).to(scale.dtype)

    def compress_v(self, v: torch.Tensor):
        """
        3-bit TurboQuant MSE for Values.
        FWHT rotation spreads outlier energy, Lloyd-Max codebook quantizes.
        Returns: (indices, norms)
        """
        original_shape = v.shape
        v_flat = v.reshape(-1, self.head_dim)
        indices, norms = self.v_quantizer.quantize(v_flat)
        return indices, norms, original_shape

    def decompress_v(self, indices, norms, original_shape) -> torch.Tensor:
        v_flat = self.v_quantizer.dequantize(indices, norms)
        return v_flat.reshape(original_shape)
