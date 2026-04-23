import torch
import math


class TurboQuantMSE:
    def __init__(self, dim: int, bits: int = 3, device: str = "cpu"):
        self.dim = dim
        self.bits = bits
        self.device = device
        # Lloyd-Max centroids for N(0,1), 3-bit (8 centroids)
        self.centroids = torch.tensor(
            [-2.152, -1.344, -0.756, -0.245, 0.245, 0.756, 1.344, 2.152],
            dtype=torch.float32, device=device
        )
        assert (dim & (dim - 1)) == 0, "dim must be a power of 2 for FWHT"

    def _fwht(self, a: torch.Tensor) -> torch.Tensor:
        """Normalized Fast Walsh-Hadamard Transform (vectorized)."""
        d = a.shape[-1]
        h = 1
        while h < d:
            a = a.view(*a.shape[:-1], d // (2 * h), 2, h)
            a1 = a[..., 0, :].clone()
            a2 = a[..., 1, :].clone()
            a = torch.stack([a1 + a2, a1 - a2], dim=-2)
            a = a.view(*a.shape[:-3], d)
            h *= 2
        return a / math.sqrt(d)

    def _fwht_unnormalized(self, a: torch.Tensor) -> torch.Tensor:
        """Unnormalized FWHT for inverse (H*H = d*I)."""
        d = a.shape[-1]
        h = 1
        while h < d:
            a = a.view(*a.shape[:-1], d // (2 * h), 2, h)
            a1 = a[..., 0, :].clone()
            a2 = a[..., 1, :].clone()
            a = torch.stack([a1 + a2, a1 - a2], dim=-2)
            a = a.view(*a.shape[:-3], d)
            h *= 2
        return a

    def quantize(self, v_flat: torch.Tensor):
        norms = v_flat.norm(p=2, dim=-1, keepdim=True)
        v_unit = v_flat / norms.clamp(min=1e-8)
        v_fwht = self._fwht(v_unit)
        v_scaled = v_fwht * math.sqrt(self.dim)
        diffs = v_scaled.unsqueeze(-1) - self.centroids
        indices = diffs.abs().argmin(dim=-1).to(torch.int8)
        return indices, norms

    def dequantize(self, indices: torch.Tensor, norms: torch.Tensor):
        v_rec_scaled = self.centroids[indices.long()]
        v_rec_unit = self._fwht_unnormalized(v_rec_scaled) / self.dim
        return v_rec_unit * norms


if __name__ == "__main__":
    torch.manual_seed(42)
    tq = TurboQuantMSE(dim=64, bits=3)
    x = torch.randn(10, 64)
    indices, norms = tq.quantize(x)
    x_hat = tq.dequantize(indices, norms)
    cos_sim = torch.nn.functional.cosine_similarity(x, x_hat).mean()
    mse = ((x - x_hat) ** 2).mean()
    print(f"Cosine similarity: {cos_sim:.4f}  (target: > 0.90)")
    print(f"MSE: {mse:.4f}             (target: < 0.50)")
