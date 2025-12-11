"""FiLM-based regressor copied from Firm-DTI (model2.FeatureTuner)."""

from __future__ import annotations

from typing import Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    # Optional: full ESM+Morgan model (used by some checkpoints)
    from esm_morgan.model import EsmMorganTuner as _EsmMorganTuner  # type: ignore
except Exception as e:  # pragma: no cover - optional dependency
    _EsmMorganTuner = None
    _esm_import_error = e
else:
    _esm_import_error = None



class AveragePool1dAlongAxis(nn.Module):
    """Mean-pool across a sequence axis while respecting padding masks."""

    def __init__(self, axis: int = 1) -> None:
        super().__init__()
        self.axis = axis

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is None:
            return torch.mean(x, dim=self.axis)

        x = x * mask.unsqueeze(-1).float()
        summed = torch.sum(x, dim=self.axis)
        counts = mask.sum(dim=self.axis, keepdim=True).clamp(min=1)
        return summed / counts


class FiLMProjector(nn.Module):
    """Applies FiLM conditioning to ligand representations."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.scale = nn.Linear(dim, dim)
        self.shift = nn.Linear(dim, dim)

    def forward(self, ligand: torch.Tensor, protein: torch.Tensor) -> torch.Tensor:
        gamma = torch.sigmoid(self.scale(protein))
        beta = self.shift(protein)
        return gamma * ligand + beta


class RBFDistanceHead(nn.Module):
    """Radial basis function head operating on cosine distances."""

    def __init__(self, k: int = 10, sigma: float = 0.2) -> None:
        super().__init__()
        self.register_buffer("centers", torch.linspace(0.0, 2.0, k))
        self.sigma = sigma
        self.out = nn.Linear(k, 1)

    def forward(self, ligand: torch.Tensor, protein: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ligand = F.normalize(ligand, dim=-1)
        protein = F.normalize(protein, dim=-1)
        distances = 1.0 - (ligand * protein).sum(dim=-1)
        phi = torch.exp(-((distances.unsqueeze(-1) - self.centers) ** 2) / (2 * self.sigma**2))
        affinity = self.out(phi).squeeze(-1)
        return affinity, distances


class FeatureTuner(nn.Module):
    """Twin MLP encoder with FiLM fusion for handcrafted descriptors."""

    def __init__(
        self,
        protein_dim: int,
        ligand_dim: int,
        hidden_dim: int = 256,
        rbf_k: int = 10,
        rbf_sigma: float = 0.2,
    ) -> None:
        super().__init__()
        self.protein_proj = nn.Linear(protein_dim, hidden_dim)
        self.ligand_proj = nn.Linear(ligand_dim, hidden_dim)
        self.protein_norm = nn.LayerNorm(hidden_dim)
        self.ligand_norm = nn.LayerNorm(hidden_dim)

        def _branch() -> nn.Sequential:
            return nn.Sequential(
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

        self.protein_head = _branch()
        self.ligand_head = _branch()
        self.film = FiLMProjector(hidden_dim)
        self.aff_head = RBFDistanceHead(k=rbf_k, sigma=rbf_sigma)

    @staticmethod
    def _ensure_tensor(x: torch.Tensor | Iterable[float]) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.float()
        return torch.as_tensor(x, dtype=torch.float32)

    def encode_protein(self, features: torch.Tensor) -> torch.Tensor:
        features = self._ensure_tensor(features)
        if features.dim() == 1:
            features = features.unsqueeze(0)
        hidden = self.protein_proj(features)
        protein_norm = getattr(self, "protein_norm", None)
        if protein_norm is not None:
            hidden = protein_norm(hidden)
        return self.protein_head(hidden)

    def encode_ligand(self, features: torch.Tensor) -> torch.Tensor:
        features = self._ensure_tensor(features)
        if features.dim() == 1:
            features = features.unsqueeze(0)
        hidden = self.ligand_proj(features)
        ligand_norm = getattr(self, "ligand_norm", None)
        if ligand_norm is not None:
            hidden = ligand_norm(hidden)
        return self.ligand_head(hidden)

    def single_pass(self, X: torch.Tensor) -> Tuple[torch.Tensor, int]:
        protein_repr = self.encode_protein(X)
        return protein_repr, 0

    def single_pass2(self, data_batch) -> torch.Tensor:
        if isinstance(data_batch, (list, tuple)):
            if len(data_batch) == 0:
                return torch.empty(0, self.protein_proj.out_features, device=self.protein_proj.weight.device)
            stacked = torch.stack([self._ensure_tensor(item) for item in data_batch], dim=0)
        else:
            stacked = self._ensure_tensor(data_batch)
        stacked = stacked.to(self.ligand_proj.weight.device)
        return self.encode_ligand(stacked)

    def forward(self, X_pos_neg, X_anchor):
        pos_batch = [item[0] for item in X_pos_neg]
        neg_batch = [item[1] for item in X_pos_neg]

        protein_repr, _ = self.single_pass(X_anchor)
        pos_repr = self.single_pass2(pos_batch)
        neg_repr = self.single_pass2(neg_batch)

        pos_fused = self.film(pos_repr, protein_repr)
        neg_fused = self.film(neg_repr, protein_repr)

        y_hat, dist_pos = self.aff_head(pos_fused, protein_repr)
        return protein_repr, pos_fused, neg_fused, y_hat

    def inference(self, X_pos, X_anchor):
        protein_repr, _ = self.single_pass(X_anchor)
        ligand_repr = self.single_pass2(X_pos)
        fused = self.film(ligand_repr, protein_repr)
        y_hat, _ = self.aff_head(fused, protein_repr)
        return protein_repr, fused, y_hat


# Expose EsmMorganTuner when available so old checkpoints can unpickle.
if _EsmMorganTuner is not None:
    EsmMorganTuner = _EsmMorganTuner
else:  # pragma: no cover - placeholder to surface a clear error if invoked
    class EsmMorganTuner(nn.Module):  # type: ignore
        def __init__(self, *_, **__):
            raise ImportError(f"esm_morgan.model is unavailable: {_esm_import_error}")

        def forward(self, *args, **kwargs):
            raise ImportError(f"esm_morgan.model is unavailable: {_esm_import_error}")

        def inference(self, *args, **kwargs):
            raise ImportError(f"esm_morgan.model is unavailable: {_esm_import_error}")


__all__ = ["FeatureTuner", "EsmMorganTuner", "FiLMProjector", "RBFDistanceHead"]
