"""Protein and ligand featurizers mirrored from the Firm-DTI trainer."""

from __future__ import annotations

from typing import Dict

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

AMINO = list("ACDEFGHIKLMNPQRSTVWY")
AA_INDEX: Dict[str, int] = {aa: idx for idx, aa in enumerate(AMINO)}


def is_kekulizable(smiles: str) -> bool:
    """Check whether a SMILES string can be sanitized and kekulized."""
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            return False
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        Chem.Kekulize(mol, clearAromaticFlags=True)
        return True
    except Exception:
        return False


def protein_feature_vector(sequence: str) -> np.ndarray:
    """Unigram+bigram amino-acid counts plus length (Firm-DTI handcrafted protein features)."""
    seq = (sequence or "").upper()
    unigram = np.zeros(len(AMINO), dtype=np.float32)
    bigram = np.zeros(len(AMINO) * len(AMINO), dtype=np.float32)
    total = 0
    total_bigrams = 0
    prev_idx: int | None = None
    for char in seq:
        idx = AA_INDEX.get(char)
        if idx is None:
            prev_idx = None
            continue
        unigram[idx] += 1.0
        total += 1
        if prev_idx is not None:
            bigram[prev_idx * len(AMINO) + idx] += 1.0
            total_bigrams += 1
        prev_idx = idx
    if total > 0:
        unigram /= total
    if total_bigrams > 0:
        bigram /= total_bigrams
    length = np.array([float(total)], dtype=np.float32)
    return np.concatenate([unigram, bigram, length], axis=0)


def ligand_feature_vector(smiles: str, bits: int, radius: int) -> np.ndarray:
    """Count-based Morgan fingerprint hashed into `bits` dimensions."""
    if not isinstance(smiles, str):
        return np.zeros(bits, dtype=np.float32)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(bits, dtype=np.float32)

    sparse = AllChem.GetMorganFingerprint(mol, radius, useCounts=True)
    arr = np.zeros(bits, dtype=np.float32)
    for feature_id, count in sparse.GetNonzeroElements().items():
        arr[feature_id % bits] += float(count)
    return arr
