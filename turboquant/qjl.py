"""
Quantized Johnson-Lindenstrauss (QJL) Transform — v0.2 DÜZELTILMIŞ
====================================================================
Referans: Zandieh et al., AAAI 2025 (arXiv:2406.03482)

Düzeltilen hatalar:
  BUG-1: S matrisi N(0,1/m) → N(0,1) olmalı. Sign ölçekten bağımsız
         ama (S·y) projeksiyonu ölçeğe bağlı, estimatör formülünü bozar.
  BUG-2: (1/m) normalizasyonu eksikti.
  BUG-3: x_norm (kaynak normu) saklanmıyordu, estimatöre verilmiyordu.

Doğru formül:
  <x,y>_hat = ||x|| · √(π/2) · (1/m) · Σᵢ sign(Sᵢ·x) · (Sᵢ·y)
  
  Türetme (bivariate Gaussian):
    Sᵢ ~ N(0, I_d) satır vektörü
    E[sign(Sᵢ·x) · (Sᵢ·y)] = √(2/π) · <x,y> / ||x||
    → (1/m) Σ ile ortala, √(π/2)·||x|| ile düzelt → tarafsız
"""

import numpy as np
from typing import Optional, Tuple


class QJL:
    def __init__(self, d: int, m: Optional[int] = None, seed: Optional[int] = None):
        self.d = d
        self.m = m if m is not None else d
        self.seed = seed
        rng = np.random.default_rng(seed)
        # BUG-1 FIX: Standart Gaussian, ölçeklenmemiş
        self.S = rng.standard_normal((self.m, d))
    
    def encode(self, x: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Returns:
            signs: (..., m) {-1, +1}
            norm: ||x|| skaler — estimatör için zorunlu
        """
        norm = float(np.linalg.norm(x)) if x.ndim == 1 else np.linalg.norm(x, axis=-1)
        z = x @ self.S.T
        signs = np.sign(z)
        signs[signs == 0] = 1.0
        return signs.astype(np.int8), norm
    
    def inner_product_estimate(
        self, y: np.ndarray, x_signs: np.ndarray, x_norm: float
    ) -> float:
        """
        Tarafsız iç çarpım: ||x||·√(π/2)·(1/m)·Σᵢ sign(Sᵢ·x)·(Sᵢ·y)
        
        BUG-2 FIX: (1/m) eklendi
        BUG-3 FIX: x_norm zorunlu
        """
        y_proj = y @ self.S.T
        raw_sum = np.sum(x_signs.astype(np.float64) * y_proj, axis=-1)
        return float(x_norm * np.sqrt(np.pi / 2.0) * raw_sum / self.m)
    
    def encode_packed(self, x: np.ndarray) -> Tuple[np.ndarray, float]:
        signs, norm = self.encode(x)
        bits = ((signs + 1) // 2).astype(np.uint8)
        flat = bits.reshape(-1, self.m)
        n_bytes = (self.m + 7) // 8
        packed = np.zeros((flat.shape[0], n_bytes), dtype=np.uint8)
        for i in range(self.m):
            packed[:, i // 8] |= (flat[:, i] << (i % 8))
        return packed.reshape(bits.shape[:-1] + (n_bytes,)), norm
    
    def decode_packed(self, packed: np.ndarray) -> np.ndarray:
        flat = packed.reshape(-1, packed.shape[-1])
        signs = np.zeros((flat.shape[0], self.m), dtype=np.int8)
        for i in range(self.m):
            signs[:, i] = 2 * ((flat[:, i // 8] >> (i % 8)) & 1).astype(np.int8) - 1
        return signs.reshape(packed.shape[:-1] + (self.m,))
    
    def get_memory_bytes(self, n_vectors: int) -> int:
        return n_vectors * ((self.m + 7) // 8 + 4)  # signs + norm(float32)
