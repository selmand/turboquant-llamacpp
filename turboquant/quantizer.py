"""
TurboQuant Kuantizer — v0.2 DÜZELTILMIŞ
==========================================
Algoritma 1: TurboQuantMSE — değişmedi, doğru
Algoritma 2: TurboQuantProd — kritik düzeltmeler:

  BUG-4: Rezidüel normu (||x - x̂_mse||) saklanmıyordu.
         QJL estimatörü bu norma ihtiyaç duyar.
         QuantizedVector'a residual_norm alanı eklendi.
         
  BUG-5: attention_score() QJL'e rezidüel normu vermiyordu.
"""

import numpy as np
from typing import Optional, Tuple, NamedTuple
from .rotation import generate_rotation_matrix, rotate_vector, inverse_rotate_vector
from .lloyd_max import LloydMaxQuantizer
from .qjl import QJL


class QuantizedVector(NamedTuple):
    """Kuantize edilmiş vektör paketi."""
    norm: float                          # Orijinal vektör normu
    indices: np.ndarray                  # Lloyd-Max indeksleri (uint8)
    qjl_signs: Optional[np.ndarray]      # QJL sign bitleri (sadece Prod)
    residual_norm: Optional[float] = None # ||x - x̂_mse|| (sadece Prod) — BUG-4 FIX


class TurboQuantMSE:
    """
    Algoritma 1: MSE-Optimal TurboQuant (değişmedi)
    
    Pipeline:
      Encode: x → norm, x/norm → Π·(x/norm) → Lloyd-Max indices
      Decode: indices → centroids → Π^T·centroids → ×norm
    """
    
    def __init__(self, d: int = 128, bits: int = 3, seed: int = 42):
        self.d = d
        self.bits = bits
        self.seed = seed
        self.Pi = generate_rotation_matrix(d, seed=seed)
        self.quantizer = LloydMaxQuantizer(d, bits)
        self._norm_bytes = 4
        self._index_bytes = (d * bits + 7) // 8
    
    def quantize(self, x: np.ndarray) -> QuantizedVector:
        is_batch = x.ndim > 1
        if not is_batch:
            x = x[np.newaxis, :]
        
        norms = np.linalg.norm(x, axis=-1)
        safe_norms = np.where(norms > 1e-10, norms, 1.0)
        x_normalized = x / safe_norms[:, np.newaxis]
        x_rotated = rotate_vector(x_normalized, self.Pi)
        indices = self.quantizer.quantize(x_rotated)
        
        if not is_batch:
            return QuantizedVector(norm=float(norms[0]), indices=indices[0],
                                   qjl_signs=None, residual_norm=None)
        return QuantizedVector(norm=norms, indices=indices,
                               qjl_signs=None, residual_norm=None)
    
    def dequantize(self, qv: QuantizedVector) -> np.ndarray:
        is_batch = qv.indices.ndim > 1
        indices = qv.indices if is_batch else qv.indices[np.newaxis, :]
        norms = qv.norm if is_batch else np.array([qv.norm])
        
        x_rot_hat = self.quantizer.dequantize(indices)
        x_hat = inverse_rotate_vector(x_rot_hat, self.Pi)
        
        if is_batch:
            return x_hat * norms[:, np.newaxis]
        return x_hat[0] * norms[0]
    
    def get_mse(self, x: np.ndarray) -> float:
        qv = self.quantize(x)
        x_hat = self.dequantize(qv)
        return float(np.mean((x - x_hat) ** 2))
    
    def compression_ratio(self) -> float:
        return (self.d * 2) / (self._norm_bytes + self._index_bytes)
    
    def memory_per_vector(self) -> int:
        return self._norm_bytes + self._index_bytes


class TurboQuantProd:
    """
    Algoritma 2: Tarafsız İç Çarpım TurboQuant — DÜZELTILMIŞ
    
    İki aşama:
      1. (b-1) bit MSE → x̂_mse
      2. residual = x - x̂_mse → QJL(residual) → 1-bit sign + residual_norm
    
    Attention skoru:
      <q, x>_hat = <q, x̂_mse> + QJL_estimate(q, residual)
                 = <q, x̂_mse> + ||res||·√(π/2)·(1/m)·Σ sign(S·res)·(S·q)
    
    Tarafsızlık kanıtı:
      E[<q, x̂_mse> + QJL(q, res)] = <q, x̂_mse> + <q, x - x̂_mse> = <q, x>
    """
    
    def __init__(self, d: int = 128, bits: int = 4, seed: int = 42):
        assert bits >= 2
        self.d = d
        self.bits = bits
        self.mse_bits = bits - 1
        self.mse_quantizer = TurboQuantMSE(d=d, bits=self.mse_bits, seed=seed)
        self.qjl = QJL(d=d, m=d, seed=seed + 1000)
    
    def quantize(self, x: np.ndarray) -> QuantizedVector:
        # Stage 1: MSE
        mse_qv = self.mse_quantizer.quantize(x)
        x_hat_mse = self.mse_quantizer.dequantize(mse_qv)
        
        # Rezidüel
        residual = x - x_hat_mse
        
        # Stage 2: QJL — BUG-4 FIX: norm da saklanıyor
        qjl_signs, residual_norm = self.qjl.encode(residual)
        
        return QuantizedVector(
            norm=mse_qv.norm,
            indices=mse_qv.indices,
            qjl_signs=qjl_signs,
            residual_norm=float(residual_norm)  # BUG-4 FIX
        )
    
    def dequantize(self, qv: QuantizedVector) -> np.ndarray:
        """MSE aşamasının geri yapılandırması (QJL dequant için değil, skor için)."""
        mse_qv = QuantizedVector(norm=qv.norm, indices=qv.indices,
                                  qjl_signs=None, residual_norm=None)
        return self.mse_quantizer.dequantize(mse_qv)
    
    def attention_score(self, query: np.ndarray, compressed_key: QuantizedVector) -> float:
        """
        Tarafsız attention skoru — BUG-5 FIX
        
        <q, k>_hat = <q, k̂_mse> + ||residual||·√(π/2)·(1/m)·Σ sign(S·res)·(S·q)
        """
        k_hat_mse = self.dequantize(compressed_key)
        mse_score = float(np.dot(query, k_hat_mse))
        
        if compressed_key.qjl_signs is not None and compressed_key.residual_norm is not None:
            # BUG-5 FIX: residual_norm QJL estimatörüne veriliyor
            qjl_correction = self.qjl.inner_product_estimate(
                query, compressed_key.qjl_signs, compressed_key.residual_norm
            )
            return mse_score + qjl_correction
        
        return mse_score
    
    def compression_ratio(self) -> float:
        fp16_bytes = self.d * 2
        mse_bytes = self.mse_quantizer.memory_per_vector()
        qjl_bytes = (self.d + 7) // 8 + 4  # signs + residual_norm
        return fp16_bytes / (mse_bytes + qjl_bytes)


# ─── Bit Packing (değişmedi, doğru) ───

def pack_indices(indices: np.ndarray, bits: int) -> np.ndarray:
    d = indices.shape[-1]
    n_bytes = (d * bits + 7) // 8
    if indices.ndim == 1:
        packed = np.zeros(n_bytes, dtype=np.uint8)
        bit_pos = 0
        for i in range(d):
            val = int(indices[i])
            for b in range(bits):
                packed[bit_pos // 8] |= ((val >> b) & 1) << (bit_pos % 8)
                bit_pos += 1
        return packed
    return np.stack([pack_indices(indices[i], bits) for i in range(indices.shape[0])])


def unpack_indices(packed: np.ndarray, bits: int, d: int) -> np.ndarray:
    if packed.ndim == 1:
        indices = np.zeros(d, dtype=np.uint8)
        bit_pos = 0
        for i in range(d):
            val = 0
            for b in range(bits):
                val |= ((int(packed[bit_pos // 8]) >> (bit_pos % 8)) & 1) << b
                bit_pos += 1
            indices[i] = val
        return indices
    return np.stack([unpack_indices(packed[i], bits, d) for i in range(packed.shape[0])])
