"""
TurboQuant KV Cache Yöneticisi — v0.2 DÜZELTILMIŞ
=====================================================
Düzeltilen hata:
  BUG-6: Prod modunda attention skorları QJL düzeltmesi içermiyordu.
         compute_attention() sadece dequantize+dot yapıyordu.
         Artık TurboQuantProd ise attention_score() metodu kullanılıyor.

Tasarım:
  - Son N token (residual window) FP16'da
  - Eski tokenlar TurboQuant ile sıkıştırılmış
  - Prod modunda QJL düzeltmesi dahil
"""

import numpy as np
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass, field

from .quantizer import TurboQuantMSE, TurboQuantProd, QuantizedVector, pack_indices, unpack_indices


@dataclass
class CacheConfig:
    head_dim: int = 128
    n_heads: int = 8
    n_layers: int = 32
    key_bits: int = 3
    value_bits: int = 4
    residual_window: int = 128
    use_qjl: bool = False        # True = TurboQuantProd, False = TurboQuantMSE
    seed: int = 42


@dataclass
class CompressedToken:
    """Sıkıştırılmış KV verisi — artık residual_norm da var."""
    token_idx: int
    # Key
    key_norm: float
    key_indices: np.ndarray
    key_qjl_signs: Optional[np.ndarray] = None
    key_residual_norm: Optional[float] = None     # BUG-4/6 FIX
    # Value
    value_norm: float = 0.0
    value_indices: Optional[np.ndarray] = None
    value_qjl_signs: Optional[np.ndarray] = None
    value_residual_norm: Optional[float] = None   # BUG-4/6 FIX


class TurboQuantKVCache:
    """
    TurboQuant KV Cache — düzeltilmiş attention hesaplaması.
    """
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.d = config.head_dim
        self.use_prod = config.use_qjl
        
        # Kuantizerlar
        if self.use_prod:
            self.key_quantizer = TurboQuantProd(d=self.d, bits=config.key_bits, seed=config.seed)
            self.value_quantizer = TurboQuantProd(d=self.d, bits=config.value_bits, seed=config.seed + 500)
        else:
            self.key_quantizer = TurboQuantMSE(d=self.d, bits=config.key_bits, seed=config.seed)
            self.value_quantizer = TurboQuantMSE(d=self.d, bits=config.value_bits, seed=config.seed + 500)
        
        # Depolama
        self.compressed: Dict[Tuple[int, int], List[CompressedToken]] = {}
        self.residual_keys: Dict[Tuple[int, int], List[np.ndarray]] = {}
        self.residual_values: Dict[Tuple[int, int], List[np.ndarray]] = {}
        
        self.total_tokens = 0
        
        for layer in range(config.n_layers):
            for head in range(config.n_heads):
                k = (layer, head)
                self.compressed[k] = []
                self.residual_keys[k] = []
                self.residual_values[k] = []
    
    def append(self, keys: np.ndarray, values: np.ndarray, token_idx: int,
               layer: int = 0, head: int = 0):
        lh = (layer, head)
        self.residual_keys[lh].append(keys.copy())
        self.residual_values[lh].append(values.copy())
        
        if len(self.residual_keys[lh]) > self.config.residual_window:
            old_k = self.residual_keys[lh].pop(0)
            old_v = self.residual_values[lh].pop(0)
            self._compress_and_store(old_k, old_v, token_idx - self.config.residual_window, lh)
        
        self.total_tokens = token_idx + 1
    
    def _compress_and_store(self, key, value, token_idx, lh):
        key_qv = self.key_quantizer.quantize(key)
        val_qv = self.value_quantizer.quantize(value)
        
        ct = CompressedToken(
            token_idx=token_idx,
            key_norm=float(key_qv.norm) if np.isscalar(key_qv.norm) else float(key_qv.norm),
            key_indices=pack_indices(key_qv.indices, self.config.key_bits),
            key_qjl_signs=key_qv.qjl_signs,
            key_residual_norm=key_qv.residual_norm,      # BUG-4/6 FIX
            value_norm=float(val_qv.norm) if np.isscalar(val_qv.norm) else float(val_qv.norm),
            value_indices=pack_indices(val_qv.indices, self.config.value_bits),
            value_qjl_signs=val_qv.qjl_signs,
            value_residual_norm=val_qv.residual_norm,    # BUG-4/6 FIX
        )
        self.compressed[lh].append(ct)
    
    def compute_attention(self, query: np.ndarray, layer: int = 0, head: int = 0,
                          temperature: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Attention hesapla — BUG-6 FIX: Prod modunda QJL düzeltmesi uygulanıyor.
        """
        lh = (layer, head)
        scale = temperature / np.sqrt(self.d)
        
        all_scores = []
        all_values = []
        
        # 1. Sıkıştırılmış tokenlar
        for ct in self.compressed[lh]:
            key_qv = QuantizedVector(
                norm=ct.key_norm,
                indices=unpack_indices(ct.key_indices, self.config.key_bits, self.d),
                qjl_signs=ct.key_qjl_signs,
                residual_norm=ct.key_residual_norm
            )
            
            # BUG-6 FIX: Prod modunda attention_score kullan
            if self.use_prod and isinstance(self.key_quantizer, TurboQuantProd):
                score = self.key_quantizer.attention_score(query, key_qv) * scale
            else:
                key_hat = self.key_quantizer.dequantize(key_qv)
                score = np.dot(query, key_hat) * scale
            
            all_scores.append(score)
            
            # Value geri yapılandır (value için QJL gereksiz, sadece MSE yeterli)
            val_qv = QuantizedVector(
                norm=ct.value_norm,
                indices=unpack_indices(ct.value_indices, self.config.value_bits, self.d),
                qjl_signs=None, residual_norm=None
            )
            all_values.append(self.value_quantizer.dequantize(val_qv))
        
        # 2. Residual window (FP16)
        for k in self.residual_keys[lh]:
            all_scores.append(np.dot(query, k) * scale)
        for v in self.residual_values[lh]:
            all_values.append(v)
        
        if not all_scores:
            return np.zeros(self.d), np.array([])
        
        # 3. Softmax
        scores = np.array(all_scores)
        scores -= np.max(scores)
        weights = np.exp(scores)
        weights /= (np.sum(weights) + 1e-10)
        
        # 4. Ağırlıklı value
        output = np.sum(weights[:, np.newaxis] * np.stack(all_values), axis=0)
        return output, weights
    
    def memory_usage(self) -> dict:
        compressed_count = sum(len(v) for v in self.compressed.values())
        residual_count = sum(len(v) for v in self.residual_keys.values())
        
        key_packed = (self.d * self.config.key_bits + 7) // 8
        val_packed = (self.d * self.config.value_bits + 7) // 8
        per_token = 4 + key_packed + 4 + val_packed
        if self.use_prod:
            per_token += (self.d + 7) // 8 + 4  # QJL signs + residual_norm (key)
        
        fp16_per = self.d * 2 * 2
        total = compressed_count + residual_count
        fp16_total = total * fp16_per
        comp_total = compressed_count * per_token + residual_count * fp16_per
        
        return {
            "total_tokens": total,
            "compressed_tokens": compressed_count,
            "residual_tokens": residual_count,
            "compressed_bytes": comp_total,
            "fp16_baseline_bytes": fp16_total,
            "compression_ratio": fp16_total / max(comp_total, 1),
            "memory_saved_mb": (fp16_total - comp_total) / (1024 * 1024),
        }
    
    def clear(self):
        for k in self.compressed:
            self.compressed[k].clear()
            self.residual_keys[k].clear()
            self.residual_values[k].clear()
        self.total_tokens = 0
