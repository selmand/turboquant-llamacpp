"""
Rastgele Ortogonal Rotasyon Matrisi
====================================
TurboQuant'ın ilk adımı: Haar dağılımlı rastgele ortogonal matris.

Amaç: Girdi vektörünü döndürerek her koordinatın Beta dağılımına
uymasını sağlamak. Bu sayede veri-bağımsız (data-oblivious) 
kuantizasyon mümkün olur.

Yöntem: Gaussian matris → QR ayrıştırma → Q ortogonal matris
"""

import numpy as np
from typing import Optional, Dict, Tuple
from functools import lru_cache


# Global cache: (d, seed) → rotation matrix
_rotation_cache: Dict[Tuple[int, int], np.ndarray] = {}


def generate_rotation_matrix(
    d: int, 
    seed: Optional[int] = None,
    use_cache: bool = True
) -> np.ndarray:
    """
    Haar dağılımlı rastgele ortogonal matris üretir.
    
    Algoritma:
      1. G ~ N(0,1)^{d×d} Gaussian matris oluştur
      2. QR ayrıştırma uygula: G = Q·R
      3. Q'yu Haar-düzeltmesi ile normalize et (R'nin diyagonal işaretleri)
    
    Args:
        d: Vektör boyutu (head_dim, genellikle 128 veya 256)
        seed: Tekrarlanabilirlik için seed
        use_cache: Aynı (d, seed) için cache kullan
        
    Returns:
        Π: (d, d) ortogonal matris, Π^T·Π = I
    """
    cache_key = (d, seed if seed is not None else -1)
    
    if use_cache and cache_key in _rotation_cache:
        return _rotation_cache[cache_key]
    
    rng = np.random.default_rng(seed)
    
    # Adım 1: Gaussian rastgele matris
    G = rng.standard_normal((d, d))
    
    # Adım 2: QR ayrıştırma
    Q, R = np.linalg.qr(G)
    
    # Adım 3: Haar düzeltmesi
    # R'nin diyagonal elemanlarının işaretlerini Q'ya uygula
    # Bu, Q'nun Haar ölçüsüne göre uniform dağılmasını sağlar
    signs = np.sign(np.diag(R))
    signs[signs == 0] = 1.0  # sıfır durumunu ele al
    Q = Q * signs[np.newaxis, :]
    
    if use_cache:
        _rotation_cache[cache_key] = Q
    
    return Q


def rotate_vector(x: np.ndarray, Pi: np.ndarray) -> np.ndarray:
    """
    Vektörü (veya vektör batch'ini) döndürür.
    
    Args:
        x: (..., d) boyutlu girdi vektörü/vektörleri
        Pi: (d, d) ortogonal rotasyon matrisi
        
    Returns:
        x_rot: (..., d) döndürülmüş vektör, x_rot = x @ Pi^T
    """
    return x @ Pi.T


def inverse_rotate_vector(x_rot: np.ndarray, Pi: np.ndarray) -> np.ndarray:
    """
    Rotasyonu geri alır (ortogonal matris → ters = transpoz).
    
    Args:
        x_rot: (..., d) döndürülmüş vektör
        Pi: (d, d) ortogonal rotasyon matrisi
        
    Returns:
        x: (..., d) orijinal uzaydaki vektör, x = x_rot @ Pi
    """
    return x_rot @ Pi


def clear_rotation_cache():
    """Rotasyon cache'ini temizler."""
    _rotation_cache.clear()
