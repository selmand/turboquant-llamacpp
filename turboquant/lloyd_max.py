"""
Lloyd-Max Optimal Skaler Kuantizer
===================================
Beta dağılımı için MSE-optimal kuantizasyon.

Rotasyon sonrası her koordinat Beta((d-1)/2, (d-1)/2) dağılımına uyar.
Bu dağılım [-1, 1] aralığında tanımlıdır ve yüksek boyutlarda 
Gaussian N(0, 1/d)'ye yakınsar.

Lloyd-Max algoritması, verilen bir olasılık dağılımı ve bit bütçesi 
(b bit = 2^b seviye) için MSE'yi minimize eden bucket sınırlarını 
ve centroid'leri bulur.
"""

import numpy as np
from scipy import special, integrate
from typing import Tuple, Dict, Optional
import json
import os

# Codebook cache: (d, b) → (boundaries, centroids)
_codebook_cache: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}


def beta_pdf(x: np.ndarray, d: int) -> np.ndarray:
    """
    Rotasyon sonrası koordinat dağılımının PDF'i.
    
    Kağıttan:
      f_X(x) = Γ(d/2) / (√π · Γ((d-1)/2)) · (1 - x²)^((d-3)/2)
      x ∈ [-1, 1]
    
    Bu, Beta((d-1)/2, (d-1)/2) dağılımının [-1, 1]'e ölçeklenmiş halidir.
    """
    if d <= 2:
        # d=1 veya d=2 için uniform
        return np.where(np.abs(x) <= 1, 0.5 * np.ones_like(x), np.zeros_like(x))
    
    alpha = (d - 1) / 2.0
    
    # Log-uzayda hesapla (numerik stabilite)
    log_norm = (
        special.gammaln(d / 2.0) 
        - 0.5 * np.log(np.pi) 
        - special.gammaln((d - 1) / 2.0)
    )
    
    # (1 - x²)^((d-3)/2) hesapla
    mask = np.abs(x) < 1.0
    result = np.zeros_like(x, dtype=np.float64)
    
    if d > 3:
        exponent = (d - 3) / 2.0
        result[mask] = np.exp(
            log_norm + exponent * np.log(1.0 - x[mask]**2)
        )
    elif d == 3:
        result[mask] = np.exp(log_norm)
    
    return result


def lloyd_max_iteration(
    boundaries: np.ndarray,
    centroids: np.ndarray,
    d: int,
    n_levels: int
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Lloyd-Max algoritmasının tek bir iterasyonu.
    
    1. Centroid güncelle: Her bucket için ağırlıklı ortalama
    2. Sınır güncelle: İki komşu centroid'in ortası
    
    Returns:
        Yeni boundaries, yeni centroids, toplam MSE distorsiyonu
    """
    new_centroids = np.zeros(n_levels)
    total_distortion = 0.0
    
    # Her bucket için centroid güncelle
    for i in range(n_levels):
        lo = boundaries[i]
        hi = boundaries[i + 1]
        
        if hi - lo < 1e-15:
            new_centroids[i] = (lo + hi) / 2
            continue
        
        # E[X | lo < X < hi] = ∫ x·f(x)dx / ∫ f(x)dx
        numerator, _ = integrate.quad(
            lambda x: x * beta_pdf(np.array([x]), d)[0],
            lo, hi, limit=100
        )
        denominator, _ = integrate.quad(
            lambda x: beta_pdf(np.array([x]), d)[0],
            lo, hi, limit=100
        )
        
        if abs(denominator) > 1e-15:
            new_centroids[i] = numerator / denominator
        else:
            new_centroids[i] = (lo + hi) / 2
        
        # Distorsiyon: E[(X - c_i)² | lo < X < hi] · P(lo < X < hi)
        dist, _ = integrate.quad(
            lambda x: (x - new_centroids[i])**2 * beta_pdf(np.array([x]), d)[0],
            lo, hi, limit=100
        )
        total_distortion += dist
    
    # Sınırları güncelle: komşu centroid'lerin ortası
    new_boundaries = boundaries.copy()
    for i in range(1, n_levels):
        new_boundaries[i] = (new_centroids[i - 1] + new_centroids[i]) / 2
    
    return new_boundaries, new_centroids, total_distortion


def compute_lloyd_max_codebook(
    d: int, 
    bits: int, 
    max_iter: int = 200,
    tol: float = 1e-10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Lloyd-Max algoritması ile optimal codebook hesaplar.
    
    Args:
        d: Vektör boyutu (head_dim)
        bits: Bit bütçesi (2, 3 veya 4)
        max_iter: Maksimum iterasyon sayısı
        tol: Yakınsama toleransı (MSE değişimi)
        
    Returns:
        boundaries: (n_levels + 1,) sınır değerleri
        centroids: (n_levels,) centroid (yeniden yapılandırma) değerleri
    """
    n_levels = 2 ** bits
    
    # Uniform başlangıç
    boundaries = np.linspace(-1.0, 1.0, n_levels + 1)
    centroids = np.zeros(n_levels)
    for i in range(n_levels):
        centroids[i] = (boundaries[i] + boundaries[i + 1]) / 2
    
    prev_distortion = float('inf')
    
    for iteration in range(max_iter):
        boundaries, centroids, distortion = lloyd_max_iteration(
            boundaries, centroids, d, n_levels
        )
        
        # Yakınsama kontrolü
        if abs(prev_distortion - distortion) < tol:
            break
        prev_distortion = distortion
    
    return boundaries, centroids


class LloydMaxQuantizer:
    """
    Lloyd-Max skaler kuantizer sınıfı.
    
    Kullanım:
        q = LloydMaxQuantizer(d=128, bits=3)
        indices = q.quantize(rotated_coords)      # float → int index
        restored = q.dequantize(indices)           # int index → float
    """
    
    def __init__(self, d: int, bits: int, precomputed: bool = True):
        self.d = d
        self.bits = bits
        self.n_levels = 2 ** bits
        
        cache_key = (d, bits)
        if cache_key in _codebook_cache:
            self.boundaries, self.centroids = _codebook_cache[cache_key]
        else:
            self.boundaries, self.centroids = compute_lloyd_max_codebook(d, bits)
            _codebook_cache[cache_key] = (self.boundaries, self.centroids)
    
    def quantize(self, x: np.ndarray) -> np.ndarray:
        """
        Float koordinatları → indeks (0 .. n_levels-1).
        
        Her koordinat, en yakın centroid'in indeksine eşlenir.
        Bu, sınırlar kullanılarak hızlı bir şekilde yapılır.
        """
        # np.searchsorted ile hızlı bucket arama
        # boundaries[0]=-1, boundaries[-1]=1
        indices = np.searchsorted(self.boundaries[1:-1], x)
        indices = np.clip(indices, 0, self.n_levels - 1)
        return indices.astype(np.uint8)
    
    def dequantize(self, indices: np.ndarray) -> np.ndarray:
        """İndeksleri → centroid değerlerine geri çevirir."""
        return self.centroids[indices]
    
    def get_mse(self) -> float:
        """Bu codebook'un teorik MSE'sini hesaplar."""
        total = 0.0
        for i in range(self.n_levels):
            lo, hi = self.boundaries[i], self.boundaries[i + 1]
            dist, _ = integrate.quad(
                lambda x: (x - self.centroids[i])**2 * beta_pdf(np.array([x]), self.d)[0],
                lo, hi, limit=100
            )
            total += dist
        return total


def precompute_codebooks(
    dimensions: list = [128, 256],
    bit_widths: list = [2, 3, 4]
) -> Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]]:
    """
    Sık kullanılan (d, b) çiftleri için codebook'ları önceden hesaplar.
    
    Tipik kullanım: uygulama başlangıcında bir kez çalıştırılır.
    """
    results = {}
    for d in dimensions:
        for b in bit_widths:
            print(f"  Codebook hesaplanıyor: d={d}, bits={b}...")
            boundaries, centroids = compute_lloyd_max_codebook(d, b)
            results[(d, b)] = (boundaries, centroids)
            _codebook_cache[(d, b)] = (boundaries, centroids)
    return results


def save_codebooks(codebooks: dict, path: str):
    """Codebook'ları JSON olarak diske kaydeder."""
    serializable = {}
    for (d, b), (boundaries, centroids) in codebooks.items():
        key = f"d{d}_b{b}"
        serializable[key] = {
            "d": d,
            "bits": b,
            "boundaries": boundaries.tolist(),
            "centroids": centroids.tolist()
        }
    with open(path, 'w') as f:
        json.dump(serializable, f, indent=2)


def load_codebooks(path: str) -> Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]]:
    """JSON'dan codebook'ları yükler ve cache'e ekler."""
    with open(path, 'r') as f:
        data = json.load(f)
    
    codebooks = {}
    for key, val in data.items():
        d, b = val["d"], val["bits"]
        boundaries = np.array(val["boundaries"])
        centroids = np.array(val["centroids"])
        codebooks[(d, b)] = (boundaries, centroids)
        _codebook_cache[(d, b)] = (boundaries, centroids)
    
    return codebooks
