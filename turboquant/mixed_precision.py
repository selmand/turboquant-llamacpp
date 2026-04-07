"""
Outlier-Aware Mixed-Precision Kuantizasyon
============================================
Kağıttan (Section 4.2):
  Kanalları outlier ve non-outlier olarak ayır.
  Outlier kanallara daha yüksek bit genişliği uygula.
  
  Örnek: 2.5-bit setup
    - 32 outlier kanal → 3-bit
    - 96 normal kanal  → 2-bit
    - Efektif: (32×3 + 96×2) / 128 = 2.5 bit/dim
    
  Örnek: 3.5-bit setup
    - 64 outlier kanal → 4-bit
    - 64 normal kanal  → 3-bit
    - Efektif: (64×4 + 64×3) / 128 = 3.5 bit/dim

Not: Rotasyon sonrası tüm kanallar istatistiksel olarak eşdeğer (i.i.d.)
olduğu için, kanal ayırma sabit ve veri-bağımsızdır.
Token başına ek overhead yoktur.
"""

import numpy as np
from typing import Optional, Tuple, NamedTuple
from .rotation import generate_rotation_matrix, rotate_vector, inverse_rotate_vector
from .lloyd_max import LloydMaxQuantizer
from .qjl import QJL


class MixedQuantizedVector(NamedTuple):
    """Mixed-precision kuantize edilmiş vektör."""
    norm: float
    outlier_indices: np.ndarray    # Yüksek bit (uint8)
    normal_indices: np.ndarray     # Düşük bit (uint8)
    outlier_channels: np.ndarray   # Hangi kanallar outlier (sabit)
    qjl_signs: Optional[np.ndarray] = None


class TurboQuantMixed:
    """
    Mixed-Precision TurboQuant.
    
    Kanalları ikiye ayırır:
      - outlier_count kanal → high_bits ile kuantize
      - geri kalan kanal   → low_bits ile kuantize
    
    Desteklenen modlar:
      - 2.5-bit: 32 kanal@3-bit + 96 kanal@2-bit (d=128)
      - 3.5-bit: 64 kanal@4-bit + 64 kanal@3-bit (d=128)
      - Özel: İstediğin kombinasyon
    """
    
    def __init__(
        self,
        d: int = 128,
        mode: str = "3.5-bit",
        seed: int = 42,
        # veya manuel ayar:
        outlier_count: Optional[int] = None,
        high_bits: Optional[int] = None,
        low_bits: Optional[int] = None
    ):
        self.d = d
        self.seed = seed
        self.mode = mode
        
        # Mod yapılandırması
        if outlier_count is not None and high_bits is not None and low_bits is not None:
            self.outlier_count = outlier_count
            self.high_bits = high_bits
            self.low_bits = low_bits
        elif mode == "2.5-bit":
            self.outlier_count = d // 4      # 32 for d=128
            self.high_bits = 3
            self.low_bits = 2
        elif mode == "3.5-bit":
            self.outlier_count = d // 2      # 64 for d=128
            self.high_bits = 4
            self.low_bits = 3
        elif mode == "3-bit":
            # Uniform 3-bit (karşılaştırma için)
            self.outlier_count = 0
            self.high_bits = 3
            self.low_bits = 3
        elif mode == "4-bit":
            self.outlier_count = 0
            self.high_bits = 4
            self.low_bits = 4
        else:
            raise ValueError(f"Bilinmeyen mod: {mode}. '2.5-bit', '3.5-bit', '3-bit', '4-bit' kullan.")
        
        self.normal_count = d - self.outlier_count
        
        # Efektif bit genişliği
        if d > 0:
            self.effective_bits = (
                self.outlier_count * self.high_bits + 
                self.normal_count * self.low_bits
            ) / d
        else:
            self.effective_bits = 0
        
        # Sabit kanal indeksleri (rotasyon sonrası eşdeğer, sıralı seçim yeterli)
        self.outlier_channels = np.arange(self.outlier_count)
        self.normal_channels = np.arange(self.outlier_count, d)
        
        # Rotasyon matrisi
        self.Pi = generate_rotation_matrix(d, seed=seed)
        
        # İki ayrı Lloyd-Max kuantizer
        if self.outlier_count > 0:
            self.high_quantizer = LloydMaxQuantizer(d, self.high_bits)
        self.low_quantizer = LloydMaxQuantizer(d, self.low_bits)
    
    def quantize(self, x: np.ndarray) -> MixedQuantizedVector:
        """
        Mixed-precision kuantizasyon.
        
        Args:
            x: (d,) girdi vektörü
            
        Returns:
            MixedQuantizedVector
        """
        # 1. Norm kaydet
        norm = float(np.linalg.norm(x))
        if norm < 1e-10:
            return MixedQuantizedVector(
                norm=0.0,
                outlier_indices=np.zeros(self.outlier_count, dtype=np.uint8),
                normal_indices=np.zeros(self.normal_count, dtype=np.uint8),
                outlier_channels=self.outlier_channels
            )
        
        # 2. Normalize ve döndür
        x_normalized = x / norm
        x_rotated = rotate_vector(x_normalized, self.Pi)
        
        # 3. Outlier kanalları yüksek bit ile kuantize et
        if self.outlier_count > 0:
            outlier_coords = x_rotated[self.outlier_channels]
            outlier_indices = self.high_quantizer.quantize(outlier_coords)
        else:
            outlier_indices = np.array([], dtype=np.uint8)
        
        # 4. Normal kanalları düşük bit ile kuantize et
        normal_coords = x_rotated[self.normal_channels]
        normal_indices = self.low_quantizer.quantize(normal_coords)
        
        return MixedQuantizedVector(
            norm=norm,
            outlier_indices=outlier_indices,
            normal_indices=normal_indices,
            outlier_channels=self.outlier_channels
        )
    
    def dequantize(self, mqv: MixedQuantizedVector) -> np.ndarray:
        """Mixed-precision vektörü geri yapılandırır."""
        if mqv.norm < 1e-10:
            return np.zeros(self.d)
        
        # Rotated uzayda geri yapılandır
        x_rotated = np.zeros(self.d)
        
        if self.outlier_count > 0:
            x_rotated[self.outlier_channels] = self.high_quantizer.dequantize(mqv.outlier_indices)
        x_rotated[self.normal_channels] = self.low_quantizer.dequantize(mqv.normal_indices)
        
        # Ters rotasyon + norm
        x_hat = inverse_rotate_vector(x_rotated, self.Pi) * mqv.norm
        
        return x_hat
    
    def get_mse(self, x: np.ndarray) -> float:
        """Round-trip MSE."""
        mqv = self.quantize(x)
        x_hat = self.dequantize(mqv)
        return float(np.mean((x - x_hat) ** 2))
    
    def compression_ratio(self) -> float:
        """FP16'ya göre sıkıştırma oranı."""
        fp16_bytes = self.d * 2  # 256 bytes for d=128
        
        # Packed bytes
        outlier_bits_total = self.outlier_count * self.high_bits
        normal_bits_total = self.normal_count * self.low_bits
        total_bits = outlier_bits_total + normal_bits_total
        packed_bytes = (total_bits + 7) // 8
        
        # + 4 byte norm
        tq_bytes = 4 + packed_bytes
        
        return fp16_bytes / tq_bytes
    
    def memory_per_vector(self) -> int:
        """Vektör başına byte."""
        total_bits = (self.outlier_count * self.high_bits + 
                      self.normal_count * self.low_bits)
        return 4 + (total_bits + 7) // 8
    
    def info(self) -> str:
        """Yapılandırma bilgisi."""
        return (
            f"TurboQuant Mixed [{self.mode}]\n"
            f"  d={self.d}, efektif={self.effective_bits:.1f} bit/dim\n"
            f"  Outlier: {self.outlier_count} kanal × {self.high_bits}-bit\n"
            f"  Normal:  {self.normal_count} kanal × {self.low_bits}-bit\n"
            f"  Sıkıştırma: {self.compression_ratio():.1f}×\n"
            f"  Bellek: {self.memory_per_vector()} byte/vektör (vs 256 FP16)"
        )


# Hazır preset'ler
def create_tq25(d: int = 128, seed: int = 42) -> TurboQuantMixed:
    """2.5-bit mixed preset (en agresif sıkıştırma)."""
    return TurboQuantMixed(d=d, mode="2.5-bit", seed=seed)

def create_tq35(d: int = 128, seed: int = 42) -> TurboQuantMixed:
    """3.5-bit mixed preset (en iyi kalite/sıkıştırma dengesi)."""
    return TurboQuantMixed(d=d, mode="3.5-bit", seed=seed)

def create_asymmetric_kv(
    d: int = 128,
    key_mode: str = "3.5-bit",
    value_mode: str = "4-bit",
    seed: int = 42
) -> Tuple[TurboQuantMixed, TurboQuantMixed]:
    """
    Asimetrik K/V kuantizasyon.
    
    Topluluk bulgularından:
      - Key'ler daha hassas (norm disparitesi 4-182×)
      - Value'lar daha toleranslı
      
    Önerilen: key@3.5-bit + value@4-bit
    Agresif:  key@3-bit   + value@3.5-bit
    """
    key_tq = TurboQuantMixed(d=d, mode=key_mode, seed=seed)
    val_tq = TurboQuantMixed(d=d, mode=value_mode, seed=seed + 500)
    return key_tq, val_tq
