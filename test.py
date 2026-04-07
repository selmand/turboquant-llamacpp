"""
TurboQuant Hızlı Test
======================
Kurulumu doğrula. Çalıştırma: python test.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

def main():
    print("🧪 TurboQuant Hızlı Test")
    print("━" * 40)
    
    passed = 0
    failed = 0
    
    # Test 1: GPU algılama
    try:
        from turboquant.gpu_detect import detect_gpu
        gpu = detect_gpu()
        assert gpu is not None
        print(f"  ✅ GPU algılama: {gpu.name} ({gpu.backend})")
        passed += 1
    except Exception as e:
        print(f"  ❌ GPU algılama: {e}")
        failed += 1
    
    # Test 2: Rotasyon
    try:
        from turboquant.rotation import generate_rotation_matrix
        Pi = generate_rotation_matrix(128, seed=42)
        err = np.max(np.abs(Pi.T @ Pi - np.eye(128)))
        assert err < 1e-10
        print(f"  ✅ Ortogonal rotasyon (hata: {err:.2e})")
        passed += 1
    except Exception as e:
        print(f"  ❌ Rotasyon: {e}")
        failed += 1
    
    # Test 3: Lloyd-Max
    try:
        from turboquant.lloyd_max import LloydMaxQuantizer
        q = LloydMaxQuantizer(d=128, bits=3)
        assert len(q.centroids) == 8
        print(f"  ✅ Lloyd-Max codebook (8 seviye)")
        passed += 1
    except Exception as e:
        print(f"  ❌ Lloyd-Max: {e}")
        failed += 1
    
    # Test 4: Kuantizasyon round-trip
    try:
        from turboquant.quantizer import TurboQuantMSE
        tq = TurboQuantMSE(d=128, bits=4, seed=42)
        x = np.random.randn(128)
        qv = tq.quantize(x)
        x_hat = tq.dequantize(qv)
        mse = np.mean((x - x_hat) ** 2)
        assert mse < 0.1
        print(f"  ✅ TQ4 round-trip MSE: {mse:.6f}")
        passed += 1
    except Exception as e:
        print(f"  ❌ Kuantizasyon: {e}")
        failed += 1
    
    # Test 5: Sıkıştırma oranı
    try:
        ratio = tq.compression_ratio()
        assert 3.5 < ratio < 4.5
        print(f"  ✅ Sıkıştırma oranı: {ratio:.1f}×")
        passed += 1
    except Exception as e:
        print(f"  ❌ Sıkıştırma: {e}")
        failed += 1
    
    # Test 6: Bit packing
    try:
        from turboquant.quantizer import pack_indices, unpack_indices
        idx = np.array([0,1,2,3,4,5,6,7] * 16, dtype=np.uint8)
        packed = pack_indices(idx, 3)
        unpacked = unpack_indices(packed, 3, 128)
        assert np.all(idx == unpacked)
        print(f"  ✅ Bit packing (48 byte)")
        passed += 1
    except Exception as e:
        print(f"  ❌ Bit packing: {e}")
        failed += 1
    
    # Test 7: Config dosyası
    try:
        from pathlib import Path
        config_exists = (Path(__file__).parent / "config.json").exists()
        if config_exists:
            print(f"  ✅ config.json mevcut")
        else:
            print(f"  ⚠️  config.json yok (install.py çalıştır)")
        passed += 1
    except Exception as e:
        print(f"  ❌ Config: {e}")
        failed += 1
    
    # Sonuç
    print()
    print(f"  Sonuç: {passed} geçti, {failed} kaldı")
    
    if failed == 0:
        print("  🎉 Tüm testler başarılı!")
    else:
        print("  ⚠️  Bazı testler başarısız oldu.")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
