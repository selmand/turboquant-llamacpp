"""
TurboQuant Diagnostik
=======================
Sorun yaşadığında bunu çalıştır:
    python diagnose.py

Tam sistem raporu üretir: OS, araçlar, izinler, disk, GPU.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path
from turboquant.errors import (
    generate_diagnostic_report, validate_project_directory,
    check_permissions, TQError, ErrorCode, handle_error
)


def main():
    project_dir = str(Path(__file__).parent.resolve())
    
    print()
    print(generate_diagnostic_report(project_dir))
    
    print()
    print("🔍 Dizin izin testi yapılıyor...")
    issues = validate_project_directory(project_dir)
    
    if issues:
        print(f"\n  ⚠️  {len(issues)} sorun bulundu:")
        for code, msg in issues:
            print(f"    [{code}] {msg}")
    else:
        print("  ✅ Tüm dizin izinleri sorunsuz.")
    
    # GPU testi
    print()
    print("🔍 GPU testi...")
    try:
        from turboquant.gpu_detect import detect_gpu
        gpu = detect_gpu()
        print(f"  ✅ GPU: {gpu.name} ({gpu.backend})")
    except Exception as e:
        handle_error(e, "GPU algılama")
    
    # Algoritma testi
    print()
    print("🔍 Algoritma testi...")
    try:
        import numpy as np
        from turboquant.quantizer import TurboQuantMSE
        tq = TurboQuantMSE(d=128, bits=3, seed=42)
        x = np.random.randn(128)
        qv = tq.quantize(x)
        x_hat = tq.dequantize(qv)
        mse = float(np.mean((x - x_hat) ** 2))
        print(f"  ✅ TQ3 round-trip MSE: {mse:.6f}")
    except Exception as e:
        handle_error(e, "Kuantizasyon testi")
    
    # Config kontrolü
    print()
    config_path = Path(project_dir) / "config.json"
    if config_path.exists():
        print("🔍 config.json kontrolü...")
        try:
            import json
            with open(config_path) as f:
                config = json.load(f)
            server = config.get("server_path", "")
            model = config.get("model_path", "")
            print(f"  Server: {'✅' if server and os.path.exists(server) else '❌'} {server}")
            print(f"  Model:  {'✅' if model and os.path.exists(model) else '❌'} {model}")
        except Exception as e:
            handle_error(e, "config.json okuma")
    else:
        print("  ⚠️  config.json yok — henüz install.py çalıştırılmamış")
    
    print()
    print("Raporu kopyalayıp GitHub issue'ya yapıştırabilirsin.")
    print()


if __name__ == "__main__":
    main()
