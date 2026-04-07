"""
GPU Otomatik Algılama
======================
Sistemdeki GPU'yu tespit eder ve en uygun backend'i seçer.

Desteklenen:
  - NVIDIA (CUDA) → -DGGML_CUDA=ON
  - AMD (Vulkan/ROCm) → -DGGML_VULKAN=ON
  - Intel (Vulkan) → -DGGML_VULKAN=ON  
  - Apple Silicon (Metal) → -DGGML_METAL=ON
  - CPU only → fallback
"""

import subprocess
import platform
import os
import re
import shutil
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class GPUInfo:
    """Algılanan GPU bilgisi."""
    name: str
    vendor: str              # "nvidia", "amd", "intel", "apple", "unknown"
    vram_mb: int             # VRAM (MB)
    backend: str             # "cuda", "vulkan", "metal", "cpu"
    cmake_flags: List[str]   # CMake derleme flag'leri
    cuda_version: Optional[str] = None
    driver_version: Optional[str] = None
    compute_capability: Optional[str] = None
    recommended_layers: int = 99  # n-gpu-layers önerisi


def run_cmd(cmd: str, timeout: int = 10) -> Optional[str]:
    """Komutu çalıştır, çıktıyı döndür. Hata varsa None."""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, 
            text=True, timeout=timeout
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return None
    except Exception:
        return None


def detect_nvidia() -> Optional[GPUInfo]:
    """NVIDIA GPU ve CUDA algıla."""
    # nvidia-smi var mı?
    if not shutil.which("nvidia-smi"):
        return None
    
    # GPU adı ve VRAM
    output = run_cmd(
        'nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap '
        '--format=csv,noheader,nounits'
    )
    if not output:
        return None
    
    parts = output.split('\n')[0].split(',')
    if len(parts) < 4:
        return None
    
    name = parts[0].strip()
    vram_mb = int(float(parts[1].strip()))
    driver = parts[2].strip()
    compute = parts[3].strip()
    
    # CUDA version
    cuda_ver = None
    nvcc_output = run_cmd("nvcc --version")
    if nvcc_output:
        match = re.search(r'release (\d+\.\d+)', nvcc_output)
        if match:
            cuda_ver = match.group(1)
    else:
        # nvidia-smi'den CUDA version
        smi_output = run_cmd("nvidia-smi")
        if smi_output:
            match = re.search(r'CUDA Version:\s*(\d+\.\d+)', smi_output)
            if match:
                cuda_ver = match.group(1)
    
    # GPU layers önerisi (VRAM'e göre)
    if vram_mb >= 80000:
        layers = 99  # A100/H100: her şeyi GPU'ya yükle
    elif vram_mb >= 24000:
        layers = 99  # RTX 3090/4090
    elif vram_mb >= 16000:
        layers = 40  # RTX 4080/A4000
    elif vram_mb >= 12000:
        layers = 35  # RTX 3060 12GB
    elif vram_mb >= 8000:
        layers = 28  # RTX 3060 8GB / RTX 4060
    else:
        layers = 20
    
    cmake_flags = ["-DGGML_CUDA=ON"]
    
    # CUDA arch flag (compute capability'ye göre)
    if compute:
        cc = compute.replace(".", "")
        cmake_flags.append(f"-DCMAKE_CUDA_ARCHITECTURES={cc}")
    
    return GPUInfo(
        name=name,
        vendor="nvidia",
        vram_mb=vram_mb,
        backend="cuda",
        cmake_flags=cmake_flags,
        cuda_version=cuda_ver,
        driver_version=driver,
        compute_capability=compute,
        recommended_layers=layers
    )


def detect_amd() -> Optional[GPUInfo]:
    """AMD GPU algıla (Vulkan veya ROCm)."""
    system = platform.system()
    
    # ROCm kontrolü (Linux)
    if system == "Linux":
        rocm_info = run_cmd("rocm-smi --showproductname")
        if rocm_info:
            name = "AMD GPU (ROCm)"
            # rocm-smi'den detay çek
            mem_output = run_cmd("rocm-smi --showmeminfo vram --csv")
            vram_mb = 8000  # varsayılan
            if mem_output:
                match = re.search(r'(\d+)', mem_output)
                if match:
                    vram_mb = int(match.group(1)) // (1024 * 1024)
            
            return GPUInfo(
                name=name, vendor="amd", vram_mb=vram_mb,
                backend="rocm",
                cmake_flags=["-DGGML_HIP=ON"],
                recommended_layers=99
            )
    
    # Vulkan kontrolü (Windows/Linux)
    vulkan_info = run_cmd("vulkaninfo --summary 2>&1")
    if vulkan_info and "AMD" in vulkan_info.upper():
        name_match = re.search(r'deviceName\s*=\s*(.+)', vulkan_info)
        name = name_match.group(1).strip() if name_match else "AMD GPU (Vulkan)"
        
        return GPUInfo(
            name=name, vendor="amd", vram_mb=8000,
            backend="vulkan",
            cmake_flags=["-DGGML_VULKAN=ON"],
            recommended_layers=99
        )
    
    # Windows: DxDiag fallback
    if system == "Windows":
        dxdiag = run_cmd('wmic path win32_videocontroller get name,adapterram /format:csv')
        if dxdiag and "AMD" in dxdiag.upper():
            return GPUInfo(
                name="AMD GPU", vendor="amd", vram_mb=8000,
                backend="vulkan",
                cmake_flags=["-DGGML_VULKAN=ON"],
                recommended_layers=99
            )
    
    return None


def detect_intel() -> Optional[GPUInfo]:
    """Intel GPU algıla (Arc serisi / Vulkan)."""
    system = platform.system()
    
    vulkan_info = run_cmd("vulkaninfo --summary 2>&1")
    if vulkan_info and "INTEL" in vulkan_info.upper():
        name_match = re.search(r'deviceName\s*=\s*(.+)', vulkan_info)
        name = name_match.group(1).strip() if name_match else "Intel GPU (Vulkan)"
        
        return GPUInfo(
            name=name, vendor="intel", vram_mb=8000,
            backend="vulkan",
            cmake_flags=["-DGGML_VULKAN=ON"],
            recommended_layers=35
        )
    
    if system == "Windows":
        dxdiag = run_cmd('wmic path win32_videocontroller get name /format:csv')
        if dxdiag and "ARC" in dxdiag.upper():
            return GPUInfo(
                name="Intel Arc GPU", vendor="intel", vram_mb=8000,
                backend="vulkan",
                cmake_flags=["-DGGML_VULKAN=ON"],
                recommended_layers=35
            )
    
    return None


def detect_apple_silicon() -> Optional[GPUInfo]:
    """Apple Silicon algıla (Metal)."""
    if platform.system() != "Darwin":
        return None
    
    # Apple Silicon kontrol
    chip_info = run_cmd("sysctl -n machdep.cpu.brand_string")
    if not chip_info or "Apple" not in chip_info:
        return None
    
    # Toplam bellek (unified memory)
    mem_output = run_cmd("sysctl -n hw.memsize")
    total_mem_mb = int(mem_output) // (1024 * 1024) if mem_output else 8000
    
    # GPU için kullanılabilir bellek tahmini (%75'i)
    gpu_mem = int(total_mem_mb * 0.75)
    
    # Chip adı
    chip_name = run_cmd("system_profiler SPHardwareDataType | grep 'Chip'")
    name = chip_name.split(":")[-1].strip() if chip_name else "Apple Silicon"
    
    layers = 99 if gpu_mem >= 16000 else 50
    
    return GPUInfo(
        name=name, vendor="apple", vram_mb=gpu_mem,
        backend="metal",
        cmake_flags=["-DGGML_METAL=ON"],
        recommended_layers=layers
    )


def detect_gpu() -> GPUInfo:
    """
    Sistemi tarar ve en uygun GPU'yu döndürür.
    
    Öncelik sırası: NVIDIA > AMD > Intel > Apple > CPU
    """
    print("🔍 GPU algılanıyor...")
    
    # 1. NVIDIA (en yaygın + en iyi desteklenen)
    gpu = detect_nvidia()
    if gpu:
        print(f"  ✅ NVIDIA bulundu: {gpu.name}")
        print(f"     VRAM: {gpu.vram_mb} MB | CUDA: {gpu.cuda_version or '?'}")
        print(f"     Backend: CUDA | Layers: {gpu.recommended_layers}")
        return gpu
    
    # 2. Apple Silicon
    gpu = detect_apple_silicon()
    if gpu:
        print(f"  ✅ Apple Silicon bulundu: {gpu.name}")
        print(f"     Unified Memory: ~{gpu.vram_mb} MB GPU payı")
        print(f"     Backend: Metal | Layers: {gpu.recommended_layers}")
        return gpu
    
    # 3. AMD
    gpu = detect_amd()
    if gpu:
        print(f"  ✅ AMD bulundu: {gpu.name}")
        print(f"     Backend: {gpu.backend.upper()} | Layers: {gpu.recommended_layers}")
        return gpu
    
    # 4. Intel
    gpu = detect_intel()
    if gpu:
        print(f"  ✅ Intel bulundu: {gpu.name}")
        print(f"     Backend: Vulkan | Layers: {gpu.recommended_layers}")
        return gpu
    
    # 5. CPU fallback
    print("  ⚠️  GPU bulunamadı, CPU modu kullanılacak")
    cpu_name = platform.processor() or "Unknown CPU"
    cores = os.cpu_count() or 4
    
    return GPUInfo(
        name=cpu_name, vendor="cpu", vram_mb=0,
        backend="cpu",
        cmake_flags=[],
        recommended_layers=0  # CPU = tüm katmanlar RAM'de
    )


def estimate_max_context(gpu: GPUInfo, model_size_gb: float = 5.0) -> int:
    """GPU'ya göre maksimum context uzunluğu tahmin et."""
    if gpu.vram_mb == 0:
        # CPU: sistem RAM'ine göre (psutil gerektirmeden)
        ram_gb = _get_system_ram_gb()
        available_gb = ram_gb * 0.6 - model_size_gb
    else:
        available_gb = (gpu.vram_mb / 1024) - model_size_gb - 1.5
    
    if available_gb <= 0:
        return 4096
    
    # TQ3.5 ile: ~1GB ≈ 8K context (Llama 8B, 8 KV heads)
    tq_bytes_per_token = 32 * 8 * ((128 * 3.5 / 8 + 4) + (128 * 4.0 / 8 + 4))
    max_tokens = int((available_gb * 1024**3) / tq_bytes_per_token)
    
    # Güvenli üst sınır
    return min(max_tokens, 131072)


def _get_system_ram_gb() -> float:
    """Sistem RAM'ini GB cinsinden döndür — psutil gerektirmez."""
    try:
        if platform.system() == "Darwin":
            # macOS
            output = run_cmd("sysctl -n hw.memsize")
            if output:
                return int(output) / (1024**3)
        elif platform.system() == "Linux":
            # Linux: /proc/meminfo
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        return kb / (1024**2)
        else:
            # Windows
            output = run_cmd("wmic computersystem get totalphysicalmemory /format:csv")
            if output:
                for line in output.strip().split("\n"):
                    parts = line.strip().split(",")
                    if len(parts) >= 2 and parts[-1].isdigit():
                        return int(parts[-1]) / (1024**3)
    except Exception:
        pass
    return 16.0  # Varsayılan tahmin


if __name__ == "__main__":
    gpu = detect_gpu()
    print(f"\n📊 Özet:")
    print(f"  GPU: {gpu.name}")
    print(f"  Backend: {gpu.backend}")
    print(f"  CMake flags: {' '.join(gpu.cmake_flags)}")
    print(f"  Tahmini max context (TQ3.5, 8B model): {estimate_max_context(gpu)//1024}K")
