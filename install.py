"""
TurboQuant Otomatik Kurulum
==============================
Tek komutla her şeyi kurar:
  1. Gerekli araçları kontrol et / kur (git, cmake, compiler)
  2. llama.cpp TurboQuant fork'unu klonla
  3. GPU'yu algıla
  4. Doğru flag'lerle derle
  5. Mevcut GGUF modeli bul veya indir
  6. Yapılandırma dosyası oluştur

Kullanım:
    python install.py
"""

import os
import sys

# ── .bat dosyası Python'a verilmiş mi? ──
_this = os.path.basename(sys.argv[0]) if sys.argv else ""
if _this.endswith(".bat"):
    print()
    print("  ╔══ YANLIŞ KOMUT ══════════════════════════════════")
    print(f"  ║ '{_this}' bir batch dosyası, Python'a verilemez.")
    print("  ║")
    print("  ║ Doğru kullanım:")
    print("  ║   python install.py    ← terminal'de yaz")
    print(f"  ║   {_this}              ← çift tıkla")
    print("  ╚══════════════════════════════════════════════════")
    print()
    sys.exit(1)

import json
import shutil
import platform
import subprocess
import urllib.request
import zipfile
import time
from pathlib import Path
from typing import Optional, Tuple

# sys.path ayarla (gpu_detect modülü için)
PROJECT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_DIR))

from turboquant.errors import (
    TQError, ErrorCode, handle_error,
    ensure_directory, ensure_writable_file, safe_write_file,
    validate_project_directory, check_permissions, generate_diagnostic_report
)

BUILD_DIR = PROJECT_DIR / "llama-cpp-turboquant"
CONFIG_FILE = PROJECT_DIR / "config.json"

# Fork bilgileri (birincil + yedek)
FORK_CANDIDATES = [
    ("https://github.com/TheTom/llama-cpp-turboquant.git", "feature/turboquant-kv-cache"),
    ("https://github.com/ggml-org/llama.cpp.git", "master"),  # Fallback: ana llama.cpp
]

# Varsayılan model (küçük ve hızlı test için)
DEFAULT_MODEL_URL = "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
DEFAULT_MODEL_NAME = "Llama-3.2-3B-Instruct-Q4_K_M.gguf"
MODELS_DIR_NAME = "models"

# ─────────────────────────────────────────────
# Model Kataloğu — VRAM'e göre öneriler
# ─────────────────────────────────────────────

MODEL_CATALOG = [
    # (ad, boyut_gb, min_vram_gb, url, açıklama)
    {
        "name": "Llama 3.2 1B Instruct (Q4_K_M)",
        "file": "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        "size_gb": 0.8,
        "min_vram_gb": 2,
        "url": "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        "desc": "En küçük, hızlı test için",
    },
    {
        "name": "Llama 3.2 3B Instruct (Q4_K_M)",
        "file": "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "size_gb": 2.0,
        "min_vram_gb": 4,
        "url": "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "desc": "Hafif, günlük kullanım",
    },
    {
        "name": "Qwen 2.5 7B Instruct (Q4_K_M)",
        "file": "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        "size_gb": 4.7,
        "min_vram_gb": 8,
        "url": "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_k_m.gguf",
        "desc": "Güçlü, çok dilli, Türkçe iyi",
    },
    {
        "name": "Llama 3.1 8B Instruct (Q4_K_M)",
        "file": "Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        "size_gb": 4.9,
        "min_vram_gb": 8,
        "url": "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        "desc": "Meta'nın en popüler modeli",
    },
    {
        "name": "Mistral 7B Instruct v0.3 (Q4_K_M)",
        "file": "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
        "size_gb": 4.4,
        "min_vram_gb": 8,
        "url": "https://huggingface.co/bartowski/Mistral-7B-Instruct-v0.3-GGUF/resolve/main/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
        "desc": "Hızlı, kod ve İngilizce'de güçlü",
    },
    {
        "name": "Gemma 2 9B Instruct (Q4_K_M)",
        "file": "gemma-2-9b-it-Q4_K_M.gguf",
        "size_gb": 5.4,
        "min_vram_gb": 10,
        "url": "https://huggingface.co/bartowski/gemma-2-9b-it-GGUF/resolve/main/gemma-2-9b-it-Q4_K_M.gguf",
        "desc": "Google, analiz ve reasoning",
    },
]


# ─────────────────────────────────────────────
# Yardımcı fonksiyonlar
# ─────────────────────────────────────────────

def log(msg: str, level: str = "info"):
    icons = {"info": "📌", "ok": "✅", "warn": "⚠️", "error": "❌", "step": "🔧", "download": "📥"}
    icon = icons.get(level, "  ")
    print(f"  {icon}  {msg}")


def run(cmd: str, cwd: Optional[str] = None, check: bool = True, 
        capture: bool = False, timeout: int = 600) -> subprocess.CompletedProcess:
    """Komutu çalıştır, çıktıyı göster."""
    try:
        result = subprocess.run(
            cmd, shell=True, cwd=cwd,
            capture_output=capture, text=True,
            timeout=timeout
        )
        if check and result.returncode != 0:
            if capture:
                log(f"Hata: {result.stderr[:300]}", "error")
            return result
        return result
    except subprocess.TimeoutExpired:
        log(f"Komut zaman aşımına uğradı: {cmd[:60]}...", "error")
        return subprocess.CompletedProcess(cmd, 1)
    except Exception as e:
        log(f"Komut başarısız: {e}", "error")
        return subprocess.CompletedProcess(cmd, 1)


def cmd_exists(cmd: str) -> bool:
    """Komut sistemde var mı? Windows'ta PATH dışı bilinen yolları da tarar."""
    if shutil.which(cmd) is not None:
        return True
    
    # Windows: bilinen kurulum dizinlerini tara
    if is_windows():
        found = _find_in_known_paths(cmd)
        if found:
            return True
    
    return False


def _find_in_known_paths(cmd: str) -> Optional[str]:
    """
    Windows'ta PATH'te olmayan ama kurulu araçları bilinen dizinlerde ara.
    Bulursa PATH'e ekler ve tam yolu döndürür.
    """
    known_locations = {
        "cmake": [
            r"C:\Program Files\CMake\bin",
            r"C:\Program Files (x86)\CMake\bin",
            os.path.expandvars(r"%LOCALAPPDATA%\CMake\bin"),
            os.path.expandvars(r"%ProgramFiles%\CMake\bin"),
        ],
        "git": [
            r"C:\Program Files\Git\bin",
            r"C:\Program Files\Git\cmd",
            r"C:\Program Files (x86)\Git\bin",
            os.path.expandvars(r"%LOCALAPPDATA%\Programs\Git\bin"),
        ],
        "nvcc": [],  # Aşağıda dinamik eklenir
    }
    
    # CUDA: tüm sürümleri tara
    cuda_base = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
    if os.path.isdir(cuda_base):
        for ver_dir in sorted(os.listdir(cuda_base), reverse=True):
            bin_path = os.path.join(cuda_base, ver_dir, "bin")
            if os.path.isdir(bin_path):
                known_locations.setdefault("nvcc", []).append(bin_path)
    
    # Aracı bilinen yollarda ara
    dirs_to_check = known_locations.get(cmd, [])
    
    exe_name = cmd + ".exe" if not cmd.endswith(".exe") else cmd
    
    for directory in dirs_to_check:
        full_path = os.path.join(directory, exe_name)
        if os.path.isfile(full_path):
            # PATH'e ekle (bu oturum için)
            current_path = os.environ.get("PATH", "")
            if directory.lower() not in current_path.lower():
                os.environ["PATH"] = directory + os.pathsep + current_path
                log(f"{cmd} bulundu: {directory} (PATH'e eklendi)", "ok")
            return full_path
    
    return None


def is_windows() -> bool:
    return platform.system() == "Windows"


def is_mac() -> bool:
    return platform.system() == "Darwin"


def is_linux() -> bool:
    return platform.system() == "Linux"


# ─────────────────────────────────────────────
# ADIM 1: Gerekli araçları kontrol et / kur
# ─────────────────────────────────────────────

def check_prerequisites() -> Tuple[bool, list]:
    """Gerekli araçları kontrol et, eksikleri listele."""
    missing = []
    
    # Git
    if not cmd_exists("git"):
        missing.append("git")
    
    # CMake — önce bilinen yollarda ara
    if not cmd_exists("cmake"):
        missing.append("cmake")
    
    # C++ Compiler
    if is_windows():
        # MSVC (Visual Studio Build Tools) veya MinGW
        has_msvc = cmd_exists("cl") or _check_msvc_installed()
        has_mingw = cmd_exists("gcc") or cmd_exists("g++")
        if not has_msvc and not has_mingw:
            missing.append("compiler")
    elif is_mac():
        if not cmd_exists("clang") and not cmd_exists("gcc"):
            missing.append("compiler")
    else:
        if not cmd_exists("g++") and not cmd_exists("clang++"):
            missing.append("compiler")
    
    # NVIDIA: CUDA toolkit (opsiyonel ama önemli)
    has_nvidia_gpu = cmd_exists("nvidia-smi")
    has_cuda = cmd_exists("nvcc")
    if has_nvidia_gpu and not has_cuda:
        missing.append("cuda-toolkit")
    
    return len(missing) == 0, missing


def _check_msvc_installed() -> bool:
    """Visual Studio Build Tools kurulu mu?"""
    # vswhere ile kontrol
    vswhere = r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
    if os.path.exists(vswhere):
        result = run(f'"{vswhere}" -latest -property installationPath', capture=True, check=False)
        if result.returncode == 0 and result.stdout.strip():
            return True
    
    # vcvarsall.bat arama
    common_paths = [
        r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat",
        r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvarsall.bat",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvarsall.bat",
    ]
    return any(os.path.exists(p) for p in common_paths)


def install_prerequisites(missing: list) -> bool:
    """Eksik araçları otomatik kur."""
    system = platform.system()
    success = True
    
    for tool in missing:
        if tool == "git":
            success &= _install_git(system)
        elif tool == "cmake":
            success &= _install_cmake(system)
        elif tool == "compiler":
            success &= _install_compiler(system)
        elif tool == "cuda-toolkit":
            # Önce bilinen yollarda ara
            found = _find_in_known_paths("nvcc") if is_windows() else None
            if found:
                log(f"CUDA Toolkit bulundu: {found}", "ok")
            else:
                log("CUDA Toolkit eksik. NVIDIA GPU algılandı ama nvcc bulunamadı.", "warn")
                log("İndirmek için: https://developer.nvidia.com/cuda-downloads", "info")
                log("CUDA olmadan devam edilecek (Vulkan veya CPU backend).", "info")
            # CUDA olmadan da devam edilebilir
    
    return success


def _install_git(system: str) -> bool:
    if system == "Windows":
        if cmd_exists("winget"):
            log("Git kuruluyor (winget)...", "step")
            r = run("winget install --id Git.Git -e --source winget --accept-package-agreements --accept-source-agreements", check=False)
            if r.returncode == 0:
                log("Git kuruldu", "ok")
                return True
        log("Git bulunamadı. İndir: https://git-scm.com/download/win", "error")
        return False
    elif system == "Darwin":
        log("Git kuruluyor (xcode-select)...", "step")
        run("xcode-select --install", check=False)
        return True
    else:
        log("Git kuruluyor (apt)...", "step")
        r = run("sudo apt-get install -y git", check=False)
        return r.returncode == 0


def _install_cmake(system: str) -> bool:
    if system == "Windows":
        # Winget dene
        if cmd_exists("winget"):
            log("CMake kuruluyor (winget)...", "step")
            r = run("winget install --id Kitware.CMake -e --source winget --accept-package-agreements --accept-source-agreements", check=False, capture=True)
        
        # Winget sonrası veya "already installed" durumunda PATH'te ara
        found = _find_in_known_paths("cmake")
        if found:
            log(f"CMake bulundu: {found}", "ok")
            return True
        
        # Son çare: where komutu ile ara
        where_result = run("where cmake", capture=True, check=False)
        if where_result.returncode == 0 and where_result.stdout.strip():
            cmake_dir = os.path.dirname(where_result.stdout.strip().split("\n")[0])
            os.environ["PATH"] = cmake_dir + os.pathsep + os.environ.get("PATH", "")
            log(f"CMake bulundu: {cmake_dir}", "ok")
            return True
        
        log("CMake kurulu ama PATH'te bulunamıyor.", "error")
        log("Çözüm seçenekleri:", "info")
        log("  1. Terminal'i kapatıp yeniden aç (PATH güncellenir)", "info")
        log("  2. Veya CMake'i manuel indir: https://cmake.org/download", "info")
        log("     Kurulumda 'Add CMake to PATH' seçeneğini işaretle", "info")
        return False
    elif system == "Darwin":
        if cmd_exists("brew"):
            run("brew install cmake", check=False)
            return cmd_exists("cmake")
        log("CMake kuruluyor: brew install cmake", "info")
        return False
    else:
        r = run("sudo apt-get install -y cmake", check=False)
        return r.returncode == 0


def _install_compiler(system: str) -> bool:
    if system == "Windows":
        log("C++ derleyicisi bulunamadı.", "warn")
        log("", "info")
        log("Otomatik kurulum deneniyor (Visual Studio Build Tools)...", "step")
        
        if cmd_exists("winget"):
            r = run(
                "winget install --id Microsoft.VisualStudio.2022.BuildTools "
                "-e --source winget --accept-package-agreements --accept-source-agreements "
                "--override \"--add Microsoft.VisualStudio.Workload.VCTools --passive --wait\"",
                check=False,
                timeout=1200  # 20 dakika (büyük indirme)
            )
            if r.returncode == 0:
                log("Visual Studio Build Tools kuruldu", "ok")
                log("⚠️  Terminal'i kapatıp yeniden aç, sonra tekrar çalıştır", "warn")
                return True
        
        log("Manuel kurulum gerekli:", "error")
        log("  1. https://visualstudio.microsoft.com/visual-cpp-build-tools/ adresine git", "info")
        log("  2. 'Build Tools for Visual Studio 2022' indir", "info")
        log("  3. 'Desktop development with C++' seç ve kur", "info")
        log("  4. Terminal'i kapatıp yeniden aç", "info")
        return False
    
    elif system == "Darwin":
        run("xcode-select --install", check=False)
        return True
    
    else:
        r = run("sudo apt-get install -y build-essential", check=False)
        return r.returncode == 0


# ─────────────────────────────────────────────
# ADIM 2: Fork'u klonla
# ─────────────────────────────────────────────

def clone_or_update_fork() -> Tuple[bool, bool]:
    """
    TurboQuant llama.cpp fork'unu klonla veya güncelle.
    
    Returns:
        (success, is_turboquant_fork) — fork mu yoksa ana llama.cpp mi
    """
    if BUILD_DIR.exists():
        log("Fork zaten mevcut, güncelleniyor...", "step")
        r = run("git pull", cwd=str(BUILD_DIR), check=False, capture=True)
        if r.returncode == 0:
            log("Fork güncellendi", "ok")
            # TurboQuant fork mu kontrol et
            has_tq = _check_turboquant_support()
            return True, has_tq
        else:
            log("Güncelleme başarısız, yeniden klonlanıyor...", "warn")
            shutil.rmtree(BUILD_DIR, ignore_errors=True)
    
    for repo_url, branch in FORK_CANDIDATES:
        log(f"Deneniyor: {repo_url} ({branch})", "download")
        r = run(
            f'git clone --branch {branch} --depth 1 "{repo_url}" "{BUILD_DIR}"',
            check=False, capture=True, timeout=300
        )
        
        if r.returncode == 0:
            is_tq = "turboquant" in repo_url.lower()
            if is_tq:
                log("TurboQuant fork klonlandı", "ok")
            else:
                log("Ana llama.cpp klonlandı (TurboQuant fork bulunamadı)", "warn")
                log("Not: --cache-type-k turbo3 henüz desteklenmeyebilir", "warn")
                log("Alternatif: q8_0/q4_0 KV cache kullanılacak", "info")
            return True, is_tq
        else:
            log(f"  Bu repo başarısız, sonraki deneniyor...", "warn")
            shutil.rmtree(BUILD_DIR, ignore_errors=True)
    
    log("Hiçbir repo klonlanamadı! İnternet bağlantısını kontrol et.", "error")
    return False, False


def _check_turboquant_support() -> bool:
    """Klonlanan repo TurboQuant cache tiplerini destekliyor mu?"""
    # Kaynak kodda turbo3/turbo4 aramak
    for pattern in ["turbo3", "turbo4", "GGML_TYPE_TQ"]:
        r = run(
            f'git grep -l "{pattern}"',
            cwd=str(BUILD_DIR), check=False, capture=True
        )
        if r.returncode == 0 and r.stdout.strip():
            return True
    return False


# ─────────────────────────────────────────────
# ADIM 3: GPU algıla ve derle
# ─────────────────────────────────────────────

def compile_llama(gpu_info) -> bool:
    """llama.cpp'yi GPU'ya uygun flag'lerle derle."""
    build_path = BUILD_DIR / "build"
    
    # Temiz build
    if build_path.exists():
        shutil.rmtree(build_path, ignore_errors=True)
    
    # CMake flags
    cmake_flags = [
        "-DCMAKE_BUILD_TYPE=Release",
        "-DLLAMA_BUILD_SERVER=ON",
    ] + gpu_info.cmake_flags
    
    # Windows: generator seç
    generator = ""
    if is_windows():
        if _check_msvc_installed():
            generator = '-G "Visual Studio 17 2022"'
        elif cmd_exists("ninja"):
            generator = '-G "Ninja"'
        # else: default generator
    
    cmake_cmd = f'cmake -B build {generator} {" ".join(cmake_flags)}'
    
    log(f"CMake yapılandırılıyor ({gpu_info.backend.upper()})...", "step")
    log(f"  Komut: {cmake_cmd}", "info")
    
    r = run(cmake_cmd, cwd=str(BUILD_DIR), check=False, timeout=120)
    if r.returncode != 0:
        log("CMake yapılandırma başarısız!", "error")
        
        # Fallback: GPU flag'leri olmadan dene
        if gpu_info.backend != "cpu":
            log("GPU desteği olmadan yeniden deneniyor (CPU modu)...", "warn")
            cmake_cmd_cpu = f'cmake -B build {generator} -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_SERVER=ON'
            r = run(cmake_cmd_cpu, cwd=str(BUILD_DIR), check=False, timeout=120)
            if r.returncode != 0:
                log("CMake tamamen başarısız!", "error")
                return False
            gpu_info.backend = "cpu"
            gpu_info.cmake_flags = []
    
    # Build
    cores = os.cpu_count() or 4
    
    if is_windows():
        build_cmd = f'cmake --build build --config Release -j {cores}'
    else:
        build_cmd = f'cmake --build build --config Release -j{cores}'
    
    log(f"Derleniyor ({cores} çekirdek)... Bu birkaç dakika sürebilir.", "step")
    r = run(build_cmd, cwd=str(BUILD_DIR), check=False, timeout=900)
    
    if r.returncode != 0:
        log("Derleme başarısız!", "error")
        return False
    
    # Binary'nin varlığını doğrula
    server_path = _find_server_binary()
    if server_path:
        log(f"Derleme başarılı: {server_path}", "ok")
        return True
    else:
        log("Derleme tamamlandı ama binary bulunamadı!", "error")
        return False


def _find_server_binary() -> Optional[str]:
    """Derlenen llama-server binary'sini bul."""
    candidates = [
        BUILD_DIR / "build" / "bin" / "llama-server",
        BUILD_DIR / "build" / "bin" / "llama-server.exe",
        BUILD_DIR / "build" / "bin" / "Release" / "llama-server.exe",
        BUILD_DIR / "build" / "bin" / "Debug" / "llama-server.exe",
        BUILD_DIR / "build" / "llama-server",
        BUILD_DIR / "build" / "llama-server.exe",
        BUILD_DIR / "build" / "Release" / "llama-server.exe",
    ]
    
    for path in candidates:
        if path.exists():
            return str(path)
    
    # Derin arama
    for path in BUILD_DIR.rglob("llama-server*"):
        if path.is_file() and not path.suffix == ".pdb":
            return str(path)
    
    return None


# ─────────────────────────────────────────────
# ADIM 4: Model bul veya indir
# ─────────────────────────────────────────────

def find_or_download_model(gpu_vram_mb: int = 0) -> Optional[str]:
    """
    Model bul veya indir — interaktif seçim menüsü.
    
    Öncelik:
      1. models/ klasöründeki GGUF'ları kontrol et
      2. Kullanıcıya VRAM'e uygun modelleri göster, seçtir veya indir
    """
    
    # Proje models/ dizinindeki modeller
    models_dir = PROJECT_DIR / MODELS_DIR_NAME
    local_models = []
    if models_dir.exists():
        for f in models_dir.glob("*.gguf"):
            size_gb = f.stat().st_size / (1024**3)
            if size_gb > 0.01:
                local_models.append((f.stem, str(f), size_gb))
    
    # ── Mevcut modeller varsa göster ──
    if local_models:
        local_models.sort(key=lambda x: x[2])
        log(f"{len(local_models)} GGUF model bulundu:", "ok")
        print()
        for i, (name, path, size_gb) in enumerate(local_models[:8]):
            print(f"    [{i+1}] {name} ({size_gb:.1f} GB)")
        
        print(f"\n    [0] Yeni model indir (katalogdan seç)")
        print()
        
        choice = _ask_choice(
            "Model seç (numara gir, Enter = 1): ",
            max_val=len(local_models[:8]),
            default=1,
            allow_zero=True
        )
        
        if choice > 0:
            chosen = local_models[choice - 1]
            log(f"Seçilen: {chosen[0]} ({chosen[2]:.1f} GB)", "ok")
            return chosen[1]
    
    # ── Katalogdan indir ──
    return _show_model_catalog_and_download(gpu_vram_mb, models_dir)


def _show_model_catalog_and_download(gpu_vram_mb: int, models_dir: Path) -> Optional[str]:
    """GPU VRAM'e göre uygun modelleri göster, kullanıcıya seçtir, indir."""
    vram_gb = gpu_vram_mb / 1024 if gpu_vram_mb > 0 else 99
    
    print()
    log("Model kataloğu (GPU'na uygun olanlar ★ ile işaretli):", "info")
    print()
    print(f"    {'#':>3}  {'★':>1}  {'Model':<40} {'Boyut':>6}  {'Açıklama'}")
    print(f"    {'─'*3}  {'─':>1}  {'─'*40} {'─'*6}  {'─'*30}")
    
    recommended_idx = None
    
    for i, m in enumerate(MODEL_CATALOG):
        fits = m["min_vram_gb"] <= vram_gb
        star = "★" if fits else " "
        
        # En büyük sığan modeli öner
        if fits:
            recommended_idx = i
        
        print(f"    [{i+1:>1}]  {star}  {m['name']:<40} {m['size_gb']:>5.1f}G  {m['desc']}")
    
    print()
    
    if recommended_idx is not None:
        rec = MODEL_CATALOG[recommended_idx]
        log(f"Önerilen: [{recommended_idx + 1}] {rec['name']} ({rec['size_gb']:.1f} GB)", "ok")
    
    default = (recommended_idx + 1) if recommended_idx is not None else 2
    
    choice = _ask_choice(
        f"Model seç (numara gir, Enter = {default}): ",
        max_val=len(MODEL_CATALOG),
        default=default,
        allow_zero=False
    )
    
    selected = MODEL_CATALOG[choice - 1]
    
    # VRAM uyarısı
    if selected["min_vram_gb"] > vram_gb and gpu_vram_mb > 0:
        log(f"⚠️  Bu model en az {selected['min_vram_gb']}GB VRAM istiyor, sende {vram_gb:.0f}GB var.", "warn")
        log("Yine de denenebilir (bazı katmanlar CPU'da çalışır).", "info")
    
    # İndir
    log(f"İndiriliyor: {selected['name']} ({selected['size_gb']:.1f} GB)", "download")
    log("Bu birkaç dakika sürebilir...", "info")
    print()
    
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / selected["file"]
    
    # Zaten indirilmiş mi?
    if model_path.exists():
        existing_size = model_path.stat().st_size / (1024**3)
        if existing_size > 0.1:
            log(f"Model zaten mevcut: {model_path.name} ({existing_size:.1f} GB)", "ok")
            return str(model_path)
    
    try:
        _download_with_progress(selected["url"], str(model_path))
        log(f"Model indirildi: {model_path.name}", "ok")
        return str(model_path)
    except Exception as e:
        log(f"İndirme başarısız: {e}", "error")
        log(f"Manuel indir: {selected['url']}", "info")
        log(f"Dosyayı şuraya koy: {models_dir}", "info")
        return None


def _ask_choice(prompt: str, max_val: int, default: int = 1, allow_zero: bool = False) -> int:
    """Kullanıcıdan numara seçimi al."""
    min_val = 0 if allow_zero else 1
    while True:
        try:
            raw = input(f"  ❯ {prompt}").strip()
            if raw == "":
                return default
            val = int(raw)
            if min_val <= val <= max_val:
                return val
            print(f"    {min_val}-{max_val} arası bir numara gir.")
        except ValueError:
            print(f"    Geçersiz giriş. Numara gir veya Enter'a bas.")
        except (EOFError, KeyboardInterrupt):
            return default


def _download_with_progress(url: str, dest: str):
    """İlerleme çubuğu ile dosya indir."""
    req = urllib.request.Request(url, headers={"User-Agent": "TurboQuant-Installer/0.1"})
    
    with urllib.request.urlopen(req) as response:
        total = int(response.headers.get('Content-Length', 0))
        downloaded = 0
        chunk_size = 1024 * 1024  # 1MB
        
        with open(dest, 'wb') as f:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                
                if total > 0:
                    pct = downloaded * 100 / total
                    bar_len = 30
                    filled = int(bar_len * downloaded / total)
                    bar = "█" * filled + "░" * (bar_len - filled)
                    mb_done = downloaded / (1024**2)
                    mb_total = total / (1024**2)
                    sys.stdout.write(f"\r  📥  [{bar}] {pct:.0f}% ({mb_done:.0f}/{mb_total:.0f} MB)")
                    sys.stdout.flush()
        
        if total > 0:
            print()  # Yeni satır


# ─────────────────────────────────────────────
# ADIM 5: Yapılandırma kaydet
# ─────────────────────────────────────────────

def save_config(gpu_info, model_path: str, server_path: str, is_tq_fork: bool = True):
    """Yapılandırmayı JSON'a kaydet."""
    # Context size tahmini
    from turboquant.gpu_detect import estimate_max_context
    max_ctx = estimate_max_context(gpu_info)
    
    # Güvenli context (max'ın %80'i)
    safe_ctx = min(max_ctx, 65536)
    if gpu_info.vram_mb < 12000:
        safe_ctx = min(safe_ctx, 32768)
    if gpu_info.vram_mb < 8000:
        safe_ctx = min(safe_ctx, 16384)
    
    # Cache tipi: TQ fork varsa turbo3/4, yoksa q8_0/q4_0
    if is_tq_fork:
        cache_k = "turbo3"
        cache_v = "turbo4"
    else:
        cache_k = "q8_0"
        cache_v = "q4_0"
    
    config = {
        "version": "0.1.0",
        "server_path": server_path,
        "model_path": model_path,
        "is_turboquant_fork": is_tq_fork,
        "gpu": {
            "name": gpu_info.name,
            "vendor": gpu_info.vendor,
            "backend": gpu_info.backend,
            "vram_mb": gpu_info.vram_mb,
        },
        "server_args": {
            "cache_type_k": cache_k,
            "cache_type_v": cache_v,
            "ctx_size": safe_ctx,
            "n_gpu_layers": gpu_info.recommended_layers,
            "host": "0.0.0.0",
            "port": 8080,
            "threads": os.cpu_count() or 4,
        },
        "installed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    try:
        safe_write_file(str(CONFIG_FILE), json.dumps(config, indent=2, ensure_ascii=False),
                        "config.json")
    except TQError:
        # Fallback: normal write
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    
    log(f"Yapılandırma kaydedildi: {CONFIG_FILE}", "ok")
    return config


# ─────────────────────────────────────────────
# ANA KURULUM
# ─────────────────────────────────────────────

def install():
    """Tam otomatik kurulum — hata yönetimli."""
    print()
    print("╔═══════════════════════════════════════════════════╗")
    print("║   TurboQuant Otomatik Kurulum                    ║")
    print("║   KV Cache Sıkıştırma ile Lokal LLM Inference    ║")
    print("╚═══════════════════════════════════════════════════╝")
    print()
    
    # ── ADIM 0: Proje dizini izin kontrolü ──
    print("━" * 50)
    print("  ADIM 0/5: Dizin izinleri kontrol ediliyor")
    print("━" * 50)
    
    try:
        issues = validate_project_directory(str(PROJECT_DIR))
        if issues:
            for code, msg in issues:
                log(f"[{code}] {msg}", "warn")
            log("İzin sorunları var ama devam ediliyor...", "warn")
        else:
            log("Dizin izinleri sorunsuz", "ok")
        
        # Kritik dizinleri oluştur
        ensure_directory(str(PROJECT_DIR / "models"), "models dizini")
    except TQError as e:
        print(str(e))
        print("\n  💡 Farklı bir dizine taşıyıp tekrar dene.")
        print(f"     Diagnostik: python diagnose.py")
        sys.exit(1)
    
    # ── ADIM 1: Gereksinimler ──
    print("━" * 50)
    print("  ADIM 1/5: Gereksinimler kontrol ediliyor")
    print("━" * 50)
    
    all_ok, missing = check_prerequisites()
    
    if all_ok:
        log("Tüm gereksinimler mevcut", "ok")
    else:
        log(f"Eksik araçlar: {', '.join(missing)}", "warn")
        log("Otomatik kurulum deneniyor...", "step")
        
        install_prerequisites(missing)
        
        # Kurulum sonrası TEKRAR kontrol et (PATH güncellendi olabilir)
        still_missing = []
        if "git" in missing and not cmd_exists("git"):
            still_missing.append("git")
        if "cmake" in missing and not cmd_exists("cmake"):
            still_missing.append("cmake")
        if "compiler" in missing:
            if is_windows():
                if not (cmd_exists("cl") or _check_msvc_installed() or cmd_exists("gcc")):
                    still_missing.append("compiler")
            elif not (cmd_exists("g++") or cmd_exists("clang++") or cmd_exists("clang")):
                still_missing.append("compiler")
        # NOT: cuda-toolkit hiçbir zaman kritik değil (Vulkan/CPU fallback var)
        
        if still_missing:
            log(f"Hâlâ eksik: {', '.join(still_missing)}", "error")
            log("Terminal'i kapatıp yeniden aç, sonra tekrar dene.", "info")
            log("Veya yukarıdaki indirme linklerini kullan.", "info")
            sys.exit(1)
        else:
            log("Tüm gereksinimler hazır (PATH güncellendi)", "ok")
    
    # ── ADIM 2: GPU Algılama ──
    print()
    print("━" * 50)
    print("  ADIM 2/5: GPU algılanıyor")
    print("━" * 50)
    
    from turboquant.gpu_detect import detect_gpu
    gpu_info = detect_gpu()
    
    # ── ADIM 3: Fork Klonla ──
    print()
    print("━" * 50)
    print("  ADIM 3/5: llama.cpp TurboQuant fork indiriliyor")
    print("━" * 50)
    
    clone_ok, is_tq_fork = clone_or_update_fork()
    if not clone_ok:
        sys.exit(1)
    
    # ── ADIM 4: Derle ──
    print()
    print("━" * 50)
    print("  ADIM 4/5: Derleniyor")
    print("━" * 50)
    
    if not compile_llama(gpu_info):
        sys.exit(1)
    
    server_path = _find_server_binary()
    
    # ── ADIM 5: Model Bul ──
    print()
    print("━" * 50)
    print("  ADIM 5/5: Model aranıyor")
    print("━" * 50)
    
    model_path = find_or_download_model(gpu_vram_mb=gpu_info.vram_mb)
    if not model_path:
        log("Model bulunamadı. 'run.py' çalıştırırken --model ile belirt.", "warn")
        model_path = ""
    
    # ── Yapılandırma Kaydet ──
    print()
    print("━" * 50)
    print("  Yapılandırma kaydediliyor")
    print("━" * 50)
    
    config = save_config(gpu_info, model_path, server_path, is_tq_fork)
    
    # ── Özet ──
    ctx_k = config["server_args"]["ctx_size"] // 1024
    cache_info = f'{config["server_args"]["cache_type_k"]}/{config["server_args"]["cache_type_v"]}'
    tq_status = "TurboQuant" if is_tq_fork else "Standart (q8_0/q4_0)"
    
    print()
    print("╔═══════════════════════════════════════════════════╗")
    print("║   ✅ KURULUM TAMAMLANDI!                         ║")
    print("╠═══════════════════════════════════════════════════╣")
    print(f"║  GPU:     {gpu_info.name[:40]:<41}║")
    print(f"║  Backend: {gpu_info.backend.upper():<41}║")
    print(f"║  KV Cache:{tq_status:<41}║")
    print(f"║  Context: {ctx_k}K token ({cache_info}){'':20}║")
    print(f"║  Model:   {Path(model_path).name[:40] if model_path else 'Belirtilmedi':<41}║")
    print("╠═══════════════════════════════════════════════════╣")
    print("║                                                   ║")
    print("║  Sunucu başlat:   python run.py                   ║")
    print("║  Sohbet aç:       python chat.py                  ║")
    print("║  Test et:         python test.py                  ║")
    print("║                                                   ║")
    print("╚═══════════════════════════════════════════════════╝")
    print()


if __name__ == "__main__":
    try:
        install()
    except TQError as e:
        print(str(e))
        print(f"\n  📋 Diagnostik rapor: python diagnose.py")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n  ⏹  Kullanıcı tarafından durduruldu.")
        sys.exit(130)
    except Exception as e:
        handle_error(e, "install.py ana işlem")
        print(f"\n  📋 Diagnostik rapor: python diagnose.py")
        sys.exit(1)
