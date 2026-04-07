"""
TurboQuant Hata Yönetim Sistemi
==================================
Her hata için benzersiz kod, açıklama ve çözüm önerisi.

Hata Kodu Şeması:
  TQ-1xx  Kurulum / Gereksinimler
  TQ-2xx  GPU Algılama
  TQ-3xx  Derleme (Build)
  TQ-4xx  Model Dosyası
  TQ-5xx  Sunucu / Çalıştırma
  TQ-6xx  Dosya Sistemi / İzinler
  TQ-7xx  Algoritmik / Kuantizasyon
  TQ-8xx  Ağ / İndirme
"""

import os
import sys
import stat
import platform
import traceback
import tempfile
import time
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass
from enum import Enum


# ═══════════════════════════════════════════════════
# Hata Kodları
# ═══════════════════════════════════════════════════

class ErrorCode(Enum):
    # ── Kurulum (1xx) ──
    PYTHON_VERSION      = "TQ-101"
    GIT_NOT_FOUND       = "TQ-102"
    CMAKE_NOT_FOUND     = "TQ-103"
    COMPILER_NOT_FOUND  = "TQ-104"
    CUDA_NOT_FOUND      = "TQ-105"
    PIP_INSTALL_FAIL    = "TQ-106"
    WINGET_FAIL         = "TQ-107"
    BREW_FAIL           = "TQ-108"
    APT_FAIL            = "TQ-109"

    # ── GPU (2xx) ──
    GPU_DETECT_FAIL     = "TQ-201"
    NVIDIA_SMI_FAIL     = "TQ-202"
    CUDA_MISMATCH       = "TQ-203"
    VRAM_TOO_LOW        = "TQ-204"
    VULKAN_NOT_FOUND    = "TQ-205"

    # ── Derleme (3xx) ──
    CLONE_FAIL          = "TQ-301"
    CMAKE_CONFIG_FAIL   = "TQ-302"
    BUILD_FAIL          = "TQ-303"
    BINARY_NOT_FOUND    = "TQ-304"
    BUILD_TIMEOUT       = "TQ-305"
    CLONE_TIMEOUT       = "TQ-306"

    # ── Model (4xx) ──
    MODEL_NOT_FOUND     = "TQ-401"
    MODEL_CORRUPT       = "TQ-402"
    MODEL_TOO_LARGE     = "TQ-403"
    MODEL_DOWNLOAD_FAIL = "TQ-404"
    GGUF_INVALID        = "TQ-405"

    # ── Sunucu (5xx) ──
    SERVER_START_FAIL   = "TQ-501"
    SERVER_CRASH        = "TQ-502"
    PORT_IN_USE         = "TQ-503"
    SERVER_TIMEOUT      = "TQ-504"
    CONFIG_MISSING      = "TQ-505"
    CONFIG_CORRUPT      = "TQ-506"
    API_ERROR           = "TQ-507"
    SERVER_NOT_RUNNING  = "TQ-508"

    # ── Dosya Sistemi / İzinler (6xx) ──
    DIR_CREATE_FAIL     = "TQ-601"
    DIR_NOT_WRITABLE    = "TQ-602"
    FILE_WRITE_FAIL     = "TQ-603"
    FILE_READ_FAIL      = "TQ-604"
    FILE_DELETE_FAIL    = "TQ-605"
    DISK_FULL           = "TQ-606"
    PATH_TOO_LONG       = "TQ-607"
    PERMISSION_DENIED   = "TQ-608"
    DIR_NOT_FOUND       = "TQ-609"
    FILE_LOCKED         = "TQ-610"

    # ── Algoritmik (7xx) ──
    QUANT_DIM_MISMATCH  = "TQ-701"
    QUANT_BITS_INVALID  = "TQ-702"
    CODEBOOK_FAIL       = "TQ-703"
    ROTATION_FAIL       = "TQ-704"
    QJL_NORM_ZERO       = "TQ-705"
    CACHE_OVERFLOW      = "TQ-706"

    # ── Ağ (8xx) ──
    NETWORK_UNREACHABLE = "TQ-801"
    DOWNLOAD_TIMEOUT    = "TQ-802"
    DOWNLOAD_CORRUPT    = "TQ-803"
    SSL_ERROR           = "TQ-804"
    DNS_FAIL            = "TQ-805"


# ═══════════════════════════════════════════════════
# Hata Açıklamaları ve Çözüm Önerileri
# ═══════════════════════════════════════════════════

ERROR_INFO = {
    # ── Kurulum (1xx) ──
    "TQ-101": {
        "title": "Python sürümü yetersiz",
        "fix": "Python 3.9+ kur: https://python.org/downloads"
    },
    "TQ-102": {
        "title": "Git bulunamadı",
        "fix_win": "Çalıştır: winget install Git.Git\nVeya: https://git-scm.com/download/win",
        "fix_mac": "Çalıştır: xcode-select --install",
        "fix_linux": "Çalıştır: sudo apt install git"
    },
    "TQ-103": {
        "title": "CMake bulunamadı",
        "fix_win": "Çalıştır: winget install Kitware.CMake\nVeya: https://cmake.org/download",
        "fix_mac": "Çalıştır: brew install cmake",
        "fix_linux": "Çalıştır: sudo apt install cmake"
    },
    "TQ-104": {
        "title": "C++ derleyicisi bulunamadı",
        "fix_win": "Visual Studio Build Tools kur:\n  https://visualstudio.microsoft.com/visual-cpp-build-tools\n  'Desktop development with C++' seç\n  Kurulumdan sonra terminal'i yeniden aç",
        "fix_mac": "Çalıştır: xcode-select --install",
        "fix_linux": "Çalıştır: sudo apt install build-essential"
    },
    "TQ-105": {
        "title": "CUDA Toolkit bulunamadı (NVIDIA GPU algılandı)",
        "fix": "İndir: https://developer.nvidia.com/cuda-downloads\nNot: CUDA olmadan Vulkan backend kullanılabilir (daha yavaş)"
    },
    "TQ-106": {
        "title": "pip paket kurulumu başarısız",
        "fix": "Çalıştır: pip install numpy scipy --upgrade\nVeya: python -m pip install --user numpy scipy"
    },
    "TQ-107": {
        "title": "winget komutu başarısız",
        "fix": "Windows Store'dan 'App Installer' güncelle.\nVeya aracı manuel indir."
    },
    "TQ-108": {
        "title": "Homebrew komutu başarısız",
        "fix": "brew güncelle: brew update && brew upgrade\nVeya: https://brew.sh ile yeniden kur"
    },
    "TQ-109": {
        "title": "apt paket yöneticisi hatası",
        "fix": "Çalıştır: sudo apt update && sudo apt upgrade\nYetki sorunu: sudo ile çalıştır"
    },

    # ── GPU (2xx) ──
    "TQ-201": {
        "title": "GPU algılanamadı",
        "fix": "GPU sürücülerinin güncel olduğundan emin ol.\nNVIDIA: https://www.nvidia.com/drivers\nCPU modu ile devam edebilirsin."
    },
    "TQ-202": {
        "title": "nvidia-smi çalıştırılamadı",
        "fix": "NVIDIA sürücüsü bozuk olabilir.\nSürücüyü yeniden kur: https://www.nvidia.com/drivers"
    },
    "TQ-203": {
        "title": "CUDA sürüm uyumsuzluğu",
        "fix": "CUDA Toolkit sürümü GPU sürücüsüyle uyumsuz.\nnvcc --version ve nvidia-smi çıktılarını karşılaştır.\nÖnerilen: CUDA 12.x"
    },
    "TQ-204": {
        "title": "GPU belleği çok düşük",
        "fix": "En az 4GB VRAM önerilir.\nDaha küçük model veya daha düşük context kullan."
    },
    "TQ-205": {
        "title": "Vulkan desteği bulunamadı",
        "fix_win": "GPU sürücüsünü güncelle (Vulkan desteği dahil gelir).",
        "fix_linux": "Çalıştır: sudo apt install vulkan-tools mesa-vulkan-drivers",
        "fix_mac": "macOS'ta Vulkan yerine Metal kullanılır."
    },

    # ── Derleme (3xx) ──
    "TQ-301": {
        "title": "Repo klonlanamadı",
        "fix": "İnternet bağlantısını kontrol et.\nFirewall veya proxy git'i engelliyor olabilir.\nManuel: git clone https://github.com/TheTom/llama-cpp-turboquant.git"
    },
    "TQ-302": {
        "title": "CMake yapılandırma hatası",
        "fix": "Derleyici veya CUDA Toolkit kurulumunu kontrol et.\nGPU desteği olmadan yeniden denenecek (CPU modu)."
    },
    "TQ-303": {
        "title": "Derleme başarısız",
        "fix": "Derleyici kurulumunu kontrol et.\nEkran kartı sürücülerini güncelle.\nDisk alanını kontrol et (en az 2GB boş alan gerekli)."
    },
    "TQ-304": {
        "title": "Derleme sonrası llama-server binary bulunamadı",
        "fix": "Derleme sessizce başarısız olmuş olabilir.\ninstall.py'yi tekrar çalıştır."
    },
    "TQ-305": {
        "title": "Derleme zaman aşımına uğradı",
        "fix": "Derleme 15 dakikadan uzun sürdü.\nDisk veya CPU performansını kontrol et.\nTekrar dene: python install.py"
    },
    "TQ-306": {
        "title": "Repo klonlama zaman aşımı",
        "fix": "İnternet bağlantısı yavaş veya kesilmiş.\nTekrar dene veya VPN'i kapat."
    },

    # ── Model (4xx) ──
    "TQ-401": {
        "title": "GGUF model dosyası bulunamadı",
        "fix": "Model dosyasını models/ klasörüne koy.\nVeya: python run.py --model /path/to/model.gguf\n"
    },
    "TQ-402": {
        "title": "Model dosyası bozuk",
        "fix": "Dosya boyutu sıfır veya çok küçük.\nModeli tekrar indir."
    },
    "TQ-403": {
        "title": "Model GPU belleğine sığmıyor",
        "fix": "Daha küçük bir model veya daha düşük kuantizasyon kullan (Q4_K_M önerilir).\nVeya --layers değerini düşür."
    },
    "TQ-404": {
        "title": "Model indirilemedi",
        "fix": "İnternet bağlantısını kontrol et.\nManuel indir: https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF"
    },
    "TQ-405": {
        "title": "Geçersiz GGUF dosyası",
        "fix": "Dosya GGUF formatında değil.\n.gguf uzantılı bir model kullan."
    },

    # ── Sunucu (5xx) ──
    "TQ-501": {
        "title": "Sunucu başlatılamadı",
        "fix": "Binary'nin çalışabilir olduğunu kontrol et.\nLog çıktısını incele.\nTekrar derle: python install.py"
    },
    "TQ-502": {
        "title": "Sunucu beklenmedik şekilde kapandı",
        "fix": "GPU belleği yetersiz olabilir.\nDaha küçük context dene: python run.py --ctx 8192\nVeya daha küçük model kullan."
    },
    "TQ-503": {
        "title": "Port zaten kullanılıyor",
        "fix_win": "Farklı port dene: python run.py --port 9090\nKullanan uygulamayı bul: netstat -ano | findstr :8080",
        "fix_mac": "Farklı port dene: python run.py --port 9090\nKullanan uygulamayı bul: lsof -i :8080",
        "fix_linux": "Farklı port dene: python run.py --port 9090\nKullanan uygulamayı bul: lsof -i :8080"
    },
    "TQ-504": {
        "title": "Sunucu yanıt vermiyor (timeout)",
        "fix": "Model yüklenmesi uzun sürebilir (büyük modeller için 1-2 dakika).\nBekle ve tekrar dene.\nVeya daha küçük model kullan."
    },
    "TQ-505": {
        "title": "config.json bulunamadı",
        "fix": "Önce kurulum yap: python install.py"
    },
    "TQ-506": {
        "title": "config.json bozuk veya okunamıyor",
        "fix": "config.json dosyasını sil ve tekrar kur: python install.py"
    },
    "TQ-507": {
        "title": "API isteği başarısız",
        "fix": "Sunucunun çalıştığından emin ol: python run.py --status\nLog çıktısını kontrol et."
    },
    "TQ-508": {
        "title": "Sunucu çalışmıyor",
        "fix": "Önce sunucuyu başlat: python run.py\nSonra sohbet aç: python chat.py"
    },

    # ── Dosya Sistemi / İzinler (6xx) ──
    "TQ-601": {
        "title": "Dizin oluşturulamadı",
        "fix_win": "Yönetici olarak çalıştır veya farklı dizinde dene.\nOneDrive/Dropbox senkronizasyonu engel olabilir.",
        "fix_mac": "İzinleri kontrol et: ls -la\nDüzelt: chmod 755 dizin_adı",
        "fix_linux": "İzinleri kontrol et: ls -la\nDüzelt: chmod 755 dizin_adı veya sudo"
    },
    "TQ-602": {
        "title": "Dizine yazma izni yok",
        "fix_win": "Sağ tık → Özellikler → Güvenlik → izinleri kontrol et.\nVeya farklı dizin kullan.",
        "fix_mac": "Çalıştır: chmod -R u+w dizin_adı",
        "fix_linux": "Çalıştır: chmod -R u+w dizin_adı"
    },
    "TQ-603": {
        "title": "Dosya yazılamadı",
        "fix_win": "Dosya başka bir uygulama tarafından açık olabilir.\nUygulamayı kapat ve tekrar dene.",
        "fix_mac": "İzinleri kontrol et: ls -la dosya_adı",
        "fix_linux": "İzinleri kontrol et: ls -la dosya_adı"
    },
    "TQ-604": {
        "title": "Dosya okunamadı",
        "fix": "Dosyanın var olduğundan ve okuma izni olduğundan emin ol."
    },
    "TQ-605": {
        "title": "Dosya/dizin silinemedi",
        "fix_win": "Dosya açık olabilir. Tüm uygulamaları kapat ve tekrar dene.",
        "fix_mac": "Çalıştır: chmod -R u+w dizin_adı && rm -rf dizin_adı",
        "fix_linux": "Çalıştır: chmod -R u+w dizin_adı && rm -rf dizin_adı"
    },
    "TQ-606": {
        "title": "Disk alanı yetersiz",
        "fix": "En az 5GB boş alan gerekli (derleme + model).\nGereksiz dosyaları temizle."
    },
    "TQ-607": {
        "title": "Dosya yolu çok uzun",
        "fix_win": "Windows 260 karakter yol sınırı var.\nProjeyi daha kısa bir yola taşı, örn: C:\\tq\\",
        "fix_mac": "Daha kısa bir dizine taşı.",
        "fix_linux": "Daha kısa bir dizine taşı."
    },
    "TQ-608": {
        "title": "Erişim izni reddedildi",
        "fix_win": "Yönetici olarak çalıştır.\nVeya proje dizininin izinlerini kontrol et.",
        "fix_mac": "Çalıştır: sudo python install.py\nVeya: chmod -R u+rwx proje_dizini",
        "fix_linux": "Çalıştır: sudo python install.py\nVeya: chmod -R u+rwx proje_dizini"
    },
    "TQ-609": {
        "title": "Dizin bulunamadı",
        "fix": "Belirtilen yolun doğru olduğundan emin ol.\nÜst dizinin var olduğunu kontrol et."
    },
    "TQ-610": {
        "title": "Dosya kilitli (başka uygulama kullanıyor)",
        "fix_win": "Dosyayı kullanan uygulamayı kapat.\nGörev Yöneticisi'nden kontrol et.",
        "fix_mac": "Çalıştır: lsof dosya_adı ile hangi uygulama kullandığını bul.",
        "fix_linux": "Çalıştır: fuser dosya_adı ile hangi process kullandığını bul."
    },

    # ── Algoritmik (7xx) ──
    "TQ-701": {
        "title": "Vektör boyutu uyumsuz",
        "fix": "head_dim değerini model yapılandırmasıyla eşleştir (genellikle 128 veya 256)."
    },
    "TQ-702": {
        "title": "Geçersiz bit genişliği",
        "fix": "Desteklenen bit genişlikleri: 2, 3, 4.\nMixed modlar: 2.5-bit, 3.5-bit."
    },
    "TQ-703": {
        "title": "Codebook hesaplama hatası",
        "fix": "Lloyd-Max yakınsamadı. Farklı d veya bits değeri dene."
    },
    "TQ-704": {
        "title": "Rotasyon matrisi hatası",
        "fix": "Bellekte sorun olabilir. Programı yeniden başlat."
    },
    "TQ-705": {
        "title": "QJL encode: sıfır normluk vektör",
        "fix": "Sıfır vektör kuantize edilemez. Model çıktısını kontrol et."
    },
    "TQ-706": {
        "title": "KV cache taşması",
        "fix": "Context çok uzun. --ctx değerini düşür."
    },

    # ── Ağ (8xx) ──
    "TQ-801": {
        "title": "İnternet bağlantısı yok",
        "fix": "İnternet bağlantısını kontrol et.\nProxy/VPN kullanıyorsan devre dışı bırakmayı dene."
    },
    "TQ-802": {
        "title": "İndirme zaman aşımı",
        "fix": "İnternet yavaş. Tekrar dene veya dosyayı manuel indir."
    },
    "TQ-803": {
        "title": "İndirilen dosya bozuk",
        "fix": "Dosya tam inmemiş olabilir. Sil ve tekrar indir."
    },
    "TQ-804": {
        "title": "SSL/TLS sertifika hatası",
        "fix_win": "Sistem saatinin doğru olduğundan emin ol.\nKurumsal ağdaysan IT departmanına danış.",
        "fix_mac": "Çalıştır: /Applications/Python\\ 3.x/Install\\ Certificates.command",
        "fix_linux": "Çalıştır: sudo apt install ca-certificates"
    },
    "TQ-805": {
        "title": "DNS çözümleme hatası",
        "fix": "DNS sunucusunu kontrol et.\nAlternatif: 8.8.8.8 veya 1.1.1.1 kullan."
    },
}


# ═══════════════════════════════════════════════════
# TurboQuant Hata Sınıfı
# ═══════════════════════════════════════════════════

class TQError(Exception):
    """TurboQuant hatası — kod, mesaj ve çözüm önerisi içerir."""

    def __init__(self, code: ErrorCode, detail: str = "", cause: Optional[Exception] = None):
        self.code = code
        self.code_str = code.value
        self.detail = detail
        self.cause = cause

        info = ERROR_INFO.get(code.value, {})
        self.title = info.get("title", "Bilinmeyen hata")

        # OS'e göre çözüm seç
        os_key = f"fix_{_os_tag()}"
        self.fix = info.get(os_key, info.get("fix", ""))

        super().__init__(self._format())

    def _format(self) -> str:
        lines = [
            "",
            f"  ╔══ HATA {self.code_str} ═══════════════════════════════════",
            f"  ║ {self.title}",
        ]
        if self.detail:
            for line in self.detail.split("\n"):
                lines.append(f"  ║ {line}")
        if self.cause:
            lines.append(f"  ║ Sebep: {type(self.cause).__name__}: {self.cause}")
        if self.fix:
            lines.append(f"  ╠══ ÇÖZÜM ═══════════════════════════════════════")
            for line in self.fix.split("\n"):
                lines.append(f"  ║ {line}")
        lines.append(f"  ╚══════════════════════════════════════════════════")
        return "\n".join(lines)


def _os_tag() -> str:
    s = platform.system()
    if s == "Windows": return "win"
    if s == "Darwin": return "mac"
    return "linux"


# ═══════════════════════════════════════════════════
# Dosya Sistemi İzin Kontrolleri
# ═══════════════════════════════════════════════════

@dataclass
class PermissionReport:
    """Dizin/dosya izin raporu."""
    path: str
    exists: bool
    readable: bool
    writable: bool
    deletable: bool
    is_dir: bool
    free_space_mb: int
    path_length: int
    errors: list


def check_permissions(path: str, need_write: bool = True, need_create: bool = True) -> PermissionReport:
    """
    Verilen yol için tüm izinleri kontrol eder.
    
    Kontrol edilen:
      - Dizin var mı / oluşturulabilir mi
      - Okuma izni
      - Yazma izni  
      - Silme izni
      - Disk alanı
      - Yol uzunluğu (Windows 260 char sınırı)
    """
    p = Path(path).resolve()
    errors = []

    report = PermissionReport(
        path=str(p),
        exists=p.exists(),
        readable=False,
        writable=False,
        deletable=False,
        is_dir=p.is_dir() if p.exists() else False,
        free_space_mb=0,
        path_length=len(str(p)),
        errors=errors
    )

    # 1. Yol uzunluğu (Windows sınırı)
    if platform.system() == "Windows" and len(str(p)) > 250:
        errors.append(("TQ-607", f"Yol uzunluğu {len(str(p))} karakter (max 260)"))

    # 2. Disk alanı (cross-platform: shutil.disk_usage)
    try:
        check_dir = p if p.is_dir() else p.parent
        if check_dir.exists():
            import shutil as _shutil
            disk = _shutil.disk_usage(str(check_dir))
            report.free_space_mb = disk.free // (1024 * 1024)
            if report.free_space_mb < 500:
                errors.append(("TQ-606", f"Boş alan: {report.free_space_mb} MB (min 500 MB)"))
    except Exception:
        pass

    # 3. Okuma izni
    if p.exists():
        report.readable = os.access(str(p), os.R_OK)
        if not report.readable:
            errors.append(("TQ-604", f"Okuma izni yok: {p}"))
    
    # 4. Yazma izni
    if need_write:
        if p.exists():
            report.writable = os.access(str(p), os.W_OK)
            if not report.writable:
                errors.append(("TQ-602", f"Yazma izni yok: {p}"))
        elif need_create:
            # Üst dizinde oluşturma testi
            parent = p.parent
            if parent.exists():
                report.writable = os.access(str(parent), os.W_OK)
                if not report.writable:
                    errors.append(("TQ-601", f"Üst dizine yazma izni yok: {parent}"))
            else:
                errors.append(("TQ-609", f"Üst dizin mevcut değil: {parent}"))

    # 5. Silme izni (varolan dosya/dizin için)
    if p.exists():
        try:
            # Geçici dosya ile silme testi
            if p.is_dir():
                test_file = p / f".tq_permission_test_{os.getpid()}"
                test_file.touch()
                test_file.unlink()
                report.deletable = True
            else:
                report.deletable = os.access(str(p.parent), os.W_OK)
        except PermissionError:
            report.deletable = False
            errors.append(("TQ-605", f"Silme izni yok: {p}"))
        except Exception:
            report.deletable = True  # Test başarısız ama ciddi değil

    report.errors = errors
    return report


def ensure_directory(path: str, label: str = "dizin") -> Path:
    """
    Dizini oluştur, izinleri kontrol et. Hata varsa TQError fırlat.
    """
    p = Path(path).resolve()
    
    # Yol uzunluğu kontrolü
    if platform.system() == "Windows" and len(str(p)) > 250:
        raise TQError(ErrorCode.PATH_TOO_LONG,
                       f"{label}: {p}\nUzunluk: {len(str(p))} karakter")
    
    if p.exists():
        if not p.is_dir():
            raise TQError(ErrorCode.DIR_CREATE_FAIL,
                           f"{label} bir dosya olarak mevcut (dizin değil): {p}")
        if not os.access(str(p), os.W_OK):
            raise TQError(ErrorCode.DIR_NOT_WRITABLE,
                           f"{label}: {p}")
        return p
    
    # Oluşturmayı dene
    try:
        p.mkdir(parents=True, exist_ok=True)
    except PermissionError as e:
        raise TQError(ErrorCode.PERMISSION_DENIED,
                       f"{label} oluşturulamadı: {p}", cause=e)
    except OSError as e:
        if "No space" in str(e) or "disk" in str(e).lower():
            raise TQError(ErrorCode.DISK_FULL, f"{label}: {p}", cause=e)
        raise TQError(ErrorCode.DIR_CREATE_FAIL, f"{label}: {p}", cause=e)
    
    return p


def ensure_writable_file(path: str, label: str = "dosya") -> Path:
    """Dosya yazılabilir mi kontrol et. Yoksa üst dizini kontrol et."""
    p = Path(path).resolve()
    
    if p.exists():
        if not os.access(str(p), os.W_OK):
            raise TQError(ErrorCode.FILE_WRITE_FAIL, f"{label}: {p}")
        # Windows: dosya kilitli mi?
        if platform.system() == "Windows":
            try:
                with open(p, 'a'):
                    pass
            except PermissionError as e:
                raise TQError(ErrorCode.FILE_LOCKED,
                               f"{label} başka bir uygulama tarafından kilitlenmiş: {p}", cause=e)
    else:
        parent = p.parent
        if not parent.exists():
            ensure_directory(str(parent), f"{label} üst dizini")
        if not os.access(str(parent), os.W_OK):
            raise TQError(ErrorCode.DIR_NOT_WRITABLE,
                           f"{label} yazılamaz (üst dizin izni yok): {parent}")
    
    return p


def safe_write_file(path: str, content: str, label: str = "dosya") -> Path:
    """Güvenli dosya yazma — atomik, izin kontrollü."""
    p = ensure_writable_file(path, label)
    
    # Atomik yazma: önce temp dosyaya yaz, sonra taşı
    tmp_path = str(p) + f".tmp.{os.getpid()}"
    try:
        with open(tmp_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # Hedef varsa sil
        if p.exists():
            try:
                p.unlink()
            except PermissionError as e:
                os.unlink(tmp_path)
                raise TQError(ErrorCode.FILE_DELETE_FAIL,
                               f"Eski {label} silinemiyor: {p}", cause=e)
        
        os.rename(tmp_path, str(p))
        return p
        
    except TQError:
        raise
    except PermissionError as e:
        _cleanup_tmp(tmp_path)
        raise TQError(ErrorCode.FILE_WRITE_FAIL, f"{label}: {p}", cause=e)
    except OSError as e:
        _cleanup_tmp(tmp_path)
        if "No space" in str(e) or "disk" in str(e).lower():
            raise TQError(ErrorCode.DISK_FULL, f"{label}: {p}", cause=e)
        raise TQError(ErrorCode.FILE_WRITE_FAIL, f"{label}: {p}", cause=e)


def safe_delete(path: str, label: str = "dosya"):
    """Güvenli silme."""
    p = Path(path)
    if not p.exists():
        return
    try:
        if p.is_dir():
            import shutil
            shutil.rmtree(str(p))
        else:
            p.unlink()
    except PermissionError as e:
        raise TQError(ErrorCode.FILE_DELETE_FAIL, f"{label}: {p}", cause=e)
    except OSError as e:
        raise TQError(ErrorCode.FILE_DELETE_FAIL, f"{label}: {p}", cause=e)


def _cleanup_tmp(path: str):
    try:
        os.unlink(path)
    except Exception:
        pass


# ═══════════════════════════════════════════════════
# Proje Dizini Doğrulama
# ═══════════════════════════════════════════════════

def validate_project_directory(project_dir: str) -> list:
    """
    Tüm kritik dizinleri/dosyaları kontrol et.
    Hataları liste olarak döndür.
    
    Returns:
        [(ErrorCode, mesaj), ...] — boş liste = sorun yok
    """
    issues = []
    p = Path(project_dir).resolve()
    
    # Ana dizin
    report = check_permissions(str(p), need_write=True)
    issues.extend(report.errors)
    
    # Alt dizinler
    for subdir in ["models", "llama-cpp-turboquant", "turboquant"]:
        sub_path = p / subdir
        if sub_path.exists():
            sub_report = check_permissions(str(sub_path), need_write=True)
            issues.extend(sub_report.errors)
    
    # config.json yazılabilirlik
    config_path = p / "config.json"
    if config_path.exists():
        if not os.access(str(config_path), os.W_OK):
            issues.append(("TQ-603", f"config.json yazılamaz: {config_path}"))
    
    return issues


# ═══════════════════════════════════════════════════
# Global Hata Yakalayıcı
# ═══════════════════════════════════════════════════

def handle_error(error: Exception, context: str = ""):
    """
    Herhangi bir hatayı yakala, güzel formatlı çıktı ver.
    TQError ise kod ve çözüm göster, değilse genel diagnostik yap.
    """
    if isinstance(error, TQError):
        print(str(error))
        return error.code_str
    
    if isinstance(error, KeyboardInterrupt):
        print("\n  ⏹  Kullanıcı tarafından durduruldu (Ctrl+C)")
        return "USER_CANCEL"
    
    if isinstance(error, PermissionError):
        tq = TQError(ErrorCode.PERMISSION_DENIED,
                      f"Bağlam: {context}\nDosya/dizin: {getattr(error, 'filename', '?')}",
                      cause=error)
        print(str(tq))
        return tq.code_str
    
    if isinstance(error, FileNotFoundError):
        tq = TQError(ErrorCode.DIR_NOT_FOUND,
                      f"Bağlam: {context}\nYol: {getattr(error, 'filename', '?')}",
                      cause=error)
        print(str(tq))
        return tq.code_str
    
    if isinstance(error, OSError):
        if "No space" in str(error) or "disk" in str(error).lower():
            tq = TQError(ErrorCode.DISK_FULL, f"Bağlam: {context}", cause=error)
        else:
            tq = TQError(ErrorCode.FILE_WRITE_FAIL, f"Bağlam: {context}", cause=error)
        print(str(tq))
        return tq.code_str
    
    # Genel hata
    print(f"""
  ╔══ BEKLENMEYEN HATA ═══════════════════════════════
  ║ Bağlam: {context}
  ║ Tür: {type(error).__name__}
  ║ Mesaj: {error}
  ╠══ DETAY ═══════════════════════════════════════════
  ║ {traceback.format_exc().replace(chr(10), chr(10) + '  ║ ')}
  ╠══ NE YAPMALI ═════════════════════════════════════
  ║ 1. Yukarıdaki hata mesajını kopyala
  ║ 2. GitHub issue aç veya geliştiriciyle paylaş
  ║ 3. python install.py ile yeniden dene
  ╚══════════════════════════════════════════════════""")
    return "UNKNOWN"


def run_with_error_handling(func, context: str = ""):
    """Fonksiyonu hata yakalama ile çalıştır."""
    try:
        return func()
    except TQError:
        raise  # TQError'u yukarı ilet
    except Exception as e:
        handle_error(e, context)
        raise TQError(ErrorCode.PERMISSION_DENIED, f"İşlem başarısız: {context}", cause=e)


# ═══════════════════════════════════════════════════
# Diagnostik Rapor
# ═══════════════════════════════════════════════════

def generate_diagnostic_report(project_dir: str) -> str:
    """Sorun giderme için tam sistem raporu oluştur."""
    lines = [
        "╔══ TurboQuant Diagnostik Rapor ════════════════════",
        f"║ Tarih: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"║ OS: {platform.system()} {platform.release()} ({platform.machine()})",
        f"║ Python: {sys.version.split()[0]}",
        f"║ Proje: {project_dir}",
        "╠══ Araçlar ════════════════════════════════════════",
    ]
    
    import shutil
    for tool in ["git", "cmake", "gcc", "g++", "cl", "nvcc", "nvidia-smi", "vulkaninfo"]:
        path = shutil.which(tool)
        status = f"✅ {path}" if path else "❌ bulunamadı"
        lines.append(f"║ {tool:15} {status}")
    
    lines.append("╠══ Dizin İzinleri ══════════════════════════════")
    
    p = Path(project_dir).resolve()
    for name in [".", "models", "llama-cpp-turboquant", "turboquant", "config.json"]:
        target = p / name if name != "." else p
        if target.exists():
            r = "R" if os.access(str(target), os.R_OK) else "-"
            w = "W" if os.access(str(target), os.W_OK) else "-"
            x = "X" if os.access(str(target), os.X_OK) else "-"
            lines.append(f"║ {name:30} [{r}{w}{x}] ✅")
        else:
            lines.append(f"║ {name:30} mevcut değil")
    
    # Disk alanı
    report = check_permissions(str(p))
    lines.append(f"╠══ Disk ════════════════════════════════════════")
    lines.append(f"║ Boş alan: {report.free_space_mb} MB")
    lines.append(f"║ Yol uzunluğu: {report.path_length} karakter")
    
    if report.errors:
        lines.append(f"╠══ Sorunlar ({len(report.errors)}) ══════════════════════")
        for code, msg in report.errors:
            lines.append(f"║ [{code}] {msg}")
    
    lines.append("╚══════════════════════════════════════════════════")
    return "\n".join(lines)
