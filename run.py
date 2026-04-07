"""
TurboQuant Başlatıcı
======================
Kurulumdan sonra tek komutla sunucuyu başlatır.

Kullanım:
    python run.py                     # Varsayılan ayarlarla başlat
    python run.py --model PATH.gguf   # Farklı model
    python run.py --ctx 131072        # 128K context
    python run.py --port 1234         # Farklı port
    python run.py --status            # Sunucu durumunu kontrol et

NOT: "python run.bat" veya "py run.bat" ÇALIŞMAZ.
     .bat dosyaları çift tıklanır veya sadece "run.bat" yazılır.
     Python dosyası: python run.py
"""

import os
import sys

# ── .bat dosyası Python'a verilmiş mi kontrol et ──
_this = os.path.basename(sys.argv[0]) if sys.argv else ""
if _this.endswith(".bat"):
    print()
    print("  ╔══ YANLIŞ KOMUT ══════════════════════════════════")
    print(f"  ║ '{_this}' bir Windows batch dosyası, Python dosyası değil.")
    print("  ║")
    print("  ║ Doğru kullanım (birini seç):")
    print("  ║")
    print("  ║   python run.py        ← terminal'de yaz")
    print(f"  ║   {_this}              ← çift tıkla veya sadece bunu yaz")
    print("  ║")
    print("  ║ İlk kurulum henüz yapılmadıysa:")
    print("  ║   python install.py")
    print("  ╚══════════════════════════════════════════════════")
    print()
    sys.exit(1)

import json
import signal
import argparse
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Optional

PROJECT_DIR = Path(__file__).parent.resolve()
CONFIG_FILE = PROJECT_DIR / "config.json"
sys.path.insert(0, str(PROJECT_DIR))

from turboquant.errors import TQError, ErrorCode, handle_error, check_permissions


def load_config() -> dict:
    """Yapılandırma dosyasını yükle — hata yönetimli."""
    if not CONFIG_FILE.exists():
        raise TQError(ErrorCode.CONFIG_MISSING,
                       f"Dosya: {CONFIG_FILE}")
    
    try:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        raise TQError(ErrorCode.CONFIG_CORRUPT,
                       f"config.json geçersiz JSON: {e}", cause=e)
    except PermissionError as e:
        raise TQError(ErrorCode.PERMISSION_DENIED,
                       f"config.json okunamıyor: {CONFIG_FILE}", cause=e)
    
    return config


def check_server_health(port: int) -> bool:
    """Sunucu çalışıyor mu kontrol et."""
    try:
        url = f"http://localhost:{port}/health"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


def start_server(config: dict, overrides: dict):
    """TurboQuant sunucusunu başlat — hata yönetimli."""
    server_path = config.get("server_path")
    if not server_path or not os.path.exists(server_path):
        raise TQError(ErrorCode.BINARY_NOT_FOUND,
                       f"Beklenen: {server_path}")
    
    # Binary çalıştırma izni kontrolü
    if not os.access(server_path, os.X_OK) and platform.system() != "Windows":
        try:
            os.chmod(server_path, 0o755)
        except PermissionError as e:
            raise TQError(ErrorCode.PERMISSION_DENIED,
                           f"Binary çalıştırma izni verilemedi: {server_path}", cause=e)
    
    args = config["server_args"].copy()
    model_path = overrides.get("model") or config.get("model_path", "")
    
    if not model_path or not os.path.exists(model_path):
        raise TQError(ErrorCode.MODEL_NOT_FOUND,
                       f"Beklenen: {model_path}\nÇözüm: python run.py --model /path/to/model.gguf")
    
    # Model dosyası okuma izni
    if not os.access(model_path, os.R_OK):
        raise TQError(ErrorCode.FILE_READ_FAIL,
                       f"Model dosyası okunamıyor: {model_path}")
    
    # Model boyut kontrolü (0 byte = bozuk)
    model_size = os.path.getsize(model_path)
    if model_size < 1024:
        raise TQError(ErrorCode.MODEL_CORRUPT,
                       f"Model dosyası çok küçük ({model_size} byte): {model_path}")
    
    # Port kontrolü
    port = overrides.get("port") or args.get("port", 8080)
    if _is_port_in_use(port):
        raise TQError(ErrorCode.PORT_IN_USE,
                       f"Port {port} zaten kullanılıyor")
    args["port"] = port
    
    # Override'ları uygula
    if overrides.get("ctx"):
        args["ctx_size"] = overrides["ctx"]
    if overrides.get("port"):
        args["port"] = overrides["port"]
    if overrides.get("layers") is not None:
        args["n_gpu_layers"] = overrides["layers"]
    
    # Komutu oluştur
    cmd_parts = [
        f'"{server_path}"',
        f'--model "{model_path}"',
        f'--cache-type-k {args["cache_type_k"]}',
        f'--cache-type-v {args["cache_type_v"]}',
        f'--ctx-size {args["ctx_size"]}',
        f'--n-gpu-layers {args["n_gpu_layers"]}',
        f'--host {args["host"]}',
        f'--port {args["port"]}',
        f'--threads {args.get("threads", os.cpu_count() or 4)}',
    ]
    cmd = " ".join(cmd_parts)
    
    gpu = config.get("gpu", {})
    ctx_k = args["ctx_size"] // 1024
    model_name = Path(model_path).stem
    
    print()
    print("╔═══════════════════════════════════════════════════╗")
    print("║   🚀 TurboQuant Server                           ║")
    print("╠═══════════════════════════════════════════════════╣")
    print(f"║  Model:    {model_name[:40]:<41}║")
    print(f"║  GPU:      {gpu.get('name', '?')[:40]:<41}║")
    print(f"║  Backend:  {gpu.get('backend', '?').upper():<41}║")
    print(f"║  Context:  {ctx_k}K token{'':40}║")
    print(f"║  KV Cache: key@{args['cache_type_k']} + val@{args['cache_type_v']}{'':24}║")
    print("╠═══════════════════════════════════════════════════╣")
    print(f"║  API:  http://localhost:{args['port']}/v1                  ║")
    print(f"║  Chat: http://localhost:{args['port']}/                    ║")
    print("╠═══════════════════════════════════════════════════╣")
    print("║  Ctrl+C ile durdur                                ║")
    print("╚═══════════════════════════════════════════════════╝")
    print()
    print(f"  Komut: {cmd[:80]}...")
    print()
    
    # Sunucuyu başlat
    try:
        process = subprocess.Popen(
            cmd, shell=True,
            stdout=sys.stdout,
            stderr=sys.stderr
        )
        
        # Sağlık kontrolü bekle
        print("  ⏳ Sunucu başlatılıyor...")
        for i in range(30):
            time.sleep(1)
            if check_server_health(args["port"]):
                print(f"\n  ✅ Sunucu hazır! → http://localhost:{args['port']}/v1")
                print()
                print("  ─── Kullanım Örnekleri ───")
                print()
                print("  curl ile:")
                print(f'    curl http://localhost:{args["port"]}/v1/chat/completions \\')
                print('      -H "Content-Type: application/json" \\')
                print("      -d '{\"model\":\"local\",\"messages\":[{\"role\":\"user\",\"content\":\"Merhaba\"}]}'")
                print()
                print("  Python ile:")
                print("    from openai import OpenAI")
                print(f'    client = OpenAI(base_url="http://localhost:{args["port"]}/v1", api_key="x")')
                print('    r = client.chat.completions.create(')
                print('        model="local",')
                print('        messages=[{"role": "user", "content": "Merhaba"}]')
                print("    )")
                print("    print(r.choices[0].message.content)")
                print()
                break
        else:
            print("  ⚠️  Sunucu henüz yanıt vermiyor, ama çalışıyor olabilir.")
            print(f"     Manuel kontrol: http://localhost:{args['port']}/health")
        
        # Ctrl+C bekle
        process.wait()
        
    except KeyboardInterrupt:
        print("\n  🛑 Sunucu durduruluyor...")
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        print("  ✅ Sunucu durduruldu.")
    except Exception as e:
        print(f"  ❌ Başlatma hatası: {e}")
        sys.exit(1)


def show_status(config: dict):
    """Kurulum ve sunucu durumunu göster."""
    gpu = config.get("gpu", {})
    args = config.get("server_args", {})
    
    print()
    print("📊 TurboQuant Durumu")
    print("━" * 40)
    print(f"  Server:   {config.get('server_path', '?')}")
    print(f"  Model:    {config.get('model_path', '?')}")
    print(f"  GPU:      {gpu.get('name', '?')} ({gpu.get('backend', '?').upper()})")
    print(f"  VRAM:     {gpu.get('vram_mb', 0)} MB")
    print(f"  Context:  {args.get('ctx_size', 0) // 1024}K")
    print(f"  KV Cache: key@{args.get('cache_type_k', '?')} + val@{args.get('cache_type_v', '?')}")
    print(f"  Kurulum:  {config.get('installed_at', '?')}")
    
    port = args.get("port", 8080)
    running = check_server_health(port)
    print(f"  Sunucu:   {'✅ Çalışıyor' if running else '⭕ Kapalı'} (port {port})")
    print()


import platform

def _is_port_in_use(port: int) -> bool:
    """Port başka bir uygulama tarafından kullanılıyor mu?"""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("0.0.0.0", port))
            return False
        except OSError:
            return True


def main():
    parser = argparse.ArgumentParser(
        description="TurboQuant Server Başlatıcı",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnekler:
  python run.py                          Varsayılan ayarlarla başlat
  python run.py --select                 Model seçim menüsü göster
  python run.py --model my_model.gguf    Belirtilen modeli kullan
  python run.py --ctx 131072             128K context
  python run.py --port 1234              Farklı port
  python run.py --status                 Durumu göster
        """
    )
    parser.add_argument("--model", help="GGUF model dosya yolu")
    parser.add_argument("--select", action="store_true", help="Model seçim menüsünü aç")
    parser.add_argument("--ctx", type=int, help="Context uzunluğu (token)")
    parser.add_argument("--port", type=int, help="API port numarası")
    parser.add_argument("--layers", type=int, help="GPU layer sayısı (0=CPU)")
    parser.add_argument("--status", action="store_true", help="Durumu göster")
    
    args = parser.parse_args()
    
    try:
        config = load_config()
    except TQError as e:
        print(str(e))
        sys.exit(1)
    
    if args.status:
        show_status(config)
        return
    
    # ── Model seçimi ──
    selected_model = args.model
    
    if args.select and not selected_model:
        selected_model = select_model_interactive(config)
        if not selected_model:
            print("  Model seçilmedi, çıkılıyor.")
            sys.exit(0)
    
    # Seçilen modeli config'e kaydet (bir sonraki run'da hatırlasın)
    if selected_model and selected_model != config.get("model_path"):
        config["model_path"] = selected_model
        try:
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            print(f"  📌 config.json güncellendi: {Path(selected_model).name}")
        except Exception:
            pass  # Yazamazsa da devam et, override zaten çalışır
    
    overrides = {
        "model": selected_model,
        "ctx": args.ctx,
        "port": args.port,
        "layers": args.layers,
    }
    
    try:
        start_server(config, overrides)
    except TQError as e:
        print(str(e))
        print(f"\n  📋 Diagnostik: python diagnose.py")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n  ⏹  Durduruldu.")
    except Exception as e:
        handle_error(e, "run.py sunucu başlatma")
        sys.exit(1)


def select_model_interactive(config: dict) -> str:
    """Model seçim menüsü — mevcut modeller + katalogdan indirme."""
    print()
    print("  🔍 Modeller taranıyor...")
    
    models = []
    
    # Sadece kendi models/ dizinimizi tara (uyumsuz formatları önlemek için 
    # sadece test edilmiş GGUF dosyaları)
    project_models = Path(__file__).parent / "models"
    if project_models.exists():
        for f in project_models.glob("*.gguf"):
            size_gb = f.stat().st_size / (1024**3)
            if size_gb > 0.01:
                # Uyumsuz formatları filtrele
                name_lower = f.stem.lower()
                if any(bad in name_lower for bad in ["mxfp", "mxint", "bnb-", "awq-"]):
                    continue
                models.append((f.stem, str(f), size_gb))
    
    # Mevcut config'teki model (models/ dışında olabilir)
    current = config.get("model_path", "")
    current_name = Path(current).stem if current else ""
    if current and os.path.exists(current):
        already = [m[1] for m in models]
        if current not in already:
            size_gb = os.path.getsize(current) / (1024**3)
            models.append((current_name, current, size_gb))
    
    # Boyuta göre sırala
    models.sort(key=lambda x: x[2])
    
    # ── Menü göster ──
    print()
    
    if models:
        print("  ─── Mevcut Modeller ───")
        print()
        for i, (name, path, size) in enumerate(models):
            marker = " ◄" if name == current_name else ""
            print(f"    [{i+1}]  {name[:50]:<50} {size:>5.1f}G{marker}")
        if current_name:
            print(f"\n    ◄ = şu an yüklü")
        print()
    
    # Katalog seçeneği
    next_idx = len(models) + 1
    print("  ─── Yeni Model İndir ───")
    print()
    
    catalog = _get_model_catalog()
    prev_category = ""
    for j, m in enumerate(catalog):
        idx = next_idx + j
        
        # Boyut kategorisi başlığı
        if m["size_gb"] < 3:
            category = "Küçük (1-3B)"
        elif m["size_gb"] < 7:
            category = "Orta (7-9B)"
        else:
            category = "Büyük (12-24B) ← TurboQuant ile 12GB VRAM'e sığar!"
        
        if category != prev_category:
            if prev_category:
                print()
            print(f"    ── {category} ──")
            prev_category = category
        
        already_have = any(m["file"] in existing[0] for existing in models)
        tag = " ✓var" if already_have else ""
        print(f"    [{idx:>2}]  {m['name']:<48} {m['size_gb']:>5.1f}G  {m['desc']}{tag}")
    
    total_options = len(models) + len(catalog)
    
    # HuggingFace arama seçeneği
    hf_idx = total_options + 1
    print()
    print("  ─── HuggingFace'ten Başka Model İndir ───")
    print()
    print(f"    [{hf_idx:>2}]  🔍 HuggingFace'te ara (link açılır, model adını yapıştır)")
    print()
    
    # Seçim
    while True:
        try:
            raw = input("  ❯ Numara seç (Enter = iptal): ").strip()
            if raw == "":
                return ""
            val = int(raw)
            
            if 1 <= val <= len(models):
                # Mevcut model
                chosen = models[val - 1]
                print(f"\n  ✅ {chosen[0]} ({chosen[2]:.1f} GB)")
                return chosen[1]
            
            elif len(models) < val <= total_options:
                # Katalogdan indir
                cat_idx = val - len(models) - 1
                selected = catalog[cat_idx]
                return _download_catalog_model(selected, project_models)
            
            elif val == hf_idx:
                # HuggingFace arama
                return _huggingface_manual_download(project_models)
            
            else:
                print(f"    1-{hf_idx} arası bir numara gir.")
        
        except ValueError:
            print("    Numara gir veya Enter'a bas.")
        except (EOFError, KeyboardInterrupt):
            return ""


def _huggingface_manual_download(models_dir: Path) -> str:
    """
    HuggingFace'te arama: kullanıcı model adını yapıştırır, biz en uygun GGUF'u bulup indiririz.
    
    Akış:
      1. Tarayıcıda arama sayfası açılır
      2. Kullanıcı model adını kopyalar (ör: bartowski/Qwen2.5-14B-Instruct-GGUF)
      3. Biz HF API ile repo'daki dosyaları tarar, en uygun GGUF'u seçeriz
    """
    print()
    print("  ┌──────────────────────────────────────────────────────────┐")
    print("  │  HuggingFace Model Arama                                │")
    print("  ├──────────────────────────────────────────────────────────┤")
    print("  │                                                          │")
    print("  │  1. Tarayıcıda açılan sayfada model ara                 │")
    print("  │  2. Beğendiğin modelin adını kopyala                    │")
    print("  │  3. Buraya yapıştır                                     │")
    print("  │                                                          │")
    print("  │  Örnek: bartowski/Qwen2.5-14B-Instruct-GGUF            │")
    print("  │  Örnek: mudler/Qwen3.5-35B-A3B-APEX-GGUF               │")
    print("  │                                                          │")
    print("  └──────────────────────────────────────────────────────────┘")
    print()
    
    # Tarayıcıda arama sayfası aç
    search_url = "https://huggingface.co/models?pipeline_tag=text-generation&num_parameters=min:1B,max:32B&apps=llama.cpp&sort=downloads"
    try:
        import webbrowser
        webbrowser.open(search_url)
        print("  ✅ Arama sayfası tarayıcıda açıldı!")
    except Exception:
        print(f"  📋 Bu linki tarayıcıda aç:")
        print(f"  {search_url}")
    
    print()
    
    while True:
        try:
            raw = input("  ❯ Model adını yapıştır (Enter = iptal): ").strip()
            if not raw:
                return ""
            
            # Temizle: URL veya repo adı olabilir
            repo_id = raw.strip().strip('"').strip("'")
            
            # URL'den repo adını çıkar
            if "huggingface.co/" in repo_id:
                parts = repo_id.split("huggingface.co/")[-1].split("/")
                if len(parts) >= 2:
                    repo_id = parts[0] + "/" + parts[1]
            
            # Doğrulama: owner/model formatı
            if "/" not in repo_id:
                print("    ⚠️  Format: sahip/model-adı (ör: bartowski/Qwen2.5-14B-Instruct-GGUF)")
                continue
            
            print(f"\n  🔍 {repo_id} taranıyor...")
            
            # HuggingFace API ile dosya listesini çek
            gguf_files = _fetch_hf_gguf_files(repo_id)
            
            if gguf_files is None:
                print(f"    ❌ Repo bulunamadı veya erişilemiyor: {repo_id}")
                print(f"    💡 Adı kontrol et, HuggingFace sayfasından kopyala")
                continue
            
            if not gguf_files:
                print(f"    ❌ Bu repo'da GGUF dosyası yok")
                continue
            
            # En uygun dosyayı seç veya kullanıcıya seçtir
            chosen_file = _pick_best_gguf(gguf_files, repo_id)
            if not chosen_file:
                continue
            
            filename = chosen_file["name"]
            api_size = chosen_file["size"]
            size_gb = api_size / (1024**3) if api_size > 0 else 0
            url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
            
            size_str = f"({size_gb:.1f} GB)" if size_gb > 0.1 else "(boyut bilinmiyor)"
            print(f"\n  📥 İndiriliyor: {filename} {size_str}")
            print()
            
            # İndir
            models_dir.mkdir(exist_ok=True)
            dest = models_dir / filename
            
            if dest.exists() and dest.stat().st_size > 1024:
                print(f"  ✅ Zaten mevcut: {filename}")
                return str(dest)
            
            try:
                import urllib.request as urlreq
                req = urlreq.Request(url, headers={"User-Agent": "TurboQuant/0.1"})
                with urlreq.urlopen(req) as resp:
                    total = int(resp.headers.get('Content-Length', 0))
                    downloaded = 0
                    
                    with open(str(dest), 'wb') as f:
                        while True:
                            chunk = resp.read(1024 * 1024)
                            if not chunk:
                                break
                            f.write(chunk)
                            downloaded += len(chunk)
                            
                            if total > 0:
                                pct = downloaded * 100 / total
                                filled = int(30 * downloaded / total)
                                bar = "█" * filled + "░" * (30 - filled)
                                mb = downloaded / (1024**2)
                                mb_total = total / (1024**2)
                                sys.stdout.write(f"\r  [{bar}] {pct:.0f}% ({mb:.0f}/{mb_total:.0f} MB)")
                                sys.stdout.flush()
                
                actual_gb = total / (1024**3) if total > 0 else dest.stat().st_size / (1024**3)
                print(f"\n\n  ✅ İndirildi: {filename} ({actual_gb:.1f} GB)")
                return str(dest)
            
            except Exception as e:
                print(f"\n  ❌ İndirme hatası: {e}")
                if dest.exists():
                    dest.unlink()
                continue
        
        except (EOFError, KeyboardInterrupt):
            return ""


def _fetch_hf_gguf_files(repo_id: str) -> Optional[list]:
    """HuggingFace API ile repo'daki GGUF dosyalarını listele + uyumluluk kontrolü."""
    import urllib.request as urlreq
    
    api_url = f"https://huggingface.co/api/models/{repo_id}"
    
    try:
        req = urlreq.Request(api_url, headers={"User-Agent": "TurboQuant/0.1"})
        with urlreq.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None
    
    # Mimari uyumluluk kontrolü
    tags = data.get("tags", [])
    model_id = data.get("modelId", repo_id)
    
    # Bilinen desteklenen mimariler (llama.cpp Haziran 2026)
    supported_archs = [
        "llama", "mistral", "gemma", "gemma2", "gemma3", "qwen", "qwen2", "qwen3",
        "phi", "phi3", "starcoder", "starcoder2", "falcon", "mpt", "gpt2", "gptj",
        "bloom", "baichuan", "internlm", "internlm2", "yi", "deepseek", "deepseek2",
        "command-r", "dbrx", "olmo", "arctic", "jamba", "mamba", "rwkv",
        "chatglm", "glm4", "minicpm", "orion", "persimmon", "refact",
        "stablelm", "plamo", "codeshell", "viking", "nemotron",
    ]
    
    # Bazı çok yeni mimariler sorun çıkarabilir
    risky_tags = ["custom-code", "trust_remote_code"]
    has_risk = any(t in tags for t in risky_tags)
    
    if has_risk:
        print(f"    ⚠️  Bu model özel kod gerektiriyor (custom architecture)")
        print(f"    ⚠️  llama.cpp ile uyumsuz olabilir — risk sende!")
    
    siblings = data.get("siblings", [])
    
    # GGUF dosyalarını filtrele
    gguf_files = []
    bad_formats = ["mxfp", "mxint", "bnb", "awq", "exl2", "gptq"]
    
    for f in siblings:
        name = f.get("rfilename", "")
        if not name.endswith(".gguf"):
            continue
        
        name_lower = name.lower()
        if any(bad in name_lower for bad in bad_formats):
            continue
        
        # mmproj dosyalarını atla (vision projector, model değil)
        if "mmproj" in name_lower:
            continue
        
        size = f.get("size", 0)
        gguf_files.append({"name": name, "size": size})
    
    # Boyuta göre sırala
    gguf_files.sort(key=lambda x: x["size"])
    
    return gguf_files


def _pick_best_gguf(files: list, repo_id: str) -> Optional[dict]:
    """GGUF dosyaları arasından en uygununu seç veya kullanıcıya seçtir."""
    
    # Öncelik sırası: Q4_K_M > Q4_K_S > IQ4_XS > Q5_K_M > Q3_K_M > ilk bulunan
    preferred = ["Q4_K_M", "Q4_K_S", "IQ4_XS", "IQ4_NL", "Q5_K_M", "Q5_K_S", "Q3_K_M"]
    
    # Otomatik seçim dene
    for pref in preferred:
        for f in files:
            if pref in f["name"]:
                return f
    
    # Bulunamadıysa listeyi göster, kullanıcıya seçtir
    print(f"\n    {len(files)} GGUF dosyası bulundu:")
    print()
    for i, f in enumerate(files):
        size_gb = f["size"] / (1024**3) if f["size"] > 0 else 0
        print(f"      [{i+1}]  {f['name']:<55} {size_gb:>5.1f}G")
    
    print()
    while True:
        try:
            raw = input("    ❯ Dosya seç (numara, Enter = iptal): ").strip()
            if not raw:
                return None
            val = int(raw)
            if 1 <= val <= len(files):
                return files[val - 1]
        except (ValueError, EOFError, KeyboardInterrupt):
            return None


def _get_model_catalog() -> list:
    """İndirilebilir model kataloğu — küçükten büyüğe, VRAM notu ile."""
    return [
        # ── Küçük (2-4 GB VRAM) ──
        {
            "name": "Llama 3.2 1B Instruct (Q4_K_M)",
            "file": "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
            "size_gb": 0.8,
            "url": "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
            "desc": "En küçük, hızlı test",
        },
        {
            "name": "Llama 3.2 3B Instruct (Q4_K_M)",
            "file": "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
            "size_gb": 2.0,
            "url": "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
            "desc": "Hafif, günlük kullanım",
        },
        # ── Orta (6-8 GB VRAM) ──
        {
            "name": "Mistral 7B Instruct v0.3 (Q4_K_M)",
            "file": "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
            "size_gb": 4.4,
            "url": "https://huggingface.co/bartowski/Mistral-7B-Instruct-v0.3-GGUF/resolve/main/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
            "desc": "Hızlı, kod için iyi",
        },
        {
            "name": "Qwen 2.5 7B Instruct (Q4_K_M)",
            "file": "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
            "size_gb": 4.7,
            "url": "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_k_m.gguf",
            "desc": "Güçlü, Türkçe iyi",
        },
        {
            "name": "Llama 3.1 8B Instruct (Q4_K_M)",
            "file": "Llama-3.1-8B-Instruct-Q4_K_M.gguf",
            "size_gb": 4.9,
            "url": "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
            "desc": "En popüler model",
        },
        {
            "name": "Gemma 2 9B Instruct (Q4_K_M)",
            "file": "gemma-2-9b-it-Q4_K_M.gguf",
            "size_gb": 5.4,
            "url": "https://huggingface.co/bartowski/gemma-2-9b-it-GGUF/resolve/main/gemma-2-9b-it-Q4_K_M.gguf",
            "desc": "Google, analiz/reasoning",
        },
        # ── Büyük (10-12 GB VRAM — TurboQuant ile sığar!) ──
        {
            "name": "Qwen 2.5 14B Instruct (Q4_K_M)",
            "file": "Qwen2.5-14B-Instruct-Q4_K_M.gguf",
            "size_gb": 9.0,
            "url": "https://huggingface.co/bartowski/Qwen2.5-14B-Instruct-GGUF/resolve/main/Qwen2.5-14B-Instruct-Q4_K_M.gguf",
            "desc": "14B, Türkçe çok iyi, TQ ile 12GB'a sığar",
        },
        {
            "name": "Mistral Small 3.1 24B Instruct (IQ4_XS)",
            "file": "Mistral-Small-3.1-24B-Instruct-2503-IQ4_XS.gguf",
            "size_gb": 10.3,
            "url": "https://huggingface.co/bartowski/Mistral-Small-3.1-24B-Instruct-2503-GGUF/resolve/main/Mistral-Small-3.1-24B-Instruct-2503-IQ4_XS.gguf",
            "desc": "24B! TQ ile 12GB'a sığar, çok güçlü",
        },
        {
            "name": "Gemma 3 12B Instruct (Q4_K_M)",
            "file": "gemma-3-12b-it-Q4_K_M.gguf",
            "size_gb": 8.1,
            "url": "https://huggingface.co/bartowski/google_gemma-3-12b-it-GGUF/resolve/main/google_gemma-3-12b-it-Q4_K_M.gguf",
            "desc": "Google 12B, reasoning güçlü",
        },
        {
            "name": "DeepSeek R1 Distill Qwen 14B (Q4_K_M)",
            "file": "DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf",
            "size_gb": 9.0,
            "url": "https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-14B-GGUF/resolve/main/DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf",
            "desc": "DeepSeek R1 reasoning, düşünce zinciri",
        },
    ]


def _download_catalog_model(model_info: dict, models_dir: Path) -> str:
    """Katalogdan model indir."""
    import urllib.request as urlreq
    
    models_dir.mkdir(exist_ok=True)
    dest = models_dir / model_info["file"]
    
    # Zaten var mı?
    if dest.exists() and dest.stat().st_size > 1024:
        print(f"\n  ✅ Zaten mevcut: {dest.name}")
        return str(dest)
    
    print(f"\n  📥 İndiriliyor: {model_info['name']} ({model_info['size_gb']:.1f} GB)")
    print(f"     Bu birkaç dakika sürebilir...\n")
    
    try:
        req = urlreq.Request(model_info["url"], headers={"User-Agent": "TurboQuant/0.1"})
        with urlreq.urlopen(req) as resp:
            total = int(resp.headers.get('Content-Length', 0))
            downloaded = 0
            
            with open(str(dest), 'wb') as f:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if total > 0:
                        pct = downloaded * 100 / total
                        filled = int(30 * downloaded / total)
                        bar = "█" * filled + "░" * (30 - filled)
                        mb = downloaded / (1024**2)
                        mb_total = total / (1024**2)
                        sys.stdout.write(f"\r  [{bar}] {pct:.0f}% ({mb:.0f}/{mb_total:.0f} MB)")
                        sys.stdout.flush()
        
        print(f"\n\n  ✅ İndirildi: {dest.name}")
        return str(dest)
    
    except Exception as e:
        print(f"\n  ❌ İndirme hatası: {e}")
        print(f"     Manuel indir: {model_info['url']}")
        print(f"     Dosyayı {models_dir} dizinine koy")
        # Yarım kalan dosyayı sil
        if dest.exists():
            dest.unlink()
        return ""


if __name__ == "__main__":
    main()
