"""
TurboQuant Terminal Chat
==========================
Sunucu çalışırken basit sohbet arayüzü.

Kullanım:
    python chat.py                  # Varsayılan port
    python chat.py --port 8080      # Farklı port
    python chat.py --system "Sen bir asistan"  # Sistem mesajı
"""

import json
import sys
import os

# ── .bat koruma ──
_this = os.path.basename(sys.argv[0]) if sys.argv else ""
if _this.endswith(".bat"):
    print(f"\n  ⚠️  '{_this}' Python dosyası değil. Doğrusu: python chat.py\n")
    sys.exit(1)

import argparse
import urllib.request
import urllib.error
from pathlib import Path


CONFIG_FILE = Path(__file__).parent / "config.json"

# Renk kodları (Windows terminal desteği)
if sys.platform == "win32":
    os.system("")  # ANSI escape aktifleştir

CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


def load_port() -> int:
    """config.json'dan port oku."""
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
        return config.get("server_args", {}).get("port", 8080)
    return 8080


def check_server(port: int) -> bool:
    """Sunucu çalışıyor mu?"""
    try:
        req = urllib.request.Request(f"http://localhost:{port}/health")
        with urllib.request.urlopen(req, timeout=3):
            return True
    except Exception:
        return False


def chat_completion(
    port: int,
    messages: list,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    stream: bool = True
) -> str:
    """OpenAI-uyumlu chat completion isteği."""
    url = f"http://localhost:{port}/v1/chat/completions"
    
    payload = {
        "model": "local",
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream,
    }
    
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    
    if stream:
        return _stream_response(req)
    else:
        return _sync_response(req)


def _stream_response(req: urllib.request.Request) -> str:
    """Streaming yanıt — UTF-8 uyumlu chunk okuma."""
    full_text = ""
    
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            line_buffer = b""  # Byte buffer — UTF-8 bozulmasını önler
            
            while True:
                chunk = resp.read(4096)  # 4KB chunk oku (1 byte değil!)
                if not chunk:
                    break
                
                line_buffer += chunk
                
                # Satırlara ayır (SSE formatı: her event \n\n ile ayrılır)
                while b"\n" in line_buffer:
                    line_bytes, line_buffer = line_buffer.split(b"\n", 1)
                    
                    # Byte → string (tam satır olduğu için UTF-8 güvenli)
                    try:
                        line = line_bytes.decode("utf-8").strip()
                    except UnicodeDecodeError:
                        continue
                    
                    if not line or not line.startswith("data: "):
                        continue
                    
                    data_str = line[6:]
                    
                    if data_str == "[DONE]":
                        return full_text
                    
                    try:
                        data = json.loads(data_str)
                        delta = data.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            sys.stdout.write(content)
                            sys.stdout.flush()
                            full_text += content
                    except json.JSONDecodeError:
                        continue
    
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace")
        print(f"\n{RED}Sunucu hatası ({e.code}): {error_body[:200]}{RESET}")
    except urllib.error.URLError as e:
        print(f"\n{RED}Bağlantı hatası: {e.reason}{RESET}")
    except Exception as e:
        print(f"\n{RED}Hata: {e}{RESET}")
    
    return full_text


def _sync_response(req: urllib.request.Request) -> str:
    """Senkron yanıt (tek seferde)."""
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            content = data["choices"][0]["message"]["content"]
            sys.stdout.write(content)
            sys.stdout.flush()
            return content
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace")
        print(f"{RED}Sunucu hatası ({e.code}): {error_body[:200]}{RESET}")
        return ""
    except Exception as e:
        print(f"{RED}Hata: {e}{RESET}")
        return ""


def main():
    parser = argparse.ArgumentParser(description="TurboQuant Terminal Chat")
    parser.add_argument("--port", type=int, default=None, help="API port")
    parser.add_argument("--system", type=str, default=None, help="Sistem mesajı")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sıcaklık (0-2)")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Max yanıt token")
    parser.add_argument("--no-stream", action="store_true", help="Streaming kapat")
    args = parser.parse_args()
    
    port = args.port or load_port()
    
    # Sunucu kontrolü
    if not check_server(port):
        print(f"{RED}❌ Sunucu çalışmıyor (port {port}){RESET}")
        print(f"   Önce başlat: {YELLOW}python run.py{RESET}")
        print(f"   Veya: {YELLOW}python chat.py --port XXXX{RESET}")
        sys.exit(1)
    
    # Sohbet geçmişi
    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    
    # Başlık
    print()
    print(f"{BOLD}╔═══════════════════════════════════════════╗{RESET}")
    print(f"{BOLD}║  💬 TurboQuant Chat                       ║{RESET}")
    print(f"{BOLD}╠═══════════════════════════════════════════╣{RESET}")
    print(f"{BOLD}║{RESET}  {DIM}Sunucu: localhost:{port}{RESET}                    {BOLD}║{RESET}")
    print(f"{BOLD}║{RESET}  {DIM}Çıkış: quit / exit / Ctrl+C{RESET}              {BOLD}║{RESET}")
    print(f"{BOLD}║{RESET}  {DIM}Geçmişi sıfırla: /clear{RESET}                  {BOLD}║{RESET}")
    print(f"{BOLD}╚═══════════════════════════════════════════╝{RESET}")
    print()
    
    try:
        while True:
            # Kullanıcı girişi
            try:
                user_input = input(f"{GREEN}Sen ► {RESET}").strip()
            except EOFError:
                break
            
            if not user_input:
                continue
            
            # Komutlar
            if user_input.lower() in ("quit", "exit", "q", "çık"):
                print(f"\n{DIM}Görüşmek üzere! 👋{RESET}")
                break
            
            if user_input.lower() == "/clear":
                messages = []
                if args.system:
                    messages.append({"role": "system", "content": args.system})
                print(f"{DIM}  Sohbet geçmişi sıfırlandı.{RESET}")
                continue
            
            if user_input.lower() == "/info":
                print(f"{DIM}  Mesaj sayısı: {len(messages)}")
                print(f"  Port: {port}")
                print(f"  Sıcaklık: {args.temperature}")
                print(f"  Max token: {args.max_tokens}{RESET}")
                continue
            
            # Mesajı ekle
            messages.append({"role": "user", "content": user_input})
            
            # Yanıt al
            print(f"\n{CYAN}AI ► {RESET}", end="")
            
            response = chat_completion(
                port=port,
                messages=messages,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                stream=not args.no_stream,
            )
            
            print("\n")
            
            # Yanıtı geçmişe ekle
            if response:
                messages.append({"role": "assistant", "content": response})
            
    except KeyboardInterrupt:
        print(f"\n\n{DIM}Görüşmek üzere! 👋{RESET}")


if __name__ == "__main__":
    main()
