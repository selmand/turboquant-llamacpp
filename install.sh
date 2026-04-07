#!/bin/bash
# TurboQuant - Tek komut kurulum (Linux/Mac)
# Kullanım: chmod +x install.sh && ./install.sh

set -e

echo ""
echo "╔═══════════════════════════════════════════════════╗"
echo "║   TurboQuant - Tek Komut Kurulum                  ║"
echo "╚═══════════════════════════════════════════════════╝"
echo ""

# Python kontrol
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 bulunamadı!"
    echo "   Ubuntu/Debian: sudo apt install python3 python3-pip"
    echo "   Mac: brew install python3"
    exit 1
fi

# pip gereksinimleri
echo "📦 Python gereksinimleri kuruluyor..."
pip3 install -r requirements.txt --quiet 2>/dev/null || pip3 install numpy scipy --quiet

# Kurulum
echo ""
python3 install.py

echo ""
echo "Başlatma komutları:"
echo "  python3 run.py     ← sunucu"
echo "  python3 chat.py    ← terminal sohbet"
