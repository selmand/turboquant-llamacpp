@echo off
chcp 65001 >nul 2>&1
title TurboQuant Kurulum

echo.
echo   ════════════════════════════════════════════════
echo     TurboQuant - Tek Tikla Kurulum (Windows)
echo   ════════════════════════════════════════════════
echo.
echo   NOT: Bu dosyayi cift tikla veya terminalde
echo        "install_and_run.bat" yaz.
echo        "python install_and_run.bat" CALISMAZ.
echo        Python komutu: python install.py
echo.

REM Python kontrolu
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo   [HATA] Python bulunamadi!
    echo.
    echo   Cozum:
    echo     1. https://python.org adresinden Python 3.9+ indir
    echo     2. Kurulumda "Add Python to PATH" kutusunu isle
    echo     3. Bu dosyayi tekrar cift tikla
    echo.
    echo   Veya terminalde: winget install Python.Python.3.12
    echo.
    pause
    exit /b 1
)

echo   [1/3] Python gereksinimleri kuruluyor...
pip install -r requirements.txt --quiet 2>nul
if %ERRORLEVEL% NEQ 0 (
    pip install numpy scipy --quiet
)

echo.
echo   [2/3] TurboQuant kuruluyor...
echo.
python install.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo   [HATA] Kurulum basarisiz.
    echo          Detaylar icin: python diagnose.py
    echo.
    pause
    exit /b 1
)

echo.
echo   ════════════════════════════════════════════════
echo     Kurulum tamamlandi! Sunucu baslatiliyor...
echo   ════════════════════════════════════════════════
echo.
echo   Durdurmak icin: Ctrl+C
echo   Tekrar baslatmak icin: run.bat cift tikla
echo.
python run.py

pause
