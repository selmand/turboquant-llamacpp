@echo off
chcp 65001 >nul 2>&1
title TurboQuant Server
echo.
echo   ════════════════════════════════════════
echo     TurboQuant Server
echo   ════════════════════════════════════════
echo.
echo   NOT: Bu dosyayi cift tikla veya terminalde "run.bat" yaz.
echo        "python run.bat" veya "py run.bat" CALISMAZ.
echo        Python komutu: python run.py
echo.

python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo   [HATA] Python bulunamadi!
    echo          https://python.org adresinden Python 3.9+ indir
    echo          Kurulumda "Add Python to PATH" kutusunu isle
    echo.
    pause
    exit /b 1
)

python run.py %*
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo   [!] Bir sorun olustu.
    echo       Detaylar icin: python diagnose.py
    echo.
)
pause
