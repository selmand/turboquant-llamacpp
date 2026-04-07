@echo off
chcp 65001 >nul 2>&1
title TurboQuant Chat
echo.
echo   NOT: "python chat.bat" CALISMAZ. Dogrusu: python chat.py
echo.
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo   [HATA] Python bulunamadi!
    pause
    exit /b 1
)
python chat.py %*
pause
