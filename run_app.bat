@echo off
echo 🚀 Starting Laptop Price Predictor App...
echo.

:: Check for virtual environment
if exist .venv\Scripts\python.exe (
    echo 🔌 using virtual environment (.venv)...
    .venv\Scripts\python.exe -m streamlit run app_final.py
) else (
    echo 🔌 using system python...
    python -m streamlit run app_final.py
)

if %errorlevel% neq 0 (
    echo.
    echo ❌ The app crashed or failed to start.
    echo ⚠️  Common fixes:
    echo    1. Ensure all dependencies are installed: pip install -r requirements_deploy.txt
    echo    2. Ensure model files exist (run train_model_simple.py)
    pause
)
