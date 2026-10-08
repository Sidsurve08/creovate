@echo off
REM Training script runner for CreoVate ML models
REM Usage: run_all.bat [--fast]

echo ============================================
echo CreoVate ML Training Pipeline
echo ============================================

REM Check if --fast flag is passed
set FAST_MODE=
if "%1"=="--fast" set FAST_MODE=--fast

echo.
echo Step 1: Building datasets...
python training/build_datasets.py
if errorlevel 1 (
    echo ERROR: Dataset building failed
    exit /b 1
)

echo.
echo Step 2: Training text classifier...
python training/train_text_classifier.py %FAST_MODE%
if errorlevel 1 (
    echo WARNING: Text classifier training failed (may require GPU)
    echo Continuing with other models...
)

echo.
echo Step 3: Training price model...
python training/train_price_model.py %FAST_MODE%
if errorlevel 1 (
    echo WARNING: Price model training failed
    echo Continuing with other models...
)

echo.
echo Step 4: Building FAISS index...
python training/build_faiss_index.py
if errorlevel 1 (
    echo WARNING: FAISS index building failed
    echo Continuing...
)

echo.
echo ============================================
echo Training Pipeline Complete
echo ============================================
echo.
echo Check ml_service/eval/ for evaluation results
echo Check ml_service/models/ for saved model artifacts
