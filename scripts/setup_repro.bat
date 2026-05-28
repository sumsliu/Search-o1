@echo off
REM Setup Search-o1 API reproduction environment (Windows)
set CONDA_ROOT=%USERPROFILE%\miniconda3
call "%CONDA_ROOT%\Scripts\activate.bat" search_o1

cd /d "%~dp0.."
echo Installing local pyext...
pip install -q scripts/lcb_runner/pyext/pyext-0.7

echo Downloading NLTK data...
python -c "import nltk; nltk.download('punkt_tab', quiet=True); nltk.download('punkt', quiet=True)"

echo Downloading NQ dataset (50 samples)...
python scripts/download_data.py --dataset nq --limit 50

echo Setup complete. Run reproduction from project root:
echo   python scripts/run_search_o1_api.py --dataset_name nq --split test --model_variant flash --subset_num 3
