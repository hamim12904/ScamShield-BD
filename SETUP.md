# ScamShield-BD: Team Setup

## 1. Install (once)
- Python 3.10, 3.11 or 3.12 (64-bit): https://www.python.org/downloads/
  (Windows: tick "Add python.exe to PATH" in the installer)
- Git: https://git-scm.com/downloads
- VS Code: https://code.visualstudio.com/ with the extensions "Python", "Jupyter", "GitLens" (optional)

## 2. Clone and open
```bash
git clone <REPO_URL> ScamShield-BD
cd ScamShield-BD
code .
```

## 3. Create the environment
Windows (PowerShell):
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```
macOS / Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 4. Install packages (same versions for everyone)
```bash
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## 5. Select the interpreter
Ctrl+Shift+P -> "Python: Select Interpreter" -> choose the one inside `.venv`.

## 6. Add the data
Download `test.csv` and `sample_submission.csv` from the Kaggle competition page and place them in `data/`.
(`sample.csv` is already in the repo.)

## 7. Verify
```bash
python -c "import pandas, sklearn, torch, sentence_transformers, ftfy; print('OK')"
python -m src.text_analysis
```

## Rules
- Never commit `.env`, `data/test.csv`, `outputs/` or model files.
- Random seed is `SEED = 42` in `src/config.py`. Do not change it without telling the team.
- Declare every external model, dataset or API in the notebook (competition rule).