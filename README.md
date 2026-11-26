# ScamShield BD

Team name: **ScamShield BD** (NSUCEC NeuraHack 2026, Preliminary round).

## Approach

Multilingual scam-conversation classifier for the three scored tasks:

- **A** — six-class label (`0` Legitimate, `1` MFS/OTP scam, `2` Prize/lottery scam, `3` Fake job/investment, `4` Delivery/parcel phishing, `5` Promotional spam).
- **B** — six techniques `T1`–`T6` (`NONE` when none applies).
- **C** — intended action (six codes).
- **D** — confidence in `[0.03, 0.97]`.

The pipeline uses TF-IDF (word, char skeleton, raw char, first/last message and cue features) plus class-balanced logistic-regression heads for A, B and C, with a logistic confidence calibrator. A delivered Model 2 branch (`paraphrase-multilingual-MiniLM-L12-v2` embeddings + LR, E020) is available for M1+M2 blending and OOD-similarity features; B adopts it only if grouped CV improves. Preprocessing normalises Unicode lookalikes and zero-width characters, preserves `|||` conversation delimiters and speaker markers, flags noise-only rows, and protects hard-legit patterns. Validation uses deterministic 5-fold × 3-repeat grouped CV with duplicate/template clusters (`outputs/validation_results/folds.csv`), leave-cluster-out checks, corruption slices and error analysis. Hand-derived rules are adopted only if they improve grouped CV; see `outputs/validation_results/eda_report.md` and `e001_m1_error_report.md`.

## External resources and licences

No hosted API is used. LLM-generated data and pseudo-labeling remain cut. The pretrained encoder `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (Hugging Face, Apache-2.0) supplies dense embeddings for Model 2 (E020) and OOD similarity; it is declared in the notebook checklist (`MEMORY.md` §11). Hand-written extras were revived by request: `data/extra/handwritten.py` contains 84 team-authored hard-legit/minimal-pair rows written from the task definitions and labeled sample patterns only, with zero raw/clean overlap against `sample.csv` or `test.csv`; B must confirm grouped-CV benefit before final training use. `.env` is empty and present only because the ZIP requires it.

| Resource | Source / provider | Licence | Purpose |
|---|---|---|---|
| scikit-learn, pandas, numpy, scipy | PyPI | BSD | modelling and data handling |
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Hugging Face | Apache-2.0 | Model 2 dense embeddings + OOD similarity |
| Hand-written extra examples | `data/extra/handwritten.py`, our own | n/a | optional training augmentation after grouped-CV check |

Other PyPI packages used by the code are listed in `requirements.txt`. Model 2 downloads its declared encoder weights from Hugging Face during the clean run; no other external dataset is downloaded.

## Seeds

`SEED = 42` in `src/config.py`; `N_SPLITS = 5`, `N_REPEATS = 3`. Do not change the seed without updating `MEMORY.md`.

## Run order

1. Use Python 3.10–3.12 (64-bit), create and activate `.venv`, then `python -m pip install --upgrade pip` and `pip install -r requirements.txt`.
2. Place `sample.csv`, `test.csv` and `sample_submission.csv` in `data/`.
3. `python -m src.training.cross_validation` — rebuild/check canonical folds and OOF artifacts.
4. `python -m src.training.train_baseline` — grouped CV, final fit and calibrated submission.
5. `python -m src.prediction` — write `outputs/predictions/submission.csv` (5 columns, test-row order).
6. `python tools/validate_submission.py` — format and rule checks (C-owned tool).
7. Run the delivered notebook top to bottom in the clean Kaggle/Colab session and confirm the regenerated CSV matches the selected Kaggle submission.
8. `python tools/make_zip.py` — assert exactly four ZIP-root items: `submission.csv`, notebook, `README.md`, `.env` (C-owned tool).

## Approximate runtime

Pending C's clean-run rehearsal: record wall time and peak memory in `MEMORY.md` §10 before filling this section. The 500,000-row run uses cached preprocessing and unique-row batching. Model 2 downloads the declared `paraphrase-multilingual-MiniLM-L12-v2` weights from Hugging Face during the clean run; a GPU is optional but recommended for the 500,000-row embedding pass (CPU extrapolation ≈2.9 h, Kaggle GPU ≈5.5–8.5 min).

## API notes

No API calls. No key is required; `.env` is empty. Rules and examples come from the labeled `sample.csv`, the task definitions, the 84 team-authored hand-written rows in `data/extra/handwritten.py`, and aggregate, unsupervised test statistics only; no test text or label is copied into training, rules or examples.
