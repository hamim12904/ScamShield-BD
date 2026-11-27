# ScamShield BD

Team name: **ScamShield BD** (NSUCEC NeuraHack 2026, Preliminary round).

## Approach

Multilingual scam-conversation classifier for the three scored tasks:

- **A** — six-class label (`0` Legitimate, `1` MFS/OTP scam, `2` Prize/lottery scam, `3` Fake job/investment, `4` Delivery/parcel phishing, `5` Promotional spam).
- **B** — six techniques `T1`–`T6` (`NONE` when none applies).
- **C** — intended action (six codes).
- **D** — confidence in `[0.03, 0.97]`.

The final candidate is **Step 5 / E028**: 5-fold × 3-repeat grouped CV combining E026 logistic regression, E027 linear SVM, the promoted LLM + pseudo-label extras, equal M1/M2 blend weights (0.50/0.50) for Tasks A/B/C, the full-test E020 Model 2 branch (`paraphrase-multilingual-MiniLM-L12-v2` + OOD similarity), cross-fitted Task B thresholds, and a constant-confidence fallback. Model 1 uses word TF-IDF on `text_norm` (preserved text) and char TF-IDF on `text_skel` (skeleton) with structure features; message blocks and the raw-char branch are disabled. Preprocessing normalises Unicode lookalikes and zero-width characters, preserves `|||` conversation delimiters and speaker markers, flags noise-only rows, and protects hard-legit patterns. Validation uses deterministic saved folds (`outputs/validation_results/folds.csv`) with duplicate/template isolation, leave-cluster-out checks, corruption slices and error analysis. See `outputs/validation_results/eda_report.md`, `e001_m1_error_report.md` and `test_diagnosis_step5.md`.

## External resources and licences

The clean run uses frozen `llm_labels_v1.jsonl` and `pseudo_labels_v1.jsonl`; with `REGENERATE_SYNTH=False` it needs **no hosted API key**. `data/extra/handwritten.py` is declared but is not used by the final E028 candidate. Model 2 downloads the declared Hugging Face encoder weights during the clean run. `.env` is empty and present because the ZIP requires it.

| Resource | Source / provider | Licence | Purpose |
|---|---|---|---|
| scikit-learn, pandas, numpy, scipy, joblib, regex, ftfy, Unidecode, torch, python-dotenv | PyPI | BSD/MIT | modelling, preprocessing, data handling, clean-run install |
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Hugging Face | Apache-2.0 | Model 2 dense embeddings + OOD similarity |
| Frozen `llm_labels_v1.jsonl` | optional hosted LLM, generated ahead of time | Provider terms | promoted supervised extra; embedded; no key needed for clean run |
| Frozen `pseudo_labels_v1.jsonl` | our semi-supervised code | n/a | promoted pseudo-label extra derived from competition test text; embedded |
| Hand-written extra examples | `data/extra/handwritten.py`, our own | n/a | declared only; not used in final E028 candidate |

Other PyPI packages used by the code are listed in `requirements.txt`. The notebook embeds modules, frozen label files and `folds.csv` through generated `%%writefile` cells; it does not embed secrets.

## Seeds

`SEED = 42` in `src/config.py`; `PYTHONHASHSEED=42`; `N_SPLITS = 5`, `N_REPEATS = 3`. Do not change the seed without updating `MEMORY.md`.

## Run order

1. Use Python 3.10–3.12 (64-bit), create and activate `.venv`, then `python -m pip install --upgrade pip` and `pip install -r requirements.txt`.
2. Place `sample.csv`, `test.csv` and `sample_submission.csv` in `data/` (or under a nested `/kaggle/input/<competition>/data/` directory).
3. Local development: `python -m src.training.cross_validation`, then `python -m src.prediction`, then `python tools/validate_submission.py`.
4. Rebuild the generated notebook after any source edit: `python tools/build_notebook.py`. Never hand-edit `notebooks/notebook.ipynb`.
5. 1,000-row smoke test: `python tools/build_notebook.py --execute-slice 1000` (or set `SCAMSHIELD_TEST_SLICE=1000` before `jupyter nbconvert --execute`).
6. Clean full run: execute `notebooks/notebook.ipynb` top to bottom in Kaggle/Colab. Set `SCAMSHIELD_COMPARE_SUBMISSION=<path>` to compare the regenerated CSV with a reference; byte equality is also reported.
7. `python tools/make_zip.py` — assert exactly four ZIP-root items: `submission.csv`, notebook, `README.md`, `.env` (C-owned tool).

## Approximate runtime

An isolated 1,000-row `jupyter nbconvert --execute` smoke test passed in **1057.30 s** wall time with **1146.3 MB** peak traced RAM on CPU; it loaded the embedded `folds.csv` and wrote a strictly valid `submission.csv`. The full 500,000-row embedding pass is expected to take roughly **3 h on CPU** or **5.5–8.5 min on a Kaggle GPU**; the whole clean run is dominated by that pass. Preprocessing is cached and `predict_df` runs in 50,000-row chunks. The notebook prints exact library versions, CUDA availability and timing; update `MEMORY.md` §10 after a full clean run.

## API notes

No API calls are made during the clean run and no key is required; `.env` is empty. `REGENERATE_SYNTH=False`, so the embedded frozen LLM and pseudo-label files are used instead. Hand-written rules and examples come only from the task definitions and labeled sample patterns; no raw test text is copied into those rules/examples. The frozen pseudo-label file is a declared semi-supervised artifact derived from competition test text.
