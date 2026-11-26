# MEMORY — ScamShield BD (project log and handoff)

> The living record. **Read this first at the start of every work session; update it before you stop.**
> Anyone (teammate, AI assistant, future us) should be able to pick up from here. Keep entries short, dated (BST) and factual. Mark unverified items with `[?]`.

---

## 1. Snapshot

| Item | Value |
|---|---|
| Competition | NSUCEC NeuraHack 2026, Preliminary round (Kaggle Community Competition) |
| Project | ScamShield BD: multilingual scam conversation intelligence |
| Tasks | A six-class label · B techniques T1–T6 · C intended action · D confidence |
| Score | `S = 0.40·F_A + 0.15·F_B + 0.10·F_C + 0.10·F_domain + 0.10·F_robust + 0.05·Cal + 0.10·F_novel` |
| Data | `sample.csv` (≈86 labeled `[?]` confirm in EDA), `test.csv` (≈500,000), `sample_submission.csv` |
| Prelims | 26 Nov 18:00 → 27 Nov 18:00 BST |
| **ZIP deadline** | **27 Nov 22:00 BST** |
| Results | 28 Nov 14:00 · Final hackathon 4 Dec |
| Kaggle snapshot (26 Nov ~23:04) | 97 teams · 265 submissions · ~19 hours left |
| Team split | Teammate: inspection, validation, preprocessing. Nabila: modelling, augmentation, calibration, submission, packaging |
| Local env | Windows, Antigravity IDE, PowerShell, `.venv` active; repo at `C:\Users\HP\Documents\ScamShield-BD` |
| Installed libs | pandas 3.1.1, numpy 2.5.5, scipy 1.18.1, scikit-learn 1.9.1, lightgbm 4.7.0, sentence-transformers 6.3.0 (torch 2.15.0, transformers 5.24.1), matplotlib 3.11.2, seaborn 0.13.2, jupyter 1.1.1, ipykernel 7.4.0 |

## 2. Verified competition facts (from the documents)

- Label priority: **1 → 2 → 4 → 3 → 5 → 0**.
- Conversations: up to **12 messages**, separated by `|||`; links defanged (`example[.]com`); speaker markers may appear.
- Adversarial text includes: character insert/delete/substitute/repeat, missing spaces, random caps, emojis, abbreviations, transliteration variants, **Unicode lookalikes, zero-width characters**, embedded noise strings (`wkrjfejejrn`), and **noise-only records**.
- Hard-legit families: genuine OTP with "do not share", transaction confirmations, delivery tracking, bill reminders, genuine promos/cashback, service notices with links, **a friend warning about a scam**.
- Public LB = 30% of rows; private = 70% and decides rank. Hidden subsets are unmarked and may overlap; subsets absent from the public portion are dropped and weights renormalised.
- Calibration = 1 − Brier(confidence, correctness of Task A).
- Task B macro-F1 is over T1–T6 computed over **all** test rows; `NONE` for non-fraudulent.
- Kaggle picks the top two by **public** score if we don't select finals manually.
- Pseudo-labeling and test-text domain adaptation: **allowed**, by code, model predictions only, declared.
- Hosted LLM APIs: **allowed** for labeling, generation, augmentation, inference, from notebook code, declared, key in `.env`.
- Top 60 by private score → notebook review (reproducibility, leakage-free validation, compliance, generalisation analysis, documentation) → Top 35. Extra unseen evaluation data may also be used.
- Reproduction tolerance: private score within **0.01**.

## 3. Discrepancies between sources (resolve before packaging)

| # | Topic | Source A | Source B | Action |
|---|---|---|---|---|
| D1 | Kaggle file columns | Rulebook v1.3 §4.1: `id,label` only | Kaggle overview + problem statement: `id,label,techniques,action,confidence` | Follow `sample_submission.csv`; confirm in Kaggle UI/Data tab |
| D2 | ZIP name | Rulebook v1.3: `NeuraHack2026_<TeamName>_<TeamCode>.zip` | Problem statement: `NeuraHack_Preli_<TeamName>.zip` | Check the guidance email / official channel; set one constant in `make_zip.py` |
| D3 | Evaluation weights | Rulebook §3 says plain Macro F1 "as described on the Kaggle page" | Kaggle page: composite S with 7 components | Treat the Kaggle page as authoritative |
| D4 | Frozen LLM data in notebook | Rules require reproducibility and tolerate non-deterministic LLM output | Not explicitly stated whether frozen embedded data is accepted | Ask in the official channel; keep generator code + frozen copy |

## 4. Decisions log

| ID | Date/time | Decision | Reason |
|---|---|---|---|
| D-001 | 26 Nov 23:15 | Notebook is **generated** from `src/` (`%%writefile` modules) | Dev code = shipped code; avoids drift |
| D-002 | 26 Nov 23:15 | Three text views: raw / norm / skeleton | Aggressive cleaning destroys scam cues; skeleton is for char models only |
| D-003 | 26 Nov 23:15 | Validate on real rows only; synthetic/augmented rows inherit source `group_id` | Prevent leakage |
| D-004 | 26 Nov 23:15 | Model 1 (TF-IDF + LR) ships first; transformers are gated | Small data; benchmark first |
| D-005 | 26 Nov 23:15 | Task B as six binary heads with per-technique thresholds | Metric is macro-F1 per technique over all rows |
| D-006 | 26 Nov 23:15 | Interfaces in `ARCHITECTURE.md` §4 are frozen; changes get logged here | Parallel work |
| D-007 | 26 Nov 23:15 | Select two finals manually: one conservative, one with CV-positive extras | Public LB is only 30%; default selection is by public score |
| D-008 | 27 Nov 00:20 | Canonical saved folds live in `outputs/validation_results/folds.csv` (unified `fold_r0`..`fold_r2` across Tasks A/B/C + sparse duplicate `group` + `cluster` for leave-cluster-out), instead of `outputs/folds.json` | Single deterministic partition across all models/tasks; OOF arrays saved to `outputs/oof/<exp>_m1.npz` |
| D-009 | 27 Nov 02:00 | `TfidfBlock` uses dict-based interface (`train_d, test_d` → `csr_matrix`); `build_text_dict(df)` is the bridge from DataFrame to dict | Richer than raw text list: carries text_norm, text_skel, first_msg, last_msg, struct_feats; falls back to `preprocess()` until A delivers `build_views`. B notified A and C. |

## 5. Findings from EDA `[to be filled by teammate]`

| Question | Answer |
|---|---|
| Number of labeled rows / class counts | |
| Language mix (English / Bangla / Banglish / mixed) | |
| Single messages vs conversations; max messages | |
| Speaker-marker format | |
| Duplicate / template clusters found | |
| Cross-tab `label × techniques × action` (rules that hold ≥95%) | |
| Label/techniques/action for **noise-only** rows | |
| Class 5 techniques/action | |
| Test: noise ratio distribution, language mix, class histogram of baseline predictions | |

## 6. Assumptions to verify

- [?] `sample.csv` has about 86 rows (taken from the project notes).
- [?] Class 5 and class 0 both map to techniques `NONE`.
- [?] Label 0 implies action 0.
- [?] Noise-only records are labeled class 0 `[?]`.
- [x] Kaggle GPU is available for embeddings; session time is sufficient for 500k rows.
  * Benchmark on 10,000 rows of `test.csv` on `paraphrase-multilingual-MiniLM-L12-v2`: 209.79 s (47.7 rows/s on local CPU).
  * Extrapolated for 500,000 rows on CPU: 174.8 minutes (2.91 hours).
  * Extrapolated for 500,000 rows on Kaggle GPU (T4/P100 fp16, ~1,000–1,500 rows/s): estimated 5.5–8.5 minutes (well within GPU limits).
  * `multilingual-e5-base` (~3x heavier) is dropped to preserve comfortable GPU session margins.
- [?] Daily submission limit resets at 00:00 UTC (06:00 BST).
- [?] The chosen embedding model(s) can be downloaded in the clean Kaggle/Colab run (internet on).

## 7. Experiment log

> One row per run (also appended automatically to `outputs/validation_results/experiments.csv`).

| Exp | Date/time | Change vs previous | F_A | F_B | F_C | Cal (1−Brier) | S_hat | Robust-slice F1 | Notes |
|---|---|---|---|---|---|---|---|---|---|
| E001 | 27 Nov 02:32 | Baseline word+char TF-IDF + LR | 0.524 | 0.291 | 0.374 | 0.761 | — | 0.482 | 5x3 grouped CV (`folds.csv`), no aug, C=5 |
| E015 | 27 Nov 02:45 | `word+char` only, `C=5`, `n_aug=5` (Set A) + Task B thresholds + `postprocess` | 0.574 | 0.578 | 0.475 | 0.782 | 0.498 | 0.538 | `F_domain=0.316`; thresholds `[0.31,0.39,0.29,0.26,0.4,0.26]` |
| E018 | 27 Nov 03:10 | Phase 3: Blend `0.4·E015(LR) + 0.6·E016(LinearSVC)` + Logistic Calibration | 0.614 | 0.578 | 0.475 | 0.777 | 0.514 | 0.538 | Cal beats const Brier (`0.2228` vs `0.2305`) |
| E020 | 27 Nov 03:57 | Model 2: paraphrase-multilingual-MiniLM-L12-v2 + LR (T-023) | 0.575 | 0.566 | 0.343 | 0.684 | 0.556 | — | 5x3 grouped CV on text_norm embeddings, C=1.0 |

## 8. Kaggle submission log

| # | Date/time | File | What changed | Public score | Selected as final? |
|---|---|---|---|---|---|
| 1 | 27 Nov 02:32 | `outputs/predictions/submission_E1126_2032.csv` | Baseline (format check, E001) | *(pending upload)* | No |
| 2 | 27 Nov 03:04 | `outputs/predictions/submission_E1126_2104.csv` | Phase 2 E015 (`word+char`, `n_aug=5`, tuned Task B thresholds, `postprocess.py`) | *(pending upload)* | Candidate A |
| 3 | 27 Nov 03:29 | `outputs/predictions/submission_E1126_2129.csv` | Phase 3 E018 (`0.4·LR + 0.6·LinearSVC` blend, logistic calibration, Task B thresholds, `postprocess.py`) | *(pending upload)* | Candidate B |

## 9. Lessons and gotchas

- Run in each new PowerShell: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned` then `.\.venv\Scripts\Activate.ps1`.
- `check_noise.py` (in repo root) inspects frequent tokens in test text; use for noise statistics only (unsupervised).
- With ~86 rows, differences of one or two F1 points are noise; prefer simple models within noise.
- Don't let any rule or example be derived from specific test rows.

## 10. Handoff notes (rewrite at the end of every session)

```
Last updated: 27 Nov 03:59 BST by C
State:
  DONE  — T-023: src/features/embedding_features.py implemented with length-sorted batching,
          L2-normalisation, dynamic padding, GPU fp16 support, and disk caching.
  DONE  — T-023: Benchmark 10k test rows on paraphrase-multilingual-MiniLM-L12-v2:
          209.8s (47.7 rows/s on CPU); extrapolated to 2.9h on CPU vs ~6-8 min on Kaggle GPU.
  DONE  — T-023: max_cosine_to_train implemented with diagonal masking for training cross-fitting.
  DONE  — T-023: src/training/train_embeddings.py implemented and executed (E020).
          Saved outputs/oof/E020_m2.npz (keys: ids, repeat, A, B, C, ood_sim, y_a, y_b, y_c, noise_only, noise_ratio, clean_proba_a, clean_proba_b, clean_proba_c, folds).
          Saved outputs/predictions/E020_m2_test.npz (keys: ids, A, B, C, ood_sim, shape (500000, 6) / (500000,)).
          Logged E020 to experiments.csv: F_A=0.5748, F_B=0.5663, F_C=0.3428, Cal=0.6835, S_hat=0.5558.
  DONE  — Added pytest>=7.0 to requirements.txt and verified test suite.
Next 3 steps for Teammate B:
  1. Blend M1 (E015/E018) + M2 (E020) using tune_blend_weight with outputs/oof/E020_m2.npz.
  2. Use ood_sim from outputs/oof/E020_m2.npz as extra_feats in fit_confidence to boost calibration.
  3. Predict test with blended M1+M2 and generate final submission candidates.
Next 3 steps for Teammate C:
  1. Implement tools/make_zip.py (C4) packaging final submission, notebook, .env, and README.
  2. Assist with clean-run rehearsal of notebook (C3).
  3. Verify compliance checklist items 7 & 8 and prepare ZIP for upload (S-07, S-08).
Blockers / questions for teammate:
  - C has delivered outputs/oof/E020_m2.npz and outputs/predictions/E020_m2_test.npz. Teammate B is unblocked!
Where results live: outputs/validation_results/experiments.csv ; outputs/oof/E020_m2.npz ; outputs/predictions/E020_m2_test.npz
```

## 11. Declaration checklist (feeds the notebook's first cells)

| Resource | Type | Source / provider | Licence | Purpose | Declared in notebook? |
|---|---|---|---|---|---|
| scikit-learn, pandas, numpy, scipy | libraries | PyPI | BSD | modelling/data | [ ] |
| sentence-transformers + sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | pretrained model | Hugging Face | Apache-2.0 | multilingual dense sentence embeddings for Model 2 and OOD similarity | [x] |
| LLM API (if used) | hosted API | `<provider/model>` | provider terms | synthetic training data | [ ] |
| Hand-written extra examples | data | our own | n/a | training | [ ] |
| Pseudo-labeling on `test.csv` text | method | our code | n/a | semi-supervised (if adopted) | [ ] |

## 12. Glossary

- **MFS:** mobile financial service (bKash, Nagad, Rocket, etc.). **Banglish:** Bangla written in Latin letters.
- **OOF:** out-of-fold predictions made on rows the model did not train on.
- **Group / template cluster:** near-duplicate messages kept together in folds.
- **Minimal pair:** a legitimate and a scam message sharing most wording.
- **S_hat:** our surrogate composite score computed from CV and slice proxies.
