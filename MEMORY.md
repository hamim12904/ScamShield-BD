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
- [?] Kaggle GPU is available for embeddings; session time is sufficient for 500k rows.
- [?] Daily submission limit resets at 00:00 UTC (06:00 BST).
- [?] The chosen embedding model(s) can be downloaded in the clean Kaggle/Colab run (internet on).

## 7. Experiment log

> One row per run (also appended automatically to `outputs/validation_results/experiments.csv`).

| Exp | Date/time | Change vs previous | F_A | F_B | F_C | Cal (1−Brier) | S_hat | Robust-slice F1 | Notes |
|---|---|---|---|---|---|---|---|---|---|
| E001 | 27 Nov 02:32 | Baseline word+char TF-IDF + LR | 0.524 | 0.291 | 0.374 | 0.761 | — | 0.482 | 5x3 grouped CV (`folds.csv`), no aug, C=5 |
| E015 | 27 Nov 02:45 | `word+char` only, `C=5`, `n_aug=5` (Set A) + Task B thresholds + `postprocess` | 0.574 | 0.578 | 0.475 | 0.780 | 0.498 | 0.538 | `F_domain=0.316`; thresholds `[0.31,0.39,0.29,0.26,0.4,0.26]` |

## 8. Kaggle submission log

| # | Date/time | File | What changed | Public score | Selected as final? |
|---|---|---|---|---|---|
| 1 | 27 Nov 02:32 | `outputs/predictions/submission_E1126_2032.csv` | Baseline (format check, E001) | *(pending upload)* | No |

## 9. Lessons and gotchas

- Run in each new PowerShell: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned` then `.\.venv\Scripts\Activate.ps1`.
- `check_noise.py` (in repo root) inspects frequent tokens in test text; use for noise statistics only (unsupervised).
- With ~86 rows, differences of one or two F1 points are noise; prefer simple models within noise.
- Don't let any rule or example be derived from specific test rows.

## 10. Handoff notes (rewrite at the end of every session)

```
Last updated: 27 Nov 02:47 BST by B
State:
  DONE  — Phase 1: E001 baseline submitted/validated (submission_E1126_2032.csv).
  DONE  — Phase 2 T-020: augment.py Set A/B tested (tests/test_augment.py, 5 passed) & integrated into training folds.
  DONE  — Phase 2 T-021: 16-experiment tuning sweep (E002-E017) on saved 5x3 grouped CV.
          Winner E015 (word+char only, C=5, n_aug=5): F_A 0.524->0.574, F_rob 0.482->0.538, F_dom 0.231->0.316, S_hat 0.423->0.498.
  DONE  — Phase 2 T-024: postprocess.py (4 rules at 100% support in sample.csv) + Task B threshold tuning (F_B 0.416->0.578).
  DONE  — prediction.py updated with E015 + Task B thresholds + postprocess.py; OOF saved at outputs/oof/E015_m1.npz.
  PENDING — Phase 3: T-030 ensemble.py (waiting for C's Model 2 OOF outputs/oof/*_m2.npz), T-031 calibration.py.
Next 3 steps:
  1. Wait for C's Model 2 OOF (outputs/oof/*_m2.npz) and A's handwritten extras (data/extra/handwritten.py).
  2. T-030 & T-031: Blend M1 + M2 in ensemble.py and fit/cross-fit calibration.py (must beat constant Brier baseline).
  3. T-040 & T-041: Generate & submit Candidate A (E015 Model 1) and Candidate B (Ensemble + calibrated).
Blockers / questions for teammate:
  - A: build_views() not yet in preprocessing.py; prediction.py falls back to preprocess_frame().
       Once your build_views is merged, update tfidf_features.build_text_dict to use
       text_norm, text_skel, first_msg columns from the DataFrame. Notify B by commit.
  - C: validate_submission.py stub is in tools/; replace with your full version (T-006).
       Also add pytest>=7.0 to requirements.txt (not in C's scope? cross-request to C).
  - Cross-team request D-009: A please add text_norm/text_skel/first_msg/last_msg to build_views
       so B can use richer features (see tfidf_features.build_text_dict for the keys needed).
Where results live: outputs/validation_results/experiments.csv ; outputs/folds.json ; outputs/predictions/
```

## 11. Declaration checklist (feeds the notebook's first cells)

| Resource | Type | Source / provider | Licence | Purpose | Declared in notebook? |
|---|---|---|---|---|---|
| scikit-learn, pandas, numpy, scipy | libraries | PyPI | BSD | modelling/data | [ ] |
| sentence-transformers + chosen encoder | pretrained model | Hugging Face | check model card | embeddings | [ ] |
| LLM API (if used) | hosted API | `<provider/model>` | provider terms | synthetic training data | [ ] |
| Hand-written extra examples | data | our own | n/a | training | [ ] |
| Pseudo-labeling on `test.csv` text | method | our code | n/a | semi-supervised (if adopted) | [ ] |

## 12. Glossary

- **MFS:** mobile financial service (bKash, Nagad, Rocket, etc.). **Banglish:** Bangla written in Latin letters.
- **OOF:** out-of-fold predictions made on rows the model did not train on.
- **Group / template cluster:** near-duplicate messages kept together in folds.
- **Minimal pair:** a legitimate and a scam message sharing most wording.
- **S_hat:** our surrogate composite score computed from CV and slice proxies.
