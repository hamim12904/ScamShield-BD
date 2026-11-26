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
| Role mapping (F-09) | A = Teammate; B and C = Nabila (B takes C's P0 validator/notebook items if needed) |
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

## 5. Findings from EDA (27 Nov, A1; full detail in `outputs/validation_results/eda_report.md`)

| Question | Answer |
|---|---|
| Number of labeled rows / class counts | 86 confirmed; 0:32, 1:12, 2:10, 3:11, 4:9, 5:12 |
| Language mix (English / Bangla / Banglish / mixed) | sample en 68.6% / banglish 24.4% / bangla 7.0% / mixed 0%; test en 56.1% / banglish 32.9% / mixed 10.4% / bangla 0.6% |
| Single messages vs conversations; max messages | sample 78 single / 8 conv (max 5); test 392,752 single / 107,248 conv (max 12) |
| Speaker-marker format | canonical `S/R/A/B/SENDER/YOU/CALLER/RECEIVER:` at turn start (`SPEAKER_RE` + `SPEAKER_CANON`); sample 8 rows, test 67,270; raw `^\s*[A-Za-z]{1,10}\s*:` over-matches `hxxps:`/`Task:` |
| Duplicate / template clusters found | test normalized duplicates 139,741 (27.95%), 360,259 unique templates; sample 0; 5 sample keys also in test; folds: 85 groups, 3 clusters |
| Cross-tab `label × techniques × action` (rules ≥95%) | 8 rules at 100%: 0→action 0, 5→action 0, 0/5→NONE, 2→T4, 3→T6, T2→label 1, T6→label 3; full tables in `eda_report.md` §6 |
| Label/techniques/action for **noise-only** rows | 2 rows, both 0 / NONE / 0 (ids 10058, 10081, noise_ratio 1.0); n=2, still an assumption (F-05) |
| Class 5 techniques/action | NONE 12/12, action 0 12/12 |
| Test: noise ratio distribution, language mix, share with lookalikes or zero-width | 50k subsample: noise_ratio [0,.01):43,968, [.01,.1):4,496, [.1,.25):1,427, [.25,.5):16, [.5,.75):5, [.99,1]:88; noise_only 0.176%; lookalikes 8.50%, zero-width 6.73%; language as above. Baseline class histogram is B's model output, not part of unsupervised EDA |

## 6. Assumptions to verify

- [x] `sample.csv` has 86 rows (A1 confirmed).
- [x] Class 5 and class 0 both map to techniques `NONE` (12/12 and 32/32).
- [x] Label 0 implies action 0 (32/32); label 5 also maps to action 0 (12/12).
- [?] Noise-only records are labeled class 0 — supported by 2/2 sample rows (ids 10058, 10081); n too small, keep as assumption until B aligns the rule (F-05).
- [x] Embedding/GPU feasibility benchmarked and Model 2 revived by C: 10k `paraphrase-multilingual-MiniLM-L12-v2` rows = 209.79 s (47.7 rows/s CPU); 500k CPU ≈174.8 min; Kaggle GPU ≈5.5–8.5 min; `multilingual-e5-base` dropped.
- [?] Daily submission limit resets at 00:00 UTC (06:00 BST).
- [?] D1: 5-column submission format follows `sample_submission.csv`; B/C to confirm in the Kaggle Data tab.

## 7. Experiment log

> One row per run (also appended automatically to `outputs/validation_results/experiments.csv`).

| Exp | Date/time | Change vs previous | F_A | F_B | F_C | Cal (1−Brier) | S_hat | Robust-slice F1 | Notes |
|---|---|---|---|---|---|---|---|---|---|
| E001 | 27 Nov 02:32 | Baseline word+char TF-IDF + LR | 0.524 | 0.291 | 0.374 | 0.761 | — | 0.482 | 5x3 grouped CV (`folds.csv`), no aug, C=5 |
| E015 | 27 Nov 02:45 | `word+char` only, `C=5`, `n_aug=5` (Set A) + Task B thresholds + `postprocess` | 0.574 | 0.578 | 0.475 | 0.782 | 0.498 | 0.538 | `F_domain=0.316`; thresholds `[0.31,0.39,0.29,0.26,0.4,0.26]` |
| E018 | 27 Nov 03:10 | Phase 3: Blend `0.4·E015(LR) + 0.6·E016(LinearSVC)` + Logistic Calibration | 0.614 | 0.578 | 0.475 | 0.777 | 0.514 | 0.538 | Cal beats const Brier (`0.2228` vs `0.2305`) |
| E020 | 27 Nov 03:57 | Model 2: paraphrase-multilingual-MiniLM-L12-v2 + LR (T-023) | 0.575 | 0.566 | 0.343 | 0.684 | 0.556 | — | 5x3 grouped CV on text_norm embeddings, C=1.0 |
| E021 | 27 Nov 04:21 | Ensemble: `0.7·(0.4·LR+0.6·SVM) + 0.3·M2` + Task B thresholds + OOD calibration | 0.625 | 0.591 | 0.531 | 0.778 | — | 0.538 | Cal beats const (`0.2218` vs `0.2305`); OOF saved `E021_ens.npz` |

Real `experiments.csv` also holds three canonical-fold E001 runs (26 Nov 19:02/19:06/19:26 UTC; F_A 0.5867, 0.5867, 0.595) and the provisional-fold 0.524 row summarised above. The A2 error report uses the saved repeat-averaged OOF array, whose argmax macro-F1 is 0.510; state the source when quoting any of these numbers.

## 8. Kaggle submission log

| # | Date/time | File | What changed | Public score | Selected as final? |
|---|---|---|---|---|---|
| 1 | 27 Nov 02:32 | `outputs/predictions/submission_E1126_2032.csv` | Baseline (format check, E001) | *(pending upload)* | No |
| 2 | 27 Nov 03:04 | `outputs/predictions/submission_E1126_2104.csv` | Phase 2 E015 (`word+char`, `n_aug=5`, tuned Task B thresholds, `postprocess.py`) | *(pending upload)* | Candidate A |
| 3 | 27 Nov 03:29 | `outputs/predictions/submission_E1126_2129.csv` | Phase 3 E018 (`0.4·LR + 0.6·LinearSVC` blend, logistic calibration, Task B thresholds, `postprocess.py`) | *(pending upload)* | No |
| 4 | 27 Nov 04:21 | `outputs/predictions/submission_E1126_2221.csv` | Phase 3+4 E021 (`M1+M2` ensemble, OOD logistic calibration, Task B thresholds, `postprocess.py`) | *(pending upload)* | Candidate B |

## 9. Lessons and gotchas

- Run in each new PowerShell: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned` then `.\.venv\Scripts\Activate.ps1`.
- `check_noise.py` (in repo root) inspects frequent tokens in test text; use for noise statistics only (unsupervised).
- With ~86 rows, differences of one or two F1 points are noise; prefer simple models within noise.
- Don't let any rule or example be derived from specific test rows.
- F-06 (27 Nov 02:40 BST): test_folds.py 6/6 pass via unittest discover; cross_validation confirms all six classes in every fold; no relaxation needed. pytest not installed in .venv (C owns requirements.txt).
- A2 (27 Nov 03:09 BST, `outputs/validation_results/e001_m1_error_report.md`): top failure pattern 1 — class 5 (promotional spam) collapses into class 0 (10/12 → 0, recall 0.167, F1 0.222); promos share offer/dial-code/brand surface forms with legitimate SMS.
- A2 pattern 2 — script/structure shift: Bangla-script macro-F1 0.056, Banglish 0.290, conversations 0.267 vs English 0.488 / single 0.526; obfuscated and homoglyph rows (ids 10053, 10035, 10020, 10070) are missed.
- A2 pattern 3 — hard-legit keyword collisions (ids 10021 parcel, 10063 expiry, 10079 transfer, 10051 bKash) plus very conservative Task B thresholds (technique recall 0.182; T2 recall 0.091, T6 0.182). B: F-03/F-04 should target these, not just C tuning.
- A2 caveat: error report uses argmax of the saved repeat-averaged OOF array `outputs/oof/e001_m1.npz` (macro-F1 0.510), so its aggregate differs slightly from per-repeat means logged in §7/§8.
- S-01 (27 Nov 03:25 BST, after B pull; recounted 03:58 after A5 test): earlier `python -m unittest discover -s tests -v` → 18 tests passed, 2 pytest import errors. The 9 pytest-style tests in `test_augment.py` (5) and `test_postprocess.py` (4) were executed directly and all passed. `pytest -q` itself remains unavailable.
- PULL-03 (27 Nov 04:10 BST, after C/B pull): `python -m unittest discover -s tests -v` → 40 discovered, 39 passed, 1 import error (`test_predict_df.py` still imports pytest); C's rewritten `test_submission.py` now runs under unittest and all 21 validator tests pass. The 9 pytest-style tests still pass directly; `python -m pytest --version` still fails because the local venv was not reinstalled.
- S-07 partial (27 Nov 04:10 BST): `tools/validate_submission.py` passed both `outputs/predictions/submission_E1126_2104.csv` (Candidate A/E015) and `submission_E1126_2129.csv` (Candidate B/E018); public scores still pending.
- S-01 FINAL (27 Nov 04:20 BST): after user authorization, installed `pytest>=7.0` into the local ignored `.venv` (resolved pytest 9.2.0); `python -m pytest -q` → **60 passed in 15.30s**. No tracked file changed for the install because `.venv/` is gitignored (`.gitignore:2`); `requirements.txt:18` already declares `pytest>=7.0`.
- S-01: `python -m src.training.cross_validation` passes after the pytest install — 86 rows, 85 groups, 3 clusters, all six classes in every fold, group-leakage and LCO checks pass; `folds.csv` regenerated deterministically.
- S-06: `.env` exists at ZIP root, is 0 bytes, is ignored by `.gitignore` (`git check-ignore -v .env` → `.gitignore:9:.env`), `.env.example` exists, and README states the empty-`.env`/no-API setup.
- A5 (27 Nov 03:55 BST): added `data/extra/handwritten.py` with 84 original rows in 14 six-row groups (`group_id` 9001–9014, ids 90001–90084); label counts `{0:33, 1:21, 2:9, 3:6, 4:6, 5:9}`.
- A5 guard: validation passes; raw and cleaned-overlap checks against `sample.csv` and `test.csv` both return 0; no test row was read or copied. Minimal pairs stay in the same `group_id` so folds cannot split them.
- PULL-02 (27 Nov 04:05 BST): merged commits `f753981` (B Phase 3 E018: blend + calibration) and `109e8d9` (C notebooks, frozen `notebooks/src/`, `tools/build_notebook.py`, `tools/validate_submission.py`, `pytest>=7.0` in requirements). MEMORY conflict resolved manually; no A/B/C source file was overwritten.
- A3 cleanup (27 Nov 04:25 BST): `README.md` API notes now state that rules/examples come from labeled sample, task definitions, 84 hand-written rows and aggregate unsupervised test statistics; no test text/label is copied. Runtime placeholder remains pending C's clean-run.
- PULL-04 (27 Nov 04:35 BST): merged commits `1d8a1c3` (C T-023 Model 2 embeddings + E020 OOF/test NPZs) and `c97ceee`; both MEMORY conflicts resolved manually, A work preserved, no A/B/C source file overwritten.
- A3 doc sync (27 Nov 04:40 BST; runtime note corrected 05:00): `README.md` declares Model 2 and the `paraphrase-multilingual-MiniLM-L12-v2` encoder (Apache-2.0), removes the stale "no external model" claim, and states that Model 2 downloads declared weights and benefits from GPU for 500k rows. Actual wall time/peak memory still pending C's clean-run.
- S-01 post-PULL-04 (27 Nov 04:50 BST): `python -m pytest -q` → **62 passed in 235.66s (0:03:55)** with C's new `test_embedding_features.py`; `cross_validation` passes; E015/E018 submissions pass strict validator. The 3m55s is suite time, not the clean-run runtime needed for `README.md:42`.

## 10. Handoff notes (rewrite at the end of every session)

```
Last updated: 27 Nov post-PULL-05 by A/C (merged B commit eed9eff; no commit)
State:
  DONE (B, PULL-05) — T-030/T-031/T-040 complete: blended M1 (E015 LR + E016 LinearSVC) with C's E020 M2; OOF gains F_A 0.5825 -> 0.6253, F_B 0.578 -> 0.5908, F_C 0.505 -> 0.5308, Cal 0.7782 (Brier 0.2218 vs const 0.2305). E021 ensemble artifacts: `outputs/oof/E021_ens.npz` and `outputs/predictions/submission_E1126_2221.csv` (B reports [OK]).
  DONE (B) — Phase 1 E001, Phase 2 E015 tuning/postprocess, Phase 3 E018 blend + logistic calibration shipped in `src/prediction.py`.
  DONE (C, PULL-04) — Model 2 T-023: `embedding_features.py` + `train_embeddings.py` executed as E020; `outputs/oof/E020_m2.npz` and `outputs/predictions/E020_m2_test.npz` delivered; E020 F_A=0.5748, F_B=0.5663, F_C=0.3428, Cal=0.6835, S_hat=0.5558.
  DONE (C, local before PULL-05) — `tools/build_notebook.py` rebuilt from `src/` with A5 frozen writefile + §11 tick; 1,000-row clean smoke test passed in 36.93s. Must redo after `eed9eff`.
  DONE (A) — F-06 verified; A1 `eda_report.md`; A2 `error_analysis.py` + e001_m1 report/PNG/CSVs; A3 `README.md` draft + API-note cleanup; F-09 `folds.csv` cleanup + role mapping.
  DONE (A, S-01 FINAL) — after Model 2 pull, `python -m pytest -q` → 62 passed in 235.66s; `cross_validation` passes; E015/E018 submissions pass strict validator. E021 validator re-check pending after PULL-05.
  DONE (A, S-06) — `.env` present, empty, git-ignored; `.env.example` present; README states no-API setup.
  DONE (A, A5) — 84 hand-written hard-legit/minimal-pair rows in `data/extra/handwritten.py`; declared in §11; zero raw/clean overlap with sample/test.
  RESOLVED — A's `build_views()` is in `src/preprocessing.py`; B's older note about `text_norm`/`text_skel`/`first_msg` fallback is stale.
  REVISED — embeddings/Model 2 are delivered and now blended in E021. LLM-generated data and pseudo-labeling remain cut; hand-written extras were revived by user request.
  HANDOFF TO B — A5 remains optional: load `data/extra/handwritten.py::load_handwritten()` (84 rows, weight 0.75 default), keep `group_id`, adopt only if grouped-CV F_A/F_rob/F_dom do not drop.
  HANDOFF TO B — F-05: 2 noise-only rows, both 0/NONE/0 (n=2; keep p=0.60 as assumption).
  HANDOFF TO B — F-04: 8 rules at >=95% (`eda_report.md` §6); B's 4 postprocess rules are a subset — check the remaining 4 on grouped CV.
  HANDOFF TO B — A2: class-5 collapse, script/conversation weakness, hard-legit collisions + low T2/T6 recall.
  HANDOFF TO C/A — README declarations are synced (Model 2 + hand-written extras); clean-run wall time/peak memory still needed.
Next 3 steps:
  1. C/A: rebuild notebook after `eed9eff`, run full clean run, and record wall time/peak memory in `README.md` §Runtime and §9.
  2. B: upload Candidate A (`submission_E1126_2104.csv`) and Candidate B (`submission_E1126_2221.csv`), log public scores in §8, and select both finals.
  3. C/A: implement/verify `tools/make_zip.py` (T-054), byte-match ZIP contents, verify compliance, and sign S-07 items 1/4/5.
Blockers / questions for teammate:
  - Resolved: local `.venv` has pytest 9.2.0; `pytest -q` passes 62/62 after the Model 2 test file; C's clean environment must install `requirements.txt`.
  - Resolved: README now declares Model 2/encoder and hand-written extras; only clean-run runtime is still pending.
  - A5 grouped-CV decision still pending; do not enable `load_handwritten()` in final training unless a grouped-CV check is positive.
  - §8 still needs public scores for final candidates; §7 already has E018/E020/E021 CV rows.

Where results live: README.md ; data/extra/handwritten.py ; notebooks/ ; tools/ ; outputs/validation_results/eda_report.md ; outputs/validation_results/e001_m1_error_report.md ; outputs/validation_results/e001_m1_confusion.png ; outputs/validation_results/experiments.csv ; outputs/validation_results/folds.csv ; outputs/oof/E015_m1.npz ; outputs/oof/E020_m2.npz ; outputs/oof/E021_ens.npz ; outputs/predictions/E020_m2_test.npz ; outputs/predictions/
```

## 11. Declaration checklist (feeds the notebook's first cells)

| Resource | Type | Source / provider | Licence | Purpose | Declared in notebook? |
|---|---|---|---|---|---|
| scikit-learn, pandas, numpy, scipy | libraries | PyPI | BSD | modelling/data | [ ] |
| sentence-transformers + sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | pretrained model | Hugging Face | Apache-2.0 | multilingual dense sentence embeddings for Model 2 and OOD similarity | [x] |
| LLM API (if used) | hosted API | `<provider/model>` | provider terms | synthetic training data | [ ] |
| Hand-written extra examples (84 rows / 14 groups) | `data/extra/handwritten.py` | our own, authored by A from PRD + labeled sample patterns only; zero overlap with test | n/a | optional training augmentation, pending B's grouped-CV check | [x] |
| Pseudo-labeling on `test.csv` text | method | our code | n/a | semi-supervised (if adopted) | [ ] |

## 12. Glossary

- **MFS:** mobile financial service (bKash, Nagad, Rocket, etc.). **Banglish:** Bangla written in Latin letters.
- **OOF:** out-of-fold predictions made on rows the model did not train on.
- **Group / template cluster:** near-duplicate messages kept together in folds.
- **Minimal pair:** a legitimate and a scam message sharing most wording.
- **S_hat:** our surrogate composite score computed from CV and slice proxies.
