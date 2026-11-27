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
| D-010 | 27 Nov 05:20 | Zero-shot LLM labeling pipeline in `src/llm_label.py` with SHA-256 caching (`data/extra/llm_labels_v1.jsonl`), calibration gate (F1 >= 0.80), and rule filtering | Augment scarce training data with model predictions from unlabeled test templates per rulebook §5 compliance rules |
| D-011 | 27 Nov 05:26 | Keep the `text_norm`/`text_skel` fix; promote E023 (SVM) but not E022 (LR) alone; downstream re-tuning handled by Step 5 | E023 improves clean F_A +0.0118, corrupted F_A +0.0058, LCO +0.0062 with no slice loss over 3 pts; E022 improves F_A +0.0235 but drops Bangla-script F1 by 0.0556 on ~6 rows. Full table in §7. |

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
| E022 | 27 Nov 05:23 | Skeleton-view fix: `text_norm=preserved`, `text_skel=skel`, LR C=5 | 0.596 | 0.447 | 0.478 | 0.778 | 0.509 | 0.550 | Saved folds; LCO 0.3092; Bangla-script slice 0.0185 vs 0.0741 old; LR not promoted alone |
| E023 | 27 Nov 05:23 | Skeleton-view fix: `text_norm=preserved`, `text_skel=skel`, LinearSVC C=1 | 0.595 | 0.463 | 0.489 | 0.680 | 0.505 | 0.541 | Saved folds; LCO 0.3036; no slice drop over 3 pts; promoted for Step 5 blend re-tune |
| E024 | 27 Nov 05:35 | Step 4: Real + LLM-augmented rows (weight=0.3, cap=400, near_real leak filter) | 0.756 | 0.691 | 0.578 | 0.790 | 0.692 | 0.711 | Promoted: +0.182 on F_A, F_dom=0.585 (+0.269), hard-legit slice 0.967 |
| E025 | 27 Nov 06:00 | T-032: Real + LLM + Pseudo-labels (360 rows, 60/class, tau>=0.82, weight=0.25) | 0.861 | 0.809 | 0.823 | 0.841 | 0.834 | 0.819 | Massive promotion: F_A=0.861 (+0.105), F_dom=0.811 (+0.225), S_hat=0.8341 (+0.143) |
| E028 | 27 Nov 11:45 BST | Step 5 re-tune: E026 LR + E027 SVM + E020 M2 + E025 extras; equal M1/M2 weights, cross-fitted Task B thresholds, constant-confidence fallback | 0.857 | 0.710 | 0.848 | 0.872 | — | — | Nested F_B (in-sample 0.821); w_lr=0.30, w_m1_a/b/c=0.50; thresholds `[0.52,0.5,0.48,0.38,0.56,0.38]`; Brier 0.1283 = constant baseline; `submission_E1127_0545.csv` |

Real `experiments.csv` also holds three canonical-fold E001 runs (26 Nov 19:02/19:06/19:26 UTC; F_A 0.5867, 0.5867, 0.595) and the provisional-fold 0.524 row summarised above. The A2 error report uses the saved repeat-averaged OOF array, whose argmax macro-F1 is 0.510; state the source when quoting any of these numbers.

Step 1 fair before/after (same saved folds, per-repeat means; old runs re-executed in memory with pre-fix views, new OOF saved as E022/E023):

| Run | F_A | F_B | F_C | Corrupted F_A | lang_english | lang_banglish | lang_bangla_script | struct_single | struct_conversation | slice_hard_legit | LCO F_A |
|---|---|---|---|---|---|---|---|---|---|---|---|
| E015-old (LR) | 0.5728 | 0.4283 | 0.4796 | 0.5379 | 0.5334 | 0.5031 | 0.0741 | 0.6015 | 0.2253 | 0.6122 | 0.3156 |
| E022-new (LR) | 0.5963 | 0.4466 | 0.4779 | 0.5500 | 0.5570 | 0.5435 | 0.0185 | 0.6120 | 0.3488 | 0.9778 | 0.3092 |
| E016-old (SVM) | 0.5834 | 0.4493 | 0.4800 | 0.5355 | 0.5397 | 0.5817 | 0.0278 | 0.6130 | 0.2427 | 0.7886 | 0.2974 |
| E023-new (SVM) | 0.5952 | 0.4631 | 0.4892 | 0.5413 | 0.5492 | 0.6595 | 0.0000 | 0.6210 | 0.3009 | 0.9778 | 0.3036 |

Gate: E022 has F_A +0.0235, corrupted F_A +0.0121, LCO −0.0064 but Bangla-script −0.0556 on ~6 rows, so LR alone fails the 3-point slice rule. E023 has F_A +0.0118, corrupted F_A +0.0058, LCO +0.0062 and worst slice −0.0278, so the fix is kept and Step 5 must re-tune the LR/SVM/M1 blend.

## 8. Kaggle submission log

| # | Date/time | File | What changed | Public score | Selected as final? |
|---|---|---|---|---|---|
| 1 | 27 Nov 02:32 | `outputs/predictions/submission_E1126_2032.csv` | Baseline (format check, E001) | *(pending upload)* | No |
| 2 | 27 Nov 03:04 | `outputs/predictions/submission_E1126_2104.csv` | Phase 2 E015 (`word+char`, `n_aug=5`, tuned Task B thresholds, `postprocess.py`) | *(pending upload)* | Candidate A |
| 3 | 27 Nov 03:29 | `outputs/predictions/submission_E1126_2129.csv` | Phase 3 E018 (`0.4·LR + 0.6·LinearSVC` blend, logistic calibration, Task B thresholds, `postprocess.py`) | *(pending upload)* | No |
| 4 | 27 Nov 04:21 | `outputs/predictions/submission_E1126_2221.csv` | Phase 3+4 E021 (`M1+M2` ensemble, OOD logistic calibration, Task B thresholds, `postprocess.py`) | *(pending upload)* | Candidate B |
| 5 | 27 Nov 06:20 | `outputs/predictions/submission_E1127_0020.csv` | Step 5 Candidate: Real + LLM + Pseudo-labels (E025), cross-fitted thresholds, nested F_B=0.845, unskewed balanced mix | *(pending upload)* | Superseded by #6 |
| 6 | 27 Nov 11:45 BST | `outputs/predictions/submission_E1127_0545.csv` | Step 5 rerun with full-test M2: E026/E027/E020/E028, equal M1/M2 weights (0.50/0.50), nested F_B=0.7103, constant-confidence Brier=0.1283, validator OK, H1 no longer supported | *(pending upload)* | Candidate C (Recommended Primary) |

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
- STEP2 (27 Nov 05:40 BST): `tools/diagnose_test.py` + `tests/test_diagnose.py` added; `python -m pytest -q` → **69 passed in 30.67s**; report `outputs/validation_results/test_diagnosis.md` generated from aggregate statistics only (no test ids/texts). H1 supported: 499,000/500,000 (99.80%) M2 rows equal the padded tail vector and ood_sim tail is constant; H2 not supported; H3 cannot tell; H4 shift supported for the shift itself with effect on errors unknown; H5 not supported. M1/M2 disagreement unavailable because M1 test probabilities are not saved.
- STEP2 gotchas: `tools/diagnose_test.py` bootstraps repo root into `sys.path` so `python tools/diagnose_test.py` works; submission ids are strings while `test.csv` and NPZ ids are numeric, so comparisons cast to `str`; `load_or_build_preprocessed_test()` cache omits `has_url`, so the diagnosis recomputes it from `URL_RE` on `preserved`; cache `outputs/cache/test_prep_9c32897ed3.pkl` built once in 125.5s.
- STEP2-FULL (27 Nov 05:44 BST): local CPU full-test M2 job completed (`python -m src.training.train_embeddings --exp-id E020 --full-test`, PID 10056; log `outputs/logs/full_m2_20261127_0728.err.log`). `outputs/predictions/E020_m2_test.npz` now has 500,000 rows; the padded-tail count is 1 (the last row itself, not 499,000), with 497,411 distinct `A` vectors and 473,785 distinct `ood_sim` values. Backup at `outputs_backup/pre_fulltest_20261127_0728/`. The run recomputed `E020_m2.npz` (F_A 0.5748 unchanged) and appended a second E020 row to `experiments.csv`. Kaggle GPU/CLI was unavailable, so CPU (~3h14m) was used. No submission regenerated yet; Step 5 blend retune is still required.

## 10. Handoff notes (rewrite at the end of every session)

```
Last updated: 27 Nov 2026 by A (Step 5 rerun + Step 6 generated notebook/smoke test complete; integrated with B's Steps 3-5 + local Step 1/2)
State:
  DONE (B/A, Steps 3-4 / T-032 / E025) — `src/llm_label.py` zero-shot labels + SHA-256 cache, `src/training/train_llm_aug.py` leak-free augmented training (E024), and `src/pseudo_label.py::mine_pseudo_labels` (tau >= 0.82, max_per_class=60, postprocess consistency, weight=0.20) generate 360 balanced rows in `data/extra/pseudo_labels_v1.jsonl`. E025 5x3 grouped CV with 68 near-real rows isolated: F_A=0.8610, F_B=0.8090, F_C=0.8230, Cal=0.8410, F_robust=0.8190, F_domain=0.8106, S_hat=0.8341; Banglish=92.24%, Bangla script=100.0%, hard-legit=96.67%. Decisive promotion to primary training configuration.
  DONE (B/A, Step 5) — `src/prediction.py` re-tuned on E026 LR + E027 SVM + E020 full-test M2 + E025 extras, wrote E028 ensemble OOF and `outputs/predictions/submission_E1127_0545.csv` (§8 #6, Candidate C, recommended primary). Backup `outputs_backup/pre_step5_rerun_20261127_1131/`; log `outputs/logs/step5_rerun_20261127_1131.log`; command `python -m src.prediction`; seed 42; 5 folds x 3 repeats. w_lr=0.30 (F_A=0.9103); M1/M2 equal weights 0.50/0.50 for A/B/C (F_A=0.8574, F_C=0.8477); thresholds `[0.52,0.5,0.48,0.38,0.56,0.38]`; nested cross-fitted F_B=0.7103 (in-sample 0.821); calibrator fell back to constant with Brier 0.1283 (Cal 0.8717). Strict validator OK; label mix {0:.368,1:.169,2:.096,3:.083,4:.129,5:.155} vs E021 {0:.735,...}; H1 no longer supported.
  DONE (Tests) — `tests/test_pseudo_label.py`, `tests/test_tfidf_views.py`, `tests/test_diagnose.py`, Step 5 equal-weight tie-break test, and `tests/test_build_notebook.py` added. Final `pytest -q` → 87 passed in 34.20s; `submission_E1127_0545.csv` passed strict validation.
  DONE (Notebook source) — `src/pseudo_label.py` added to `tools/build_notebook.py`; final Step 6 rebuild is complete below.
  DONE (Step 6) — `tools/build_notebook.py` now validates embedded modules/stubs, embeds `llm_labels_v1.jsonl`, `pseudo_labels_v1.jsonl` and `folds.csv` as `%%writefile` cells, installs/verifies required libraries, recursively finds nested `/kaggle/input` data dirs, generates E020 Model 2 inside the notebook, calls `prediction.main(test_limit=...)`, performs strict sliced validation, reports optional `SCAMSHIELD_COMPARE_SUBMISSION` comparison, and sets the nbconvert timeout to 3600 s. Generated `notebooks/notebook.ipynb` has 35 cells and is never hand-edited.
  DONE (Step 6 smoke) — isolated `jupyter nbconvert --execute` with `SCAMSHIELD_TEST_SLICE=1000` passed: embedded `folds.csv` was loaded, E020 M2 was generated inside the notebook, strict validation passed, wall time 1057.30 s, peak traced RAM 1146.3 MB. Full 500,000-row clean run is still pending; expected CPU ≈3 h, Kaggle GPU ≈5.5–8.5 min dominated by embeddings.
  DONE (Step 6 tests) — new `tests/test_build_notebook.py` rejects `NotImplementedError` stubs and missing embedded imports and checks `find_nested_data_dir()`. Focused run: 3 passed; full `pytest -q` → 87 passed in 34.20s.
  DONE (B, PULL-05) — T-030/T-031/T-040 complete: blended M1 (E015 LR + E016 LinearSVC) with C's E020 M2; OOF gains F_A 0.5825 -> 0.6253, F_B 0.578 -> 0.5908, F_C 0.505 -> 0.5308, Cal 0.7782 (Brier 0.2218 vs const 0.2305). E021 ensemble artifacts: `outputs/oof/E021_ens.npz` and `outputs/predictions/submission_E1126_2221.csv`.
  DONE (A, Step 1) — `build_text_dict` maps `text_norm` to `preprocess()["preserved"]` and `text_skel` to `preprocess()["skel"]`; E022/E023 OOF saved on saved folds. E022 LR fails the 3-point slice gate, E023 SVM passes; keep the fix, with downstream re-tuning now handled by Step 5 (D-011).
  DONE (A, Step 2) — aggregate-only diagnosis tool/report delivered (`tools/diagnose_test.py`, `tests/test_diagnose.py`, `outputs/validation_results/test_diagnosis.md`). Re-run on the Step 5 candidate produced `outputs/validation_results/test_diagnosis_step5.md`: 0.00% padded M2 rows, H1 not supported, class mix close to the labeled sample. No individual test texts or ids were logged or written.
  DONE (A, Step 2 full-test fix) — full-test M2 embeddings generated locally on CPU at 05:44 BST; `outputs/predictions/E020_m2_test.npz` covers all 500,000 rows with no 499k padded tail (1 self-match, 497,411 distinct A vectors, 473,785 distinct ood_sim). E020 OOF recomputed and a second E020 row logged. Kaggle GPU/CLI was unavailable.
  DONE (B) — Phase 1 E001, Phase 2 E015 tuning/postprocess, Phase 3 E018 blend + logistic calibration shipped in `src/prediction.py`.
  DONE (C, PULL-04) — Model 2 T-023: `embedding_features.py` + `train_embeddings.py` executed as E020; E020 F_A=0.5748, F_B=0.5663, F_C=0.3428, Cal=0.6835, S_hat=0.5558.
  DONE (C, local before PULL-05) — `tools/build_notebook.py` rebuilt from `src/` with A5 frozen writefile + §11 tick; 1,000-row clean smoke test passed in 36.93s. Superseded by the final Step 6 rebuild/smoke above.
  DONE (A) — F-06 verified; A1 `eda_report.md`; A2 `error_analysis.py` + e001_m1 report/PNG/CSVs; A3 `README.md` draft + API-note cleanup; F-09 `folds.csv` cleanup + role mapping.
  DONE (A, S-01 FINAL) — after the Model 2 pull, `python -m pytest -q` → 62 passed in 235.66s; post-integration 83 passed in 32.51s; after Step 5 changes 84 passed in 33.91s; after Step 6 tests 87 passed in 34.20s. E015/E018 and Step 5 candidates pass the strict validator.
  DONE (A, S-06) — `.env` present, empty, git-ignored; `.env.example` present; README states no-API setup.
  DONE (A, A5) — 84 hand-written hard-legit/minimal-pair rows in `data/extra/handwritten.py`; declared in §11; zero raw/clean overlap with sample/test.
  RESOLVED — A's `build_views()` is in `src/preprocessing.py`; B's older note about `text_norm`/`text_skel`/`first_msg` fallback is stale.
  REVISED — LLM labeling, leak-free LLM-augmented training and pseudo-labeling are now implemented/promoted (E024/E025); hand-written extras remain optional.
  HANDOFF TO B — A5 remains optional: load `data/extra/handwritten.py::load_handwritten()` (84 rows, weight 0.75 default), keep `group_id`, adopt only if grouped-CV F_A/F_rob/F_dom do not drop.
  HANDOFF TO B — F-05: 2 noise-only rows, both 0/NONE/0 (n=2; keep p=0.60 as assumption).
  HANDOFF TO B — F-04: 8 rules at >=95% (`eda_report.md` §6); B's 4 postprocess rules are a subset — check the remaining 4 on grouped CV.
  HANDOFF TO B — A2: class-5 collapse, script/conversation weakness, hard-legit collisions + low T2/T6 recall.
  HANDOFF TO C/A — README declarations synced (Model 2 + hand-written extras + frozen LLM labels + pseudo-labeling); 1k-row smoke wall time/peak memory recorded in README §Approximate runtime and §10; full 500k clean-run timing still needed.
Next 3 steps:
  1. B/A: upload Candidate C (`submission_E1127_0545.csv`) plus a second candidate, log public scores in §8, and select both finals; do not hand-edit CSVs.
  2. C/A: implement/verify `tools/make_zip.py` (T-054), validate ZIP/SHA-256, and sign S-07 items 1/4/5; notebook rebuild and 1k smoke are done, full clean run still pending.
  3. A/B: decide on optional A5 hand-written extras and the remaining 4 postprocess rules only if grouped-CV is positive; otherwise freeze the current pipeline.
Blockers / questions for teammate:
  - Resolved: local `.venv` has pytest 9.2.0; post-Step-5 `pytest -q` → 84 passed; C's clean environment must install `requirements.txt`.
  - Resolved: README now declares Model 2/encoder, hand-written extras, frozen LLM labels and pseudo-labeling; 1k smoke runtime/peak memory recorded, full clean-run runtime still pending.
  - A5 grouped-CV decision still pending; do not enable `load_handwritten()` in final training unless a grouped-CV check is positive.
  - §8 #6 still needs a public score; §7 now includes E028 Step 5 re-tune numbers.
  - Resolved: full-test M2 embeddings were used in Step 5; `test_diagnosis_step5.md` reports H1 not supported.

Where results live: README.md ; data/extra/handwritten.py ; data/extra/llm_labels_v1.jsonl ; data/extra/pseudo_labels_v1.jsonl ; notebooks/ ; notebooks/notebook.ipynb ; tools/ ; tools/build_notebook.py ; tools/diagnose_test.py ; tests/test_build_notebook.py ; tests/test_diagnose.py ; outputs_backup/pre_step5_rerun_20261127_1131/ ; outputs/cache/test_prep_9c32897ed3.pkl ; outputs/logs/step5_rerun_20261127_1131.log ; outputs/validation_results/eda_report.md ; outputs/validation_results/e001_m1_error_report.md ; outputs/validation_results/e001_m1_confusion.png ; outputs/validation_results/experiments.csv ; outputs/validation_results/folds.csv ; outputs/validation_results/test_diagnosis.md ; outputs/validation_results/test_diagnosis_step5.md ; outputs/oof/E015_m1.npz ; outputs/oof/E020_m2.npz ; outputs/oof/E021_ens.npz ; outputs/oof/E022_m1.npz ; outputs/oof/E023_m1.npz ; outputs/oof/E024_m1.npz ; outputs/oof/E025_m1.npz ; outputs/oof/E028_ens.npz ; outputs/predictions/E020_m2_test.npz ; outputs/predictions/submission_E1127_0545.csv ; outputs/predictions/
```

## 11. Declaration checklist (feeds the notebook's first cells)

| Resource | Type | Source / provider | Licence | Purpose | Declared in notebook? |
|---|---|---|---|---|---|
| scikit-learn, pandas, numpy, scipy, ftfy, regex, unidecode, torch, python-dotenv | libraries | PyPI | BSD/MIT | modelling, preprocessing, data handling, clean-run install | [x] |
| sentence-transformers + sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | pretrained model | Hugging Face | Apache-2.0 | Model 2 OOF + full-test embeddings and OOD similarity | [x] |
| LLM API (if used) | hosted API | Configurable via `.env` (`LLM_PROVIDER`, `LLM_MODEL`, `LLM_API_KEY`) | Provider terms | frozen `llm_labels_v1.jsonl` is embedded; `REGENERATE_SYNTH=False` needs no key; key only for explicit regeneration | [x] |
| Hand-written extra examples (84 rows / 14 groups) | `data/extra/handwritten.py` | our own, authored by A from PRD + labeled sample patterns only; zero overlap with test | n/a | declared only; not used in final E028/Step 5 pipeline | [x] |
| Pseudo-labeling on `test.csv` text | method | our code | n/a | semi-supervised (adopted in E025/Step 5; frozen file embedded) | [x] |

## 12. Glossary

- **MFS:** mobile financial service (bKash, Nagad, Rocket, etc.). **Banglish:** Bangla written in Latin letters.
- **OOF:** out-of-fold predictions made on rows the model did not train on.
- **Group / template cluster:** near-duplicate messages kept together in folds.
- **Minimal pair:** a legitimate and a scam message sharing most wording.
- **S_hat:** our surrogate composite score computed from CV and slice proxies.
