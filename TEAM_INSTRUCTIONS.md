# TEAM_INSTRUCTIONS: ScamShield BD (v2, status-aware)

**Version 2.** Replaces the v1 plan. v1 assumed a build from zero at Thu 23:15. The repo has moved on: the data, preprocessing, validation, augmentation, calibration and baseline pipeline exist. This version lists what is done, what was changed, and what is left, ordered by risk against the deadlines.

How to use this file: start Claude Code in the repo root and say "I am Teammate A" (or B or C). Claude Code must read this file, do Part 0 and Part 1, then execute only that teammate's section in Part 3, plus the shared tasks in Part 4. If only two people are active, B takes C's P0 items 1 and 2 (validator, notebook builder) and A takes the README and `.env`.

Source docs in `docs/` (read order): `MEMORY.md`, `RULES.md`, `ARCHITECTURE.md`, `DESIGN.md`, `PRD.md`, `TASKS.md`. If this file and those docs disagree, the docs win, except for task ownership and the status/cut decisions below, where this file wins. Log conflicts in `MEMORY.md` section 3.

---

## Part 0. Operating rules for Claude Code

1. **Identify the teammate.** Edit only files that teammate owns (Part 2). For a change in someone else's file, write the request in `MEMORY.md` under "Cross-team requests" and continue with a stub.
2. **Read before you write.** At session start read `MEMORY.md` sections 4, 5, 10. At session end rewrite section 10 (handoff) and append to the experiment or submission log.
3. **Frozen interfaces.** Signatures in `ARCHITECTURE.md` section 4 and the formats in Part 1.3 change only with a `MEMORY.md` section 4 entry (**next ID: D-009**) and a note to the others.
4. **Engineering standards.** Python 3.x, type hints on public functions, one-line docstrings. No printing inside `preprocessing` or `augment` (use `logging`). No global mutable state; pass `rng` or `seed`; all randomness derives from `config.SEED` (42). No silent fallbacks: log a warning. Public functions get a test. Run `pytest -q` before every commit. Commits `<area>: <what>`, small and frequent, pull before push, `main` must always run. Never hand-edit `notebooks/notebook.ipynb` (generated).
5. **Compliance (disqualification risk).** Never copy test text or labels into training, rules, examples or chat prompts. Never hand-edit predictions. Never hardcode ids or strings from test rows. Never print or commit keys. Look at test text only for unsupervised statistics. Every external model, dataset or API goes into `MEMORY.md` section 11 the moment it is used.
6. **Validation discipline.** Validate on real labeled rows only. Augmented and synthetic rows inherit the source `group_id` and enter training folds only. Compare experiments on the same saved folds. Never tune on the public leaderboard.
7. **Definition of done for any task:** code merged, tests pass, acceptance criteria met, `MEMORY.md` updated, the next person's input available at the agreed path.
8. **When blocked** more than 15 minutes: write it in `MEMORY.md` section 10, switch to the next unblocked task, tell the team.
9. **Verify before you trust.** The status in Part 1.1 was read from the files, not executed. First action of every teammate: `pytest -q` and `python -m src.training.cross_validation`. Record real results in `MEMORY.md`.

---

## Part 1. Shared foundation

### 1.1 Status snapshot (from file review)

| Area | State | Notes |
|---|---|---|
| `config.py` | Done, **incomplete** | Fixed `ROOT/data` path, creates folders at import, no `IS_KAGGLE`, `find_data_dir`, `set_seed`, `MAX_LEN`, `REGENERATE_SYNTH`. Will likely break on Kaggle/Colab |
| `data_loader.py` | Done | Matches spec |
| `preprocessing.py`, `test_preprocessing.py` | Done | 22-column contract, noise handling, ZWJ rule. Skeleton drops noise tokens instead of `<NOISE>` (accepted deviation) |
| `cross_validation.py`, `test_folds.py` | Done | Saves **`folds.csv`** (D-008), 5x3 grouped, leave-cluster-out, OOF npz. Test asserts all classes in every validation fold: may fail on rare classes, verify |
| `augment.py` | Done | Sets A and B. `test_augment.py` missing |
| `calibration.py` | Done, partial | Logistic on max prob, margin, entropy, noise ratio. No OOD or model-disagreement features (fine for now) |
| `train_baseline.py` | Done | One `Baseline` class holds Tasks A/B/C, CV, calibration, test prediction, 5-column output. Also writes a **2-column `submission.csv`** (risk, see F-02) |
| `text_analysis.py`, `check_noise.py` | Done | EDA helpers only. No `eda_report.md` |
| `metrics.py`, `error_analysis.py`, `postprocess.py`, `ensemble.py`, `submission.py`, `prediction.py` | **Missing** | No `predict_df` (FR-14) |
| `tools/validate_submission.py`, `build_notebook.py`, `make_zip.py` | **Missing** | P0 |
| `README.md`, `.env`, `.env.example`, `data/extra/handwritten.py` | **Missing** | README and `.env` are ZIP items |
| Embeddings, synthetic LLM data, pseudo-labels, transformer | Missing | **Cut** (see 1.4) |
| `MEMORY.md` sections 5, 7, 8, 10, 11 | Empty | Must be filled before packaging |

### 1.2 Fix list (found in review)

| ID | Fix | Owner | Priority |
|---|---|---|---|
| F-01 | `config.py`: add `IS_KAGGLE`, `find_data_dir()` (search for `sample.csv` locally and under `/kaggle/input`), `OUT_DIR` switch to `/kaggle/working`, `MAX_LEN`, `REGENERATE_SYNTH=False`, `set_seed()`. Make folder creation lazy (function, not import side effect) | B | P0 |
| F-02 | Stop writing the 2-column `outputs/predictions/submission.csv`. Write only the 5-column file named by experiment id. Never put a 2-column file in the ZIP | B | P0 |
| F-03 | Per-technique thresholds for Task B (currently fixed 0.50). Choose on OOF `proba_b`, shrink toward 0.5 when a technique has fewer than 5 positives, cross-fit if time allows. Adopt only if grouped-CV `F_B` improves | B | P1 |
| F-04 | Consistency rules from the real cross-tab (A reports, B implements). Only rules holding in 95% or more of labeled rows | A then B | P1 |
| F-05 | Noise-only rows currently forced to class 0 with p=0.60 on an unverified hypothesis. A checks `sample.csv` noise-only rows' real label/techniques/action and reports; B aligns the rule | A, B | P0 |
| F-06 | Verify `test_folds.py` passes. If the all-classes-in-every-fold assertion fails, relax it to "log which classes are missing" and note it in `MEMORY.md` | A | P0 |
| F-07 | TF-IDF is fit on **train only** (not train+test as DESIGN says). Keep as is (safe, simpler). Declare it in the notebook | B | P0 (doc only) |
| F-08 | Align experiment log columns: add `git_hash`, `config_json`, `S_hat` (or log "not computed"). Minimal `metrics.py` with `s_hat()` | B | P1 |
| F-09 | Doc cleanup: `folds.json` mentions in TASKS, MEMORY section 10, TEAM text become `folds.csv`. Role names A/B/C map to Teammate/Nabila as the team decides; record the mapping in `MEMORY.md` | A | P1 |

### 1.3 Frozen data contracts (updated)

**Labeled frame** (`load_sample()`): `id, text, label, techniques, action`.

**Views frame** (`build_views(df)`): input columns plus the 22 contract columns in `VIEW_CONTRACT_COLS`, plus the helper columns `clean, preserved, skel, noise_only, template_key, n_noise_tokens`. Never drops rows, never edits `id`.

**Folds file** (D-008): `outputs/validation_results/folds.csv` with columns `id, group, cluster, fold_r0, fold_r1, fold_r2`. `get_folds(df)` returns a list of `(train_idx, val_idx)` ordered repeat then fold.

**OOF file** `outputs/oof/<exp_id>_<model>.npz`: arrays `ids, y_a, y_b, y_c, noise_only, noise_ratio, repeat, A (n,6), B (n,6), C (n,6)` plus backward-compatible aliases `clean_proba_*`, and `corrupted_proba_*` when computed. Rows follow `sample.csv` order, averaged over repeats.

**Label maps:** classes 0 to 5, actions 0 to 5, techniques `T1` to `T6` serialised as sorted `T1|T2`, or `NONE` when empty (in `config.py` and `data_loader.py`).

**Submission** (5 columns, exact order of `sample_submission.csv`): `id,label,techniques,action,confidence`. `label` and `action` are integers, `techniques` is never empty, `confidence` is a float in [0.03, 0.97]. **No 2-column variant is ever produced** (F-02). `sample_submission.csv` decides the format (D1), but confirm in the Kaggle Data tab.

### 1.4 Cut decisions (effective now)

Dropped for this round: embeddings (Model 2), LLM-generated data, pseudo-labeling, fine-tuned transformer, ensemble, OOD calibration features, hand-written extras (only if time remains after P0). Reason: the baseline is the only validated model, packaging is incomplete, and packaging is never cut. The shipped system is therefore:

> TF-IDF (char skeleton + word + raw-char + first/last message + cues) + class-balanced logistic regression for Tasks A, B, C + Set A corruption augmentation + noise-only rule + logistic confidence calibrator.

The notebook declaration must say plainly that no external pretrained model, dataset or API is used. In that case `.env` is an empty file (still present, still in the ZIP). If an API is added later it goes into section 11 first.

### 1.5 Deadline map (BST, Fri 27 Nov)

| Time | Event | Rule |
|---|---|---|
| 17:30 | Code freeze | Only validated bug fixes after this |
| 18:00 | Prelims end | **Both finals selected manually on Kaggle before this** (otherwise Kaggle picks the top two by public score) |
| 21:30 | ZIP target | Upload, with 30-minute buffer |
| 22:00 | **ZIP deadline** | Late or missing items can disqualify |

If you are reading this past 17:30, skip straight to Part 4 (finals, validator, notebook, ZIP). Skip every P1 item.

---

## Part 2. File ownership

| Owner | Files |
|---|---|
| A | `src/data_loader.py`, `src/preprocessing.py`, `src/text_analysis.py`, `src/training/cross_validation.py`, `src/evaluation/error_analysis.py`, `data/extra/handwritten.py`, `README.md`, `outputs/validation_results/eda_report.md`, `tests/test_preprocessing.py`, `tests/test_folds.py` |
| B | `src/config.py`, `src/augment.py`, `src/training/train_baseline.py`, `src/models/*`, `src/evaluation/metrics.py`, `src/evaluation/calibration.py`, `src/postprocess.py`, `src/submission.py`, `src/prediction.py`, `tests/test_augment.py`, `tests/test_predict_df.py` |
| C | `tools/validate_submission.py`, `tools/build_notebook.py`, `tools/make_zip.py`, `tests/test_submission.py`, `.env.example`, `requirements.txt`, `SETUP.md` |
| Shared | `MEMORY.md`, `docs/*`, `.gitignore`, `outputs/*` (each person writes only their own outputs) |

Embedding, synthetic and pseudo-label files from v1 are removed from the active plan (Part 1.4).

---

## Part 3. Teammate instructions

### TEAMMATE A: Data, quality, documentation

**Order:** F-06, EDA report (A1), F-05 check, error report (A2), README (A3), F-09.

**A1. EDA report (P0, 45 min).** Write `outputs/validation_results/eda_report.md` and copy answers into `MEMORY.md` section 5. Reuse `python -m src.text_analysis` output and `cross_validation` main output. Answer with numbers:
1. Labeled row count and class counts (confirm or correct "about 86").
2. Language mix (English, Bangla script, Banglish, mixed).
3. Single vs conversation counts, `n_msgs` distribution.
4. Speaker marker format with the regex that matches it.
5. Duplicate and template clusters: sizes, group count (printed by `cross_validation` main).
6. Cross-tab `label x techniques x action`; list every rule at 95% or more with counts (B needs this for F-04).
7. **Noise-only rows: label, techniques, action** (F-05). If none exist in the sample, write "unknown".
8. Class 5 techniques and action.
9. Test statistics, unsupervised only: noise ratio histogram, language mix, share with lookalikes or zero-width characters. Never copy individual test rows.
10. Hard cases: legit rows containing OTP, link, cashback, parcel, urgent.

Acceptance: report exists, `MEMORY.md` section 5 filled, section 6 assumptions ticked or corrected.

**A2. Error report (P1, 45 min).** Minimal `error_analysis.py`: `report(y_true, y_pred, probs, df, name)` writing a confusion matrix PNG, per-class precision/recall/F1, slice table (use `compute_slice_metrics` from `cross_validation`), lists of legit-predicted-as-scam and the reverse (labeled rows only), per-technique precision/recall. Write the top three failure patterns into `MEMORY.md`. This feeds the notebook's "generalisation analysis" which reviewers grade.

**A3. README.md (P0, 30 min).** Required contents (RULES A4): team name, approach summary, external resources (state "none" if none) with licences for libraries, seeds (`SEED=42`), run order, approximate runtime (get the number from C's clean-run), API notes ("no API calls" if true). Pull facts from `MEMORY.md` section 11, not memory.

**A4. Doc cleanup (P1).** F-09. Fill `MEMORY.md` sections 7 and 8 from real logs (`experiments.csv`, Kaggle submission list).

**A5. Hand-written extras (P2, only if everything else is done).** Per v1 A6, 60 to 120 rows, written without looking at test rows, embedded in the notebook, declared.

**Handoffs:** cross-tab rules to B; EDA numbers to C for the declaration; README to C for the ZIP.

---

### TEAMMATE B: Modelling and submission

**Order:** F-01, F-02, `prediction.py` (B1), F-05, finals (B4), then P1 items.

**B1. `prediction.py` (P0, 60 min).** Wrap the existing `Baseline` and calibrator so that organisers can run it on new text (FR-14):
- `fit_all(df_labeled) -> fitted`: runs grouped CV once (`cross_validate` with `return_oof=True`), fits the calibrator on OOF Task A probabilities with `folds["fold_r0"]`, fits the final `Baseline` on all labeled rows with the saved `group` column. Returns an object with `model`, `calibrator`, and any thresholds.
- `predict_df(df_text, fitted) -> DataFrame`: accepts any DataFrame with `id, text`. Handles empty strings, noise-only strings, 12-message conversations. Chunks at 50,000 rows (the `Baseline.predict` batching already does). Returns the 5-column frame via `submission.build`.
- `main()`: load sample and test, `fit_all`, `predict_df`, validate with C's validator, write `outputs/predictions/<exp>_<hhmm>.csv`.
- `tests/test_predict_df.py`: five unseen strings including `""`, a gibberish string, and a 12-message conversation; assert 5 columns, valid ranges, same ids.

**B2. `submission.py` (P0, 20 min).** `build(ids, label, tech, action, conf) -> DataFrame`: strict 5 columns, techniques serialised with `multihot_to_techniques`, raises on any invalid value. Replace the inline assembly in `train_baseline.main()` with it (F-02).

**B3. `config.py` (P0, 20 min).** F-01. Do this first; C's notebook build depends on it. Keep every existing name so other modules still import.

**B4. Finals (P0).** Produce two files with `prediction.main()` only:
- Candidate A: the current baseline, as validated by grouped CV.
- Candidate B: baseline plus only changes that were CV-positive (per-technique thresholds F-03, rules F-04, tuned `C`). If nothing was CV-positive, B equals A with a different `n_aug` or `C` that CV supports, and say so in `MEMORY.md`.
Submit both, log public scores in `MEMORY.md` section 8 (sanity only), byte-save both CSVs. Keep two submission slots free before 18:00. Record the exact command and seeds used.

**B5. F-05 noise rule (P0, 15 min).** After A reports the real noise-only labels, align `Baseline._predict_batch` noise handling (label, techniques, action) to the data. If the sample has none, keep the current rule and mark it "assumption" in the notebook.

**B6. P1 items (only before the code freeze, one change at a time, same saved folds):**
1. `C` grid for Task A {1, 3, 10} and `n_aug` {2, 3, 5}; keep the simpler model within noise (one or two F1 points on this sample is noise).
2. F-03 per-technique thresholds.
3. F-04 rules from A's cross-tab (hard rules only if 95% or more).
4. F-08 minimal `metrics.py` with `s_hat()` and aligned `experiments.csv` columns.
Promotion gate: accept only if the clean CV improves and no slice (language, single/conversation, hard-legit, corrupted) drops by more than about 3 points. Log every run.

**B7. `tests/test_augment.py` (P1, 20 min).** Seeded determinism, `|||` count preserved, Set A and Set B differ on the same seed for text longer than 20 characters, `expand` inherits `group_id`.

**Handoffs:** `config.py` and `submission.py` to C first. Final CSVs and commands to C. Cross-tab rule requests to A.

---

### TEAMMATE C: Tooling and packaging

**Order:** C1 validator, C2 notebook builder (skeleton immediately), C3 clean run, C4 ZIP.

**C1. `tools/validate_submission.py` and `tests/test_submission.py` (P0, 30 min, do first).**
CLI: `python tools/validate_submission.py <csv> [--sample data/sample_submission.csv] [--test data/test.csv]`; also importable as `validate(path, ...) -> list[str]`. Exit 0 on success, 1 on failure; print each failure with row counts and up to 5 offending ids.
Checks:
1. Header equals the sample submission header exactly (5 columns, same order).
2. Row count equals `test.csv`.
3. Every test id exactly once, none unknown, compared as strings.
4. `label` and `action`: integer strings 0 to 5, reject `2.0`, blanks, text.
5. `techniques`: not empty; `NONE` or `|`-joined sorted unique `T1` to `T6`; `NONE` never mixed.
6. `confidence`: numeric in [0, 1], no NaN.
7. No empty cell, UTF-8 without BOM, no index column.
8. Warnings only: constant confidence, a class never predicted, over 90% in one class.
Tests use synthetic fixtures: good file accepted; float label, empty techniques, duplicate id, missing id, extra id, confidence above 1, wrong column order, NaN rejected.

**C2. `tools/build_notebook.py` (P0, 60 min, skeleton first).** Build `notebooks/notebook.ipynb` with `nbformat`. Cell order:
1. **Markdown declaration** (must be first): libraries and versions (or install cell); every external dataset, pretrained model and API with source and licence (write "none" where true, and state that the TF-IDF vocabulary is fit on train data only); method and validation summary (grouped CV 5x3, corruption Set A for training and Set B for evaluation, calibration); random seed 42 and reproduction steps.
2. Install and version-print cell.
3. One `%%writefile src/<module>.py` cell per module in dependency order (`config`, `data_loader`, `preprocessing`, `augment`, `text_analysis`, `evaluation/calibration`, `training/cross_validation`, `training/train_baseline`, `submission`, `prediction`), then imports. Create the package folders and `__init__.py` files in the setup cell.
4. Data location cell: use `config.find_data_dir()` so `sample.csv` and `test.csv` resolve on Kaggle.
5. Run cells: grouped-CV summary with slices and corruption score, final fit, `predict_df` on test, validator call, write `submission.csv`.
6. Keys are read with `os.getenv` or `python-dotenv`; no key appears in the notebook.
Test locally with `jupyter nbconvert --execute` on a 1,000-row test slice. Declaration text must come from `MEMORY.md` section 11, not be improvised.

**C3. Clean-run rehearsal (P0, 90 min after freeze).** In a new Kaggle or Colab session with internet, run top to bottom. Record wall time and peak memory (give the runtime to A for the README). Compare the produced CSV to the selected Kaggle submission (`DataFrame.equals` or byte equality). If it differs, find the nondeterminism (set `PYTHONHASHSEED`, seed numpy/random, sort before group operations). The reproduced private score must land within 0.01. If the full 500,000-row run is too slow, the preprocessing cache and `Baseline.predict` unique-row batching already help; report what remains slow.

**C4. `tools/make_zip.py` (P0, 20 min).**
- One constant `ZIP_NAME`. Rulebook v1.3 says `NeuraHack2026_<TeamName>_<TeamCode>.zip` (example `NeuraHack2026_MissionX_NH344.zip`); the problem statement said `NeuraHack_Preli_<TeamName>.zip`. **Resolve (D2) from the guidance email or official channel before uploading; do not guess.**
- The ZIP contains **exactly four items** at the root: the final CSV (`submission.csv`, 5 columns, equal to one selected Kaggle submission), the `.ipynb`, `.env` (empty file if no API), `README.md`. Fail if any is missing, if a fifth item would be added, or if `.env` holds a hardcoded key not meant for the competition.
- After building, extract to a temp folder, run `validate_submission`, byte-compare with the saved candidate, and print the listing and SHA-256.

**C5. Organiser questions (10 min).** Record answers in `MEMORY.md` section 3: ZIP name (D2), whether the 5-column format applies (D1; check the Kaggle Data tab). D4 (frozen LLM data) is moot while no LLM data is used.

**Handoffs:** validator to B immediately. Notebook skeleton to A for the declaration text. ZIP to the shared upload step.

---

## Part 4. Shared tasks and checkpoints

| ID | Task | Who | By |
|---|---|---|---|
| S-01 | Run `pytest -q` and `python -m src.training.cross_validation`; paste real results into `MEMORY.md` | All | now |
| S-02 | F-01/F-02 fixed, validator working, `prediction.main()` writes a file that passes the validator | B, C | before 15:00 |
| S-03 | Decide candidates A and B with CV evidence in `MEMORY.md` | All | 15:00 |
| S-04 | **Select the two finals manually on Kaggle**; B selects, A and C verify on screen, one person screenshots | All | **before 18:00** |
| S-05 | Code freeze | All | 17:30 |
| S-06 | `.env` present (empty if no API) and noted in README | C, A verifies | 20:00 |
| S-07 | Compliance checklist (`RULES.md` A6): A signs items 1, 4, 5; B signs 2, 3, 6; C signs 7, 8 | All | 21:15 |
| S-08 | Upload ZIP | All | **21:30, deadline 22:00** |

### Handoff note template (rewrite `MEMORY.md` section 10 at every stop)
```
Last updated: <date time BST> by <A|B|C>
State: <what works, what is half done>
Next 3 steps:
  1.
  2.
  3.
Blockers or questions for teammates:
Where results live: outputs/validation_results/experiments.csv ; outputs/validation_results/folds.csv ; outputs/oof/ ; outputs/predictions/
```

### Quality gates before any Kaggle submission
1. `pytest -q` passes.
2. File produced by `prediction.main()`, never edited by hand.
3. `validate_submission.py` exits 0 on the 5-column file.
4. Logged in `MEMORY.md` section 8 with what changed.
5. Fewer than 8 daily slots used before 18:00.

### Escalation rules
- Disagreement on a modelling choice: grouped-CV result decides, then simplicity.
- A task will miss its window by more than 30 minutes: tell the team and cut P1 items first.
- Any doubt about whether something is allowed: do not use it until `RULES.md` A2 or an organiser confirms.

### Final definition of done
- Two Kaggle finals selected manually; one equals the ZIP's `submission.csv`.
- Notebook reproduces within 0.01 on a clean run and works on new text through `predict_df`.
- Declaration section complete and accurate (no external model, dataset or API, if that remains true).
- Validation story (grouped CV, slices, corruption robustness, error analysis) is inside the notebook.
- ZIP has exactly four items, the resolved name, uploaded before 22:00 BST on 27 Nov.
