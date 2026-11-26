# TEAM_INSTRUCTIONS: ScamShield BD

How to use this file: start Claude Code in the repo root and say "I am Teammate A" (or B or C). Claude Code must then read this file, do Part 0 and Part 1, and execute only the Part 2 section for that teammate, plus the shared tasks in Part 3.

Source docs in `docs/` (read in this order): `MEMORY.md`, `RULES.md`, `ARCHITECTURE.md`, `DESIGN.md`, `PRD.md`, `TASKS.md`. If this file and those docs disagree, the docs win, except for task ownership, where this file wins. Log any conflict in `MEMORY.md` section 3.

---

## Part 0. Operating rules for Claude Code

1. **Identify the teammate.** Work only on files that teammate owns (Part 1.2). If a task needs a change in someone else's file, do not edit it. Write the request in `MEMORY.md` under "Cross-team requests" and continue with a stub.
2. **Read before you write.** At session start read `MEMORY.md` sections 4, 5, 10. At session end rewrite section 10 (handoff notes) and append to the experiment or submission log if relevant.
3. **Frozen interfaces.** The signatures in `ARCHITECTURE.md` section 4 and the file formats in Part 1.3 below do not change without a `MEMORY.md` section 4 entry (next ID: D-008) and a note to the other two teammates.
4. **Engineering standards.**
   - Python 3.x, type hints on public functions, one-line docstring stating inputs and outputs.
   - No printing inside `preprocessing` or `augment`. Use `logging`.
   - No global mutable state. Pass `rng` or `seed` explicitly. All randomness derives from `config.SEED` (42).
   - No silent fallbacks. If an optional component is skipped, log a warning.
   - Public functions get a test. Run `pytest -q` before every commit. Do not commit with failing tests.
   - Commits: `<area>: <what>`, small and frequent. Branch names `a/<topic>`, `b/<topic>`, `c/<topic>`. Pull before push. `main` must always run.
   - Never hand-edit `notebooks/notebook.ipynb`. It is generated.
5. **Compliance (disqualification risk).** Never copy test text or labels into training, rules, examples or chat prompts. Never hand-edit predictions. Never hardcode ids or strings from test rows. Never print or commit keys. Look at test text only for unsupervised statistics (noise ratio, vocabulary, language mix). Every external model, dataset or API must be added to `MEMORY.md` section 11 the moment it is used.
6. **Validation discipline.** Validate on real labeled rows only. Augmented and synthetic rows inherit the source `group_id` and enter training folds only. Compare experiments on the same saved folds. Never tune on the public leaderboard.
7. **Definition of done for any task:** code merged, tests pass, acceptance criteria met, `MEMORY.md` updated, and the next person's input is available at the agreed path.
8. **When blocked** for more than 15 minutes: write the blocker in `MEMORY.md` section 10, switch to the next unblocked task on your list, and tell the team.

---

## Part 1. Shared foundation

### 1.1 Roles

| Teammate | Role | One-line mission |
|---|---|---|
| **A** | Data and quality | Know the data, clean it safely, build leakage-proof validation, audit errors, write human-facing documentation |
| **B** | Modelling and submission | Build the TF-IDF models, augmentation, task heads, calibration, ensemble and the final predictions |
| **C** | Semantic models, extra data, tooling, packaging | Build embeddings, extra training data, validator, notebook generator, ZIP and the clean-run proof |

### 1.2 File ownership

| Owner | Files and folders |
|---|---|
| A | `src/data_loader.py`, `src/preprocessing.py`, `src/text_analysis.py`, `src/training/cross_validation.py`, `src/evaluation/error_analysis.py`, `data/extra/handwritten.py`, `README.md`, `outputs/validation_results/eda_report.md`, `tests/test_preprocessing.py`, `tests/test_folds.py` |
| B | `src/config.py`, `src/augment.py`, `src/features/tfidf_features.py`, `src/models/task_a.py`, `task_b.py`, `task_c.py`, `ensemble.py`, `src/training/train_baseline.py`, `src/evaluation/metrics.py`, `src/evaluation/calibration.py`, `src/postprocess.py`, `src/submission.py`, `src/prediction.py`, `tests/test_augment.py`, `tests/test_predict_df.py` |
| C | `tools/validate_submission.py`, `tools/build_notebook.py`, `tools/make_zip.py`, `src/features/embedding_features.py`, `src/training/train_embeddings.py`, `src/synthetic_llm.py`, `src/pseudo_label.py`, `data/extra/synth_v1.jsonl`, `tests/test_submission.py`, `.env.example`, `requirements.txt`, `SETUP.md` |
| Shared | `MEMORY.md`, `docs/*`, `.gitignore`, `outputs/*` (each person writes only their own subfolder files, see 1.3) |

`config.py` is created by B in the first 20 minutes (T-001) because everyone imports it. A and C wait for that commit before importing.

### 1.3 Frozen data contracts

**Labeled frame** (`data_loader.load_sample()`): `id, text, label, techniques, action`. `techniques` is the raw string.

**Views frame** (`preprocessing.build_views(df)`): all input columns unchanged plus `text_raw, text_norm, text_skel, n_msgs, first_msg, last_msg, mid_msgs, has_url, n_urls, noise_ratio, script_bangla_ratio, script_latin_ratio, emoji_count, digit_ratio, upper_ratio, excl_count, qmark_count, mean_msg_len, max_msg_len, has_otp_like_number, asks_question, first_msg_len`. Never drops rows, never edits `id`.

**Folds file** `outputs/folds.json`:
```json
{"n_splits": 5, "n_repeats": 3, "seed": 42,
 "group_id": {"<row_id>": <int>},
 "folds": {"<row_id>": {"0": <fold>, "1": <fold>, "2": <fold>}}}
```
`get_folds(df)` returns a list of `(train_idx, val_idx)` of length `n_splits * n_repeats`, ordered by repeat then fold.

**OOF file** `outputs/oof/<exp_id>_<model>.npz` with arrays: `ids` (n,), `repeat` (n,), `A` (n,6) probabilities, `B` (n,6) independent probabilities, `C` (n,6) probabilities. Rows are the labeled rows in `sample.csv` order, averaged over repeats. `<model>` is `m1`, `m2`, `m3` or `ens`.

**Test probability file** `outputs/predictions/<exp_id>_<model>_test.npz`: same arrays for `test.csv` order, fit on all labeled data.

**Label maps** (single source, `config.py`): classes 0 to 5 as in `PRD.md` section 4. Actions 0 to 5. Techniques `T1` to `T6`, serialised as `T1|T2` in sorted order, `NONE` when empty.

**Submission** (5 columns, exact order of `sample_submission.csv`): `id,label,techniques,action,confidence`. Integers for `label` and `action`, `NONE` never empty, confidence a float in [0.03, 0.97] unless the validator says otherwise.

### 1.4 Hour-by-hour dependency map (BST)

| By | Who delivers | Who waits |
|---|---|---|
| Thu 23:35 | B: `config.py`, repo skeleton, stubs (T-001) | A, C |
| Thu 23:45 | All: interface agreement logged (T-002) | All |
| Fri 00:00 | C: `validate_submission.py` (T-006) | B for first submission |
| Fri 00:15 | A: first-pass EDA findings in `MEMORY.md` section 5 | B, C |
| Fri 00:30 | A: `preprocessing.py` and `folds.json` (T-004, T-005). B: `metrics.py` (T-007) | B uses plain text until then |
| Fri 01:30 | B: baseline submission #1 (M1) | |
| Fri 12:00 | C: Model 2 OOF and test probabilities. B: Model 1 tuned OOF. A: first error report | B ensemble |
| Fri 15:00 | Candidate finals A and B decided (T-035) | |
| Fri 17:30 | Code freeze | |
| Fri 21:30 | ZIP uploaded | |

Cut line: at 12:00 drop P2 items, Model 3 and LLM generation. At 15:00 drop embedding-ensemble tuning. Packaging is never cut.

---

## Part 2. Teammate instructions

---

### TEAMMATE A: Data and quality

**Mission.** Make sure everyone trains on clean, correctly understood data and evaluates on a validation scheme that cannot leak. Then find where the models fail and document the project for human reviewers.

**Order of work:** T-003, T-004, T-005 (in parallel where possible: start preprocessing while the EDA report is being written), then T-022, T-025, T-033, T-052.

#### A1. `data_loader.py` (20 min, do first)
- `load_sample()`, `load_test()`, `load_submission_template()` using `config.DATA_DIR`.
- Read CSV with `encoding="utf-8"`, `keep_default_na=False`, `dtype=str`. Cast `label` and `action` to int in `load_sample` only. Empty or NaN `text` becomes `""`, never an exception.
- Test: loads both files, ids unique, row counts logged.

#### A2. T-003 EDA, `outputs/validation_results/eda_report.md` (60 min)
Use `text_analysis.py` for helpers. Answer each question with a number or a table and copy the answers into `MEMORY.md` section 5.
1. Labeled row count and class counts. Confirm or correct the "about 86" assumption (section 6 of MEMORY).
2. Language mix: English, Bangla script, Banglish, mixed. Use script ratios plus a Banglish word list of common romanised Bangla words (build the list from `sample.csv` only).
3. Single message versus conversation counts. Distribution of `n_msgs`, maximum, mean length in characters.
4. Speaker-marker format. Print 5 raw examples and write the regex that matches them.
5. Duplicate and template clusters: exact duplicates after normalisation, then near duplicates (char 3-gram cosine above 0.8). Report cluster sizes.
6. Cross-tab `label x techniques x action`. List every rule that holds in at least 95% of rows (needed by B's `postprocess.py`). Report counts, not only percentages.
7. Labels, techniques, action of noise-only rows (`noise_ratio` near 1, empty, or only gibberish tokens). If the sample has none, say so and write "unknown".
8. Class 5 behaviour: techniques and action values.
9. Test set statistics, unsupervised only: `noise_ratio` histogram, language mix, message count distribution, share of rows with Unicode lookalikes or zero-width characters (use `check_noise.py`). Do not print or copy individual test rows into any notes.
10. A short "hard cases" list from the labeled data: legit rows that contain OTP, link, cashback, parcel, urgent.

Acceptance: report exists, all ten questions answered or marked unknown with a reason, `MEMORY.md` section 5 filled, assumptions in section 6 ticked or corrected.

#### A3. T-004 `preprocessing.py` (90 min)
Implement exactly the three-view design in `DESIGN.md` section 2.
- `normalise_chars(s: str) -> str`: NFKC, remove U+200B, U+2060, U+FEFF, keep U+200C and U+200D only when adjacent to a Bangla character (U+0980 to U+09FF), map Unicode lookalikes (Cyrillic, Greek, fullwidth, mathematical alphanumerics) to Latin only inside tokens that are otherwise Latin, Bangla digits to ASCII, re-fuse defanged links (`hxxp`, `[.]`, `(dot)`, `[:]`), collapse whitespace. Keep emojis, numbers, casing, `|||` and speaker markers.
- `is_noise_token(tok: str) -> bool`: gibberish detector. Use vowel-to-consonant ratio, longest consonant run, character entropy, and absence from the vocabulary built from `sample.csv`. Never flag Bangla-script tokens, numbers, URLs or tokens shorter than 5 characters. Tune on `sample.csv` (should flag `wkrjfejejrn`-style strings, should flag none of the real words). Report precision on a hand-checked list of 50 tokens.
- `build_views(df) -> DataFrame`: adds the columns in 1.3. `text_skel` does: lowercase, de-leetspeak only inside tokens that mix letters and `0 1 3 4 5 @ $`, collapse 3 or more repeats to 2, re-join spaced-out letters (`v e r i f y`, `v.e.r.i.f.y`), remove all zero-width characters, noise tokens become `<NOISE>`, digit runs become shape tokens (`<N6>`) while small amounts keep magnitude (`<N3>`).
- `first_msg`, `last_msg`, `mid_msgs` come from splitting on `|||`. Single messages: first equals last, mid is empty.
- `noise_ratio` = share of whitespace tokens that are noise tokens (1.0 for empty text).
- Performance: vectorised or chunked. Must process 500,000 rows in under 5 minutes on a laptop. Benchmark and log it.

Tests (`tests/test_preprocessing.py`): idempotent (`f(f(x)) == f(x)` on `text_norm` and `text_skel`), `|||` preserved, Bangla characters preserved, ZWJ kept inside a Bangla conjunct, all zero-width removed from skel, non-empty input never gives empty output, empty string and None do not raise, lookalike `Vеrify` (Cyrillic e) maps to `Verify`, `0TP` inside a mixed token maps to `otp` in skel but a standalone `0` is untouched.

Acceptance: tests pass, benchmark logged, B and C notified by commit message `preproc: v1 ready`.

#### A4. T-005 `cross_validation.py` (60 min)
- Group assignment: normalise text, hash exact duplicates, then union-find over near-duplicate pairs (char 3-gram cosine above 0.8 on `text_skel`). Also group by shared first message. Result is `group_id` per row.
- `make_folds(df, n_splits=5, n_repeats=3, seed=42)`: `StratifiedGroupKFold` with a different seed per repeat. If a class has fewer rows than `n_splits`, fall back to fewer splits for that run and log a warning. Write `outputs/folds.json` in the 1.3 format.
- `get_folds(df)`: reads the file, validates that every id is present, returns index pairs.
- Also provide `leave_cluster_out_folds(df)` for the domain-shift proxy: clusters (template or brand groups) held out entirely, at most 5 folds.
- Tests (`tests/test_folds.py`): no group in both train and validation; every row in validation exactly once per repeat; class presence per fold logged; determinism with the same seed; augmented rows (add a fake `group_id`) stay with their source.

Acceptance: `folds.json` exists and is committed to `outputs/`, tests pass, class counts per fold printed in the EDA report. Tell B and C by commit `cv: folds.json v1`.

#### A5. T-022 `error_analysis.py` and slice reports (60 min, Fri 07:00)
- `slice_masks(df) -> dict[str, np.ndarray]`: language (4), single versus conversation, hard-legit subset (legit or class 5 rows containing OTP, link, parcel, cashback, urgent, bill), corrupted copies (generated by B's `augment.corrupt` operator set B, with fixed seed).
- `report(y_true, y_pred, probs, df, name)`: writes to `outputs/validation_results/`: confusion matrix PNG for Task A, per-class precision, recall, F1, slice F1 table (markdown), list of legit predicted as scam and scam predicted as legit (ids and truncated text, labeled rows only), confusions among 1, 2, 3, 4, 5, per-technique precision and recall at the chosen thresholds, robustness curve (F1 versus corruption level light, medium, heavy).
- Run it on every accepted model change (B runs it with one call: `error_analysis.report(...)`).

Acceptance: one full report for Model 1 before 10:00, with a short written summary of the top three failure patterns appended to `MEMORY.md`.

#### A6. T-025 hand-written extras `data/extra/handwritten.py` (90 min)
- A Python list of dicts with `text, label, techniques, action, group_id, source="hand"`. 60 to 120 rows.
- Cover the hard-legit families from `PRD.md` section 4: real OTP with "do not share", transaction confirmation, real delivery tracking, bill reminder, service notice with a link, genuine promo and cashback (class 5), a friend warning about a scam. Write each in English, Bangla script and Banglish.
- Write minimal pairs: for each legit item, a scam sibling sharing most words (real OTP versus OTP harvesting, real tracking versus fake parcel fee).
- Follow the label priority 1 > 2 > 4 > 3 > 5 > 0 and the cross-tab rules from A2.
- Write from your own knowledge of the domain. Never look at or paste test rows. Links defanged. No real phone numbers or names.
- Add a top comment block describing method, date and author for the notebook declaration. Give each pair a shared `group_id` prefix that never overlaps real-row groups.

Acceptance: file imports cleanly, passes a schema test, B can load it with one function call `load_handwritten()`.

#### A7. T-033 test prediction review, read-only (60 min, Fri 12:00)
- Compare predicted class histogram with the sample histogram, report `noise_ratio` buckets against predicted class, share of confidence above 0.9, share of empty `techniques`. Read about 30 random predictions for sanity and write observations (not edits) to `MEMORY.md`. Do not alter any prediction file.

#### A8. T-052 `README.md` (30 min, Fri 15:30)
Contents required by `RULES.md` A4: team name, approach summary, every external resource with licence, seeds, run order, approximate runtime, API usage notes. Write plain, specific text. Pull facts from `MEMORY.md` section 11, never from memory.

**A's handoffs:** EDA findings to B (cross-tab rules) and C (noise and language mix). Preprocessing and folds to both. Extras to B. Error reports to B. README text to C for the ZIP.

---

### TEAMMATE B: Modelling and submission

**Mission.** Produce the best reliable predictions for all four tasks with the TF-IDF family, make them robust to corruption, calibrate the confidence, blend with C's embedding model, and generate the final submission files.

**Order of work:** T-001, T-007, T-010, T-011, T-020, T-021, T-024, T-030, T-031, T-040, T-041.

#### B1. T-001 scaffolding and `config.py` (20 min, first)
- Create every folder and stub file in `ARCHITECTURE.md` section 2, each stub with a docstring and `raise NotImplementedError`, except config.
- `config.py` as in `ARCHITECTURE.md` section 5, plus: label name maps, `TECHNIQUES = ["T1".."T6"]`, `find_data_dir()`, `set_seed(seed)` that seeds `random`, `numpy`, and torch if installed, `CONF_CLIP = (0.03, 0.97)`.
- `.gitignore`: `.env`, `data/`, `outputs/cache/`, `.venv/`, `__pycache__/`, `*.npy`. Keep `outputs/folds.json` tracked.
- Make `python -c "import src.config"` work from the repo root. Add `src/__init__.py` in every package. Commit `scaffold: v1` and tell A and C.

#### B2. T-007 `metrics.py` (45 min)
- `macro_f1_multiclass(y_true, y_pred, labels=range(6))`: include classes absent from `y_true` with F1 0 only if absent from both too, then skip; document the rule in the docstring and use `zero_division=0`.
- `macro_f1_techniques(Y_true, Y_pred)`: both (n,6) binary matrices, F1 per technique, mean over T1 to T6, computed over all rows (legit rows are all-zero rows).
- `brier_calibration(conf, correct)`: returns `1 - mean((conf - correct)**2)`.
- `s_hat(F_A, F_B, F_C, F_domain, F_robust, Cal, F_novel)` using the PRD weights. Proxies: F_domain from leave-cluster-out CV, F_robust from corrupted copies (operator set B), F_novel set equal to F_domain unless C supplies a generation-B proxy.
- `score_all(...) -> dict` with `F_A, F_B, F_C, Cal, S_hat` and per-slice `F_A`. `log_experiment(row: dict)` appends to `outputs/validation_results/experiments.csv` with columns: `exp_id, timestamp, git_hash, config_json, F_A, F_B, F_C, Cal, S_hat, F_domain, F_robust, notes` plus one column per slice.
- Tests against hand-computed examples (perfect, all-wrong, one absent class, calibration with constant 0.5).

#### B3. T-010 Model 1 baseline (60 min)
Files: `tfidf_features.py`, `task_a.py`, `task_b.py`, `task_c.py`, `train_baseline.py`, `submission.py`, `prediction.py`.
- `TfidfBlock`: configurable blocks: word 1 to 2 grams on `text_norm` (token pattern must match Bangla: `(?u)[\w\u0980-\u09FF]+`), `char_wb` 2 to 5 on `text_skel`, optional `char` 1 to 4 on `text_raw`, first-message and last-message word blocks, scaled structure features from 1.3. `sublinear_tf=True`, `min_df=1`, `max_features` capped (start 300,000). Fit vocabulary on train plus test text (unsupervised, declare in `MEMORY.md` section 11). Return one `csr_matrix` via `hstack`. Pickle the fitted object to `outputs/cache/`.
- Task A: multinomial `LogisticRegression(class_weight="balanced", max_iter=2000)`, `C` grid {0.5, 1, 2, 5, 10, 20}. Also `LinearSVC` for comparison with softmax over scaled decision values.
- Task B: six independent binary `LogisticRegression` with `class_weight="balanced"`. If a technique has fewer than 2 positives in a training fold, output a constant prior and log a warning.
- Task C: multinomial LR. Option to add Task A OOF probabilities as features (cross-fitted).
- Before `folds.json` exists, use plain `StratifiedKFold(5, shuffle=True, random_state=42)` and log that it is provisional. Switch to `get_folds` as soon as A commits.
- `submission.build(ids, label, tech, action, conf)` returns the strict 5-column frame. Techniques serialised as sorted `T1|T3` or `NONE`. Raises on any invalid value.
- `prediction.py`: `fit_all(df_labeled, df_test_text)` returns a `fitted` object. `predict_df(df_text, fitted)` works on any text (empty string, noise-only string, 12-message conversation) and processes in chunks of 50,000. `main()` fits, predicts, validates and writes `outputs/predictions/<exp>_<hhmm>.csv`.
- Confidence for the first submission: max class probability shrunk toward OOF accuracy. Real calibration comes in B7.

Acceptance (T-011): `python -m src.prediction` writes a file that passes C's `validate_submission.py`. Submit to Kaggle as submission #1, log it in `MEMORY.md` section 8. Do not tune against its score.

#### B4. T-020 `augment.py` (60 min)
- `corrupt(text, rng, level) -> str`, `level` in `light, medium, heavy`; `operator_set` argument `"A"` or `"B"`.
  - Set A (training): leet and lookalike substitution, character repeat, random insertion, random deletion, spacing changes, random capitalisation, `!!!` or emoji suffix, zero-width insertion.
  - Set B (evaluation only, never used in training): a different homoglyph table, word-boundary removal, transliteration-style respelling (for example `ph` to `f`, `v` to `bh` patterns on Banglish only), embedded noise strings, heavy mixed corruption.
  - Label-preserving rules: always keep `|||`; never corrupt more than 30% of characters on `heavy`; protect at least one decisive cue (token from the row's top TF-IDF terms) from deletion; never turn text into empty.
- `expand(df, k, seed) -> DataFrame`: `k` variants per row (2 to 5), adds `group_id` (source row's), `source="aug"`, `weight`. Weights 1.0 light, 0.8 medium, 0.5 heavy.
- Tests: seeded determinism, `|||` count preserved, length bounds, set A and set B never produce identical output on the same seed for text longer than 20 characters, group_id inherited.
- Wire into `train_baseline.py`: augmentation applied inside the training fold only. Validation rows stay real and clean.

#### B5. T-021 tuning, one change at a time (60 min)
Run on the same saved folds with 3 repeats. Compare in this order and log every run:
1. word only, char only, both, both plus structure.
2. `C` grid.
3. first and last-message blocks on or off.
4. augmentation on or off, then number of variants.
5. add A's hand-written extras (weight 1.0) and, when ready, C's synthetic data (weight 0.3 to 0.5).
6. LR versus LinearSVC.
Promotion gate (`RULES.md` C3): accept only if the surrogate score improves on repeated grouped CV and no slice drops more than 3 macro-F1 points. If the difference is within noise (one or two points), keep the simpler model. After each accepted change call A's `error_analysis.report`.

#### B6. T-024 Task B thresholds, Task C head, `postprocess.py` (60 to 75 min)
- Per-technique threshold: grid 0.05 to 0.95, pick the value maximising OOF F1 per technique, cross-fitted (choose on 4 folds, apply on the 5th, report the average). Shrink toward 0.5 when the technique has fewer than 5 positives.
- `apply_rules(label, tech, action, probs)`: implement only the rules A found at 95% or more in the cross-tab. Hard rules for the confirmed ones, soft prior reweighting for hypotheses (1 to T2, 2 to T4, 3 to T6, 4 to T1 or T3). If a scam label has no technique above threshold, emit the argmax technique. Log how many rows each rule changed.
- Rules must never key off text strings or ids.

#### B7. T-030 and T-031 ensemble and calibration (120 min, Fri 12:00)
- `ensemble.py`: `blend(prob_list, weights)` for probabilities. Start with equal weights, then one mixing parameter in {0.3, 0.4, 0.5, 0.6, 0.7} chosen on OOF. No stacking with this little data. Compare to each model alone on every slice. Needs C's `m2` OOF file in the 1.3 format.
- `calibration.py`: `fit_confidence(oof_probs, y_true, extra_feats)` and `apply(...)`. Target is 1 when the Task A prediction equals the true label. Features: max probability, top-2 margin, entropy, disagreement between models, `noise_ratio`, OOD signal (max cosine similarity to training embeddings, supplied by C). Model: logistic calibrator with 2 to 4 parameters, or temperature scaling. No isotonic unless OOF n is several hundred. Cross-fit. Clip to [0.03, 0.97]. For noise-only rows shrink toward base accuracy on such rows.
- Must beat the constant-confidence baseline (accuracy as constant) in Brier. If not, ship the constant baseline and note it in `MEMORY.md`.

#### B8. T-040 and T-041 final candidates (Fri 15:00 to 17:30)
- Candidate A (conservative): the best simple model by grouped CV.
- Candidate B: adds only components that were CV-positive (embeddings blend, synthetic data, pseudo-labels).
- Run `prediction.main()` for each, validate, submit both to Kaggle, record public scores in `MEMORY.md` section 8 as sanity checks only. Keep two submission slots free before 18:00. Byte-save both CSVs under `outputs/predictions/`.
- The final selection on Kaggle is shared task T-042.

**B's handoffs:** `config.py` to all (first). Predictions and the two candidate CSVs to C for packaging. The exact command and seeds used for each candidate, written in `MEMORY.md`, for C's clean-run rehearsal.

---

### TEAMMATE C: Semantic models, extra data, tooling and packaging

**Mission.** Add the semantic model and external data that help unseen scenarios, build the tools that stop format mistakes, and own the path from code to a verified ZIP that reproduces on a clean machine.

**Order of work:** T-006, T-023, T-026, T-032, T-034 (gated), T-050 (skeleton early), T-051, T-054, plus ZIP naming and organiser questions.

#### C1. Organiser questions (10 min, Thu night)
Post in the official channel (or find in the guidance email) and record the answers in `MEMORY.md` section 3:
1. ZIP file name rule (D2).
2. Whether a frozen LLM-generated dataset embedded in the notebook is accepted, with the generation code kept (D4).
3. Whether `sample_submission.csv` columns decide the Kaggle format (D1). Check the Kaggle Data tab directly.
Do not paste any competition data into the message.

#### C2. T-006 `tools/validate_submission.py` and `tests/test_submission.py` (30 min, do first)
CLI: `python tools/validate_submission.py <csv> [--sample data/sample_submission.csv] [--test data/test.csv]`. Exit code 0 on success, 1 on any failure, print every failure with row counts and up to 5 offending ids.
Checks:
1. Header equals the sample submission header exactly, same order, no extra columns.
2. Row count equals `test.csv` row count.
3. Every test id appears exactly once, no unknown ids, ids not altered (compare as strings, so `007` stays `007`).
4. `label` and `action`: integer strings 0 to 5, rejecting `2.0`, blanks, text.
5. `techniques`: not empty; either `NONE` or `|`-joined codes from T1 to T6, no duplicates, sorted, no spaces; `NONE` never mixed with others.
6. `confidence`: numeric, within [0, 1], not NaN, no blanks.
7. No NaN or empty cell anywhere. File is UTF-8, no BOM, no index column.
8. Soft warnings (do not fail): all confidences identical, a class never predicted, over 90% of rows in one class.
Tests: accept a good file; reject each bad pattern (float label, empty techniques, duplicate id, missing id, extra id, confidence above 1, wrong column order, NaN). Build the fixtures from synthetic ids, not test rows.
Acceptance: B can call it as a function (`validate(path, ...) -> list[str]`) and as a CLI. Commit `tools: validator v1` and tell B.

#### C3. T-023 Model 2 embeddings (120 min, Fri 07:00)
- `embedding_features.embed(texts, model_name, batch, max_len) -> np.ndarray`: sort by length, dynamic padding, fp16 on GPU when available, L2 normalise, cache to `outputs/cache/emb_<model>_<hash>.npy` keyed by model and a hash of the text list. Handle empty strings. Use `config.MAX_LEN`. For `intfloat/multilingual-e5-base` prepend `query: `.
- Model selection: time 10,000 rows of `test.csv` text for `paraphrase-multilingual-MiniLM-L12-v2` first. Extrapolate to 500,000 rows. If the estimate is under about 60 minutes on Kaggle GPU, also try `multilingual-e5-base`. Skip LaBSE and `bge-m3` unless time is clearly available. Record timings in `MEMORY.md`.
- Inputs: `text_norm`. Optionally a second embedding of `last_msg`, concatenated.
- `train_embeddings.py`: LR on embeddings for A and C (multinomial), six binary heads for B. Same folds, same OOF format as Model 1 (`m2`). Save OOF to `outputs/oof/` and test probabilities to `outputs/predictions/` per 1.3. Log every run through B's `log_experiment`.
- Provide `max_cosine_to_train(emb_query, emb_train)` for B's OOD signal, computed cross-fitted (a training row never compares with itself).
- Add every encoder to `MEMORY.md` section 11 with licence from the model card.
- Handoff deadline to B: OOF and test probabilities by Fri 12:00. If the large model misses it, deliver MiniLM first and the larger one later.

#### C4. T-026 `synthetic_llm.py` and frozen data (120 min, optional but high value)
Run only if the organisers' answer on D4 is yes or still pending and a capped competition-only key exists in `.env`.
- Scenario matrix: 6 classes x {English, Bangla script, Banglish, mixed} x {single message, 2 to 4 messages, 5 to 12 messages with delayed intent} x scenario seeds (MFS agent or support, mistaken transfer, prize, job or task or crypto, parcel or customs or address, promo, plus hard-legit seeds) x brand lists (several MFS, couriers, telcos, e-commerce).
- Minimal pairs: for every scam template also generate a legit sibling that shares vocabulary.
- Prompt: include the label definitions and the priority order 1 > 2 > 4 > 3 > 5 > 0; request strict JSON `{messages[], label, techniques[], action, language}`; vary style; forbid real phone numbers and PII; links defanged. Join messages with `|||`.
- Filters in order: JSON schema check; dedup by normalised hash and char-n-gram cosine above 0.9; round-trip label check with an independent labeling call (drop disagreements); drop anything near-identical (cosine above 0.9) to any test row. The test comparison is computed in code and only the drop decision is used; never print or inspect the matched test rows.
- Output: `data/extra/synth_v1.jsonl` with `group_id` (synthetic family id, never equal to a real-row group), `source="llm"`, `weight` 0.3 to 0.5. Target 1,000 to 3,000 rows within budget. Log provider, model, date, cost, prompt version.
- Guard generation behind `REGENERATE_SYNTH` (default False). Read the key with `os.getenv("LLM_API_KEY")`. If the key is missing, the pipeline loads the frozen file and logs it.
- Evaluate with B: real-only grouped CV with and without synthetic rows. Drop the data if real-only CV does not improve or if any slice falls by more than 3 points.

#### C5. T-032 and T-034 gated experiments (Fri 12:00)
- Pseudo-labeling (`pseudo_label.py`, P2): take test rows where Model 1 and Model 2 agree and ensemble probability is at least 0.9, cap per class, skip noise-only rows, up to 2 rounds. Fully programmatic. Pseudo-labeled rows go into training folds only. Adopt only if real-only grouped CV improves F_A and the corrupted slice.
- Fine-tuned transformer (P2): only if the DESIGN section 5 gate holds (GPU, 2 to 3 thousand clean examples, Models 1 and 2 and calibration done, at least 3 hours left). Otherwise skip and record "gate not met" in `MEMORY.md`.

#### C6. T-050 `tools/build_notebook.py` (45 min, skeleton early, fill after freeze)
- Build `notebooks/notebook.ipynb` with `nbformat` from a manifest. Cell order:
  1. Markdown declaration: libraries and versions, every external model, dataset and API with source and licence or provider, method, validation strategy, seeds, reproduction steps. Pull the content from `MEMORY.md` section 11, so it cannot drift.
  2. Install and version-print cell.
  3. One `%%writefile src/<module>.py` cell per module, in dependency order, then plain imports.
  4. Frozen data cells (hand-written extras; synthetic set compressed with gzip and base64 if used).
  5. Run cells: grouped-CV summary with slices and confusion matrix, final fit, `predict_df`, validator call, write `submission.csv`.
- The notebook reads keys with `python-dotenv` or `os.getenv`, never contains one.
- Test: run `jupyter nbconvert --execute` locally on a 1,000-row slice of the test file, with embeddings off if needed, to prove the cells run in order.

#### C7. T-051 clean-run rehearsal (90 min, Fri 17:30 to 19:00)
- On Kaggle or Colab with internet, new session, upload the repo-free notebook and data only. Run top to bottom. Record wall time and peak memory.
- Compare the produced `submission.csv` with the selected Kaggle submission: byte equality or row equality (`pandas.DataFrame.equals`). If different, find the non-determinism (set seeds for numpy, sklearn, torch; fix `PYTHONHASHSEED`; sort before group operations; `torch.use_deterministic_algorithms` for embeddings where practical) and rerun.
- If the full run exceeds the Kaggle session limit, make the heavy steps (embeddings) cache-aware and time-boxed, and write the fallback rule into `README.md`.

#### C8. T-054 `tools/make_zip.py` (20 min, then use at 20:30)
- One constant `ZIP_NAME` set from the resolved rule in C1. Never guess. Until resolved, build both candidate names locally, upload nothing.
- Include exactly four items at the archive root: `submission.csv`, `notebook.ipynb`, `README.md`, `.env`. Fail if any is missing, if a fifth item would be added, or if `.env` contains a key in plain text that is not the competition key.
- Run `validate_submission` on the CSV inside the ZIP after building (extract to a temp folder). Byte-compare it with the saved final candidate.
- Print the archive listing and SHA-256.

**C's handoffs:** validator to B (first). Embedding OOF and test probabilities to B (by 12:00). Frozen synthetic file to B and to the notebook builder. Final ZIP to the shared upload step.

---

## Part 3. Shared tasks and checkpoints

| ID | Task | Who | When |
|---|---|---|---|
| T-002 | Agree frozen interfaces (Part 1.3 and `ARCHITECTURE.md` section 4). Log in `MEMORY.md` section 4 | A, B, C | Thu 23:45 |
| T-012 | First commit, push, `MEMORY.md` update | B leads | Fri 01:30 |
| T-027 | Mid-day checkpoint: `experiments.csv` current, handoff notes rewritten | All | Fri 12:00 |
| T-035 | Decide finals A and B with CV evidence in `MEMORY.md` | All | Fri 15:00 |
| T-042 | Select the two finals manually on Kaggle. B selects, A and C verify on screen, one person screenshots | All | by Fri 17:00 |
| T-043 | Code freeze. Afterwards only fixes for validated bugs | All | Fri 17:30 |
| T-053 | `.env` holds the competition key (or is empty if no API is called); confirm it works and has credit | C does, A verifies | Fri 20:00 |
| T-055 | Compliance checklist (`RULES.md` A6). Split: A signs items 1, 4, 5. B signs 2, 3, 6. C signs 7, 8. Then upload the ZIP | All | target 21:30, deadline 22:00 |

### Handoff note template (rewrite `MEMORY.md` section 10 at every stop)
```
Last updated: <date time BST> by <A|B|C>
State: <what works, what is half done>
Next 3 steps:
  1.
  2.
  3.
Blockers or questions for teammates:
Where results live: outputs/validation_results/experiments.csv ; outputs/folds.json ; outputs/oof/ ; outputs/predictions/
```

### Quality gates before any Kaggle submission
1. `pytest -q` passes.
2. File produced by `prediction.main()`, never edited by hand.
3. `validate_submission.py` exits 0.
4. Submission logged in `MEMORY.md` section 8 with what changed.
5. Fewer than 8 slots used before 18:00.

### Escalation rules
- Disagreement on a modelling choice: the grouped-CV result on the surrogate score decides, then simplicity.
- A task will miss its window by more than 30 minutes: tell the team and apply the cut line in Part 1.4.
- Any doubt about whether something is allowed: do not use it until `RULES.md` section A2 or an organiser confirms it.

### Final definition of done
- Two Kaggle finals selected manually; one equals the ZIP's `submission.csv`.
- Notebook reproduces within 0.01 on a clean run and works on new text through `predict_df`.
- Declaration section complete and accurate.
- Validation story (grouped CV, slices, error analysis) is inside the notebook.
- ZIP has exactly four items, the correct name, uploaded before 22:00 BST on 27 Nov.
