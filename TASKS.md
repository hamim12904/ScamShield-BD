# TASKS — ScamShield BD

> Time-boxed plan from now to the ZIP deadline. Owners: **[T]** teammate (inspection, validation, preprocessing), **[N]** Nabila (modelling, augmentation, calibration, submission, packaging), **[A]** either.
> Priorities: **P0** must ship, **P1** should ship, **P2** only if ahead of schedule.
> Check boxes as you go; log decisions and results in `MEMORY.md`. All times BST. Plan written Thu 26 Nov ~23:15.

---

## Timeline at a glance

| Phase | Window | Outcome |
|---|---|---|
| 0 Foundations | Thu 23:15 → Fri 00:30 | Contracts, scaffolding, validator, metrics; EDA first pass |
| 1 Baseline | Fri 00:30 → 01:30 | **M1:** first valid Kaggle submission |
| Rest | Fri ~01:30 → 07:00 | Sleep (stagger if one person wants to run long jobs; set jobs to log + checkpoint) |
| 2 Build | Fri 07:00 → 12:00 | **M2:** Models 1 + 2, augmentation, B/C heads, OOF logged |
| 3 Improve | Fri 12:00 → 15:00 | **M3:** ensemble + calibration; candidate finals trained |
| 4 Freeze | Fri 15:00 → 17:30 | **M4:** final training, two Kaggle finals selected manually |
| 5 Package | Fri 17:30 → 21:30 | **M5:** notebook clean-run, README, `.env`, ZIP validated and uploaded |
| Buffer | 21:30 → 22:00 | Deadline 22:00 |

**Cut line:** if behind at 12:00, drop P2 and Model 3 and LLM generation; focus on Model 1 + corruption augmentation + embeddings + calibration + packaging. If behind at 15:00, drop embeddings ensemble tuning and ship the best validated single model with calibration. Packaging is never cut.

---

## Phase 0 — Foundations (Thu 23:15 → Fri 00:30)

- [ ] **T-001 [N] P0 · 20 min** Create missing folders and stubs per `ARCHITECTURE.md` (`notebooks/`, `outputs/*`, `src/features|models|training|evaluation`, `tools/`, `tests/`, `docs/`); add `.gitignore` entries (`.env`, `data/`, `outputs/cache/`, `.venv/`). *Done when:* repo tree matches the doc and imports resolve.
- [ ] **T-002 [N+T] P0 · 10 min** Agree the frozen interfaces (ARCHITECTURE §4): columns produced by `build_views`, schema of `folds.csv`, name of the group column. *Done when:* both confirm in chat and `MEMORY.md` §4 logs it.
- [ ] **T-003 [T] P0 · 60 min** Stage 1 EDA → `outputs/validation_results/eda_report.md`. Must answer the PRD open questions: class counts, languages, single vs convo, speaker-marker format, duplicate/template clusters, **label × techniques × action cross-tab**, **noise-only rows' labels**, class 5 behaviour. Post findings in `MEMORY.md` §5.
- [ ] **T-004 [T] P0 · 90 min** Stage 2 `preprocessing.py`: three views + structure features + `normalise_chars`, `is_noise_token`. Include zero-width handling and the ZWJ/ZWNJ-in-Bangla rule. Pass `test_preprocessing.py`. (You already have `check_noise.py`; use its findings for noise stats.)
- [ ] **T-005 [T] P0 · 60 min** Stage 3 `cross_validation.py`: duplicate/template clustering → `group_id`; `StratifiedGroupKFold` 5×3; save `outputs/validation_results/folds.csv`; fallback for rare classes; `get_folds`. Pass `test_folds.py`.
- [ ] **T-006 [N] P0 · 30 min** `tools/validate_submission.py`: compare with `sample_submission.csv` and `test.csv` (columns/order, ids exactly once, int labels, `NONE` rule, confidence range, row count). Plus `test_submission.py`.
- [ ] **T-007 [N] P0 · 45 min** `evaluation/metrics.py`: macro-F1 A/C, multi-label macro-F1 B, Brier calibration, composite surrogate, slice reporting.

## Phase 1 — Baseline (Fri 00:30 → 01:30)

- [ ] **T-010 [N] P0 · 60 min** Model 1 end-to-end: `tfidf_features.py`, `task_a/b/c.py`, `train_baseline.py`, `submission.py`, `prediction.py`. Use quick random folds until `folds.csv` lands, then switch. *Done when:* `python -m src.prediction` writes a file that passes the validator.
- [ ] **T-011 [N] P0 · 10 min** Submit to Kaggle → **Submission #1** (log in `MEMORY.md`). Purpose: prove the format and see the public score. Do not tune against it.
- [ ] **T-012 [A] P0 · 10 min** Commit, push, update `MEMORY.md` (state, scores, open issues).

*Rest block. Before sleeping, record "next 3 steps" in `MEMORY.md` §10 (handoff notes).*

## Phase 2 — Build (Fri 07:00 → 12:00)

- [ ] **T-020 [N] P0 · 60 min** `augment.py` with operator sets A/B; `test_augment.py`; integrate into training folds (variants inherit `group_id`, validation stays real-only).
- [ ] **T-021 [N] P0 · 60 min** Grouped-CV runs of Model 1: tune `C`, feature blocks (word vs char vs both vs +structure), with/without augmentation. Log all in `experiments.csv`.
- [ ] **T-022 [T] P0 · 60 min** `error_analysis.py` + slices: language, single/convo, hard-legit, corrupted copies (operator set B); confusion matrices. First report in `validation_results/`.
- [ ] **T-023 [N] P1 · 120 min** Model 2: `embedding_features.py` (cache, length-sorted batches), `train_embeddings.py`. Start with the small multilingual encoder; time 10k rows; extrapolate to 500k before choosing the model.
- [ ] **T-024 [N] P0 · 45 min** Task B thresholds per technique and Task C head on OOF; learn consistency rules from the cross-tab (`postprocess.py`).
- [ ] **T-025 [T] P1 · 90 min** Hand-written hard-legit extras + minimal pairs in `data/extra/handwritten.py` (no test rows used). Add to training with group ids.
- [ ] **T-026 [N] P1 · 120 min** `synthetic_llm.py`: scenario matrix, JSON schema, dedup, round-trip verification; generate ~1–3k rows within budget; freeze `data/extra/synth_v1.jsonl`. Needs a competition-only key with a spending cap in `.env` (not committed).
- [ ] **T-027 [A] P0 · 20 min** Mid-day checkpoint: update `MEMORY.md`, push, confirm `experiments.csv` is current. **M2 reached when Models 1+2 have OOF.**

## Phase 3 — Improve (Fri 12:00 → 15:00)

- [ ] **T-030 [N] P1 · 60 min** `ensemble.py`: equal/one-parameter blend of Model 1 and Model 2 on OOF; compare with each alone on all slices.
- [ ] **T-031 [N] P0 · 60 min** `calibration.py`: confidence model with OOD signal; compare Brier with the constant-confidence baseline; cross-fitted.
- [ ] **T-032 [N] P2 · 90 min** `pseudo_label.py` experiment (τ = 0.9, class caps, ≤ 2 rounds); adopt only if real-only CV improves F_A and the corrupted slice.
- [ ] **T-033 [T] P1 · 60 min** Review predictions on test: class histogram vs sample, `noise_ratio` buckets, random read of ~30 predictions for sanity (read-only; never edit predictions).
- [ ] **T-034 [N] P2 · 120 min** Fine-tune transformer only if the gate in DESIGN §5 is met.
- [ ] **T-035 [A] P0 · 30 min** Decide candidate finals A (conservative) and B (adds CV-positive extras). Write the decision and CV evidence into `MEMORY.md`. **M3.**

## Phase 4 — Freeze and submit (Fri 15:00 → 17:30)

- [ ] **T-040 [N] P0 · 60 min** `prediction.main()` for candidate A and B: fit on all labeled (+ accepted extras), predict 500k, postprocess, calibrate, validate.
- [ ] **T-041 [N] P0 · 20 min** Submit A and B; log public scores (only as a sanity check; do not switch to the public-best by default).
- [ ] **T-042 [A] P0 · 15 min** **Select the two finals manually on Kaggle** and screenshot/note which are selected. Byte-save both CSVs.
- [ ] **T-043 [A] P0 · 5 min** **17:30 code freeze.**

## Phase 5 — Package (Fri 17:30 → 21:30)

- [ ] **T-050 [N] P0 · 45 min** `tools/build_notebook.py`: declaration markdown cells, install cell, `%%writefile` module cells, frozen data cells, run cells. Fill the declaration with every external resource (models, APIs, datasets) and seeds. *(Start a skeleton earlier, e.g. during Phase 2, so only content remains.)*
- [ ] **T-051 [N] P0 · 90 min** **Clean-run rehearsal** on Kaggle or Colab with internet: run top-to-bottom, compare the produced CSV with the selected Kaggle submission (byte or row equality), check runtime. Fix non-determinism.
- [ ] **T-052 [T] P0 · 30 min** `README.md`: team name, approach, external resources (with licences), seeds, run order, approximate runtime, API notes.
- [ ] **T-053 [A] P0 · 15 min** `.env`: the competition-only key (or empty file if no API is called by the notebook); confirm it works and has credit; do not commit it.
- [ ] **T-054 [N] P0 · 20 min** `tools/make_zip.py`: ZIP name per the **resolved** rule (guidance email), assert exactly four items; run `validate_submission.py` on the CSV inside.
- [ ] **T-055 [A] P0 · 20 min** Compliance checklist (`RULES.md` A6). Upload the ZIP. **Target 21:30, deadline 22:00.**

---

## Submission budget (verify limits in the Kaggle UI)

| Slot | Purpose |
|---|---|
| #1 (Fri ~01:30) | Format + baseline sanity |
| #2–#5 | Milestone checks after M2 (corruption aug, embeddings) |
| #6–#8 | Ensemble / calibration variants (**only submit if CV-supported**) |
| #9–#10 | Candidate A and B before the freeze |
| Reserve | Keep at least two slots free before 18:00 for emergencies |

## Definition of Done (whole project)
- [ ] Two selected Kaggle finals, one of which equals the ZIP's `submission.csv`.
- [ ] Notebook reproduces within 0.01 in a clean environment and works on new text via `predict_df`.
- [ ] Declaration section is complete and honest.
- [ ] Validation story (grouped CV, slices, error analysis) is written up inside the notebook.
- [ ] ZIP has exactly four items, correct name, uploaded before 22:00.
