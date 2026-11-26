# RULES — ScamShield BD

> Two kinds of rules: **competition rules** (breaking them can disqualify us) and **team engineering rules** (keep us fast, safe and reproducible). Read this before every submission and before packaging.
> Sources: NeuraHack 2026 Rulebook v1.3, Preliminary Problem Statement, Kaggle overview page (pasted 26 Nov 23:04 BST). Where sources disagree, see `MEMORY.md` §3.

---

## Part A — Competition rules (distilled)

### A1. Eligibility
- Teams of 2–3, all current undergraduates at a university in Bangladesh. One Kaggle account per person. No duplicate accounts or teams.

### A2. Permitted
- Open-source pretrained models (XLM-R, mBERT, BanglaBERT, multilingual sentence embeddings) subject to their licences.
- Hosted LLM / inference APIs for **training, labeling, data generation, augmentation, feature extraction, inference**, *if called from code inside the notebook and declared*.
- Public external datasets, if declared.
- Feature engineering, fine-tuning, ensembling, augmentation, public code (disclosed), AI coding assistants.
- **Writing our own extra training examples** or generating them with a model, provided the process is reproducible from the notebook and **no test example or its label is copied** into training.
- **Unsupervised / semi-supervised use of `test.csv` text**, including pseudo-labeling and domain adaptation, provided it is done **by code in the notebook, uses only model predictions, and is declared**.
- Kaggle Notebooks.

### A3. Prohibited (disqualification risk)
1. Recovering, looking up or inferring test labels by any means other than our model's predictions.
2. Hand labeling or manually editing test predictions, including pasting answers from a chat interface.
3. Hardcoded or manually created prediction/label files.
4. Private sharing of code, data, predictions or keys with other teams.
5. Multiple accounts or submitting for another team.
6. Using any **undeclared** dataset, model or API.
7. Failing to deliver the ZIP on time, omitting `.env` or `README.md`, or a notebook/key that cannot reproduce the score within 0.01.

### A4. Deliverables and deadlines (BST)
| Item | Value |
|---|---|
| Prelims window | 26 Nov 18:00 → 27 Nov 18:00 |
| Final ZIP deadline | **27 Nov 22:00** |
| Daily Kaggle submissions | 10 per team (limit resets at 00:00 UTC = 06:00 BST; verify in the Kaggle UI) |
| Final selection | **We must select 2 final submissions manually.** If we don't, Kaggle picks the two best by *public* score (only 30% of rows) |
| ZIP contents | **Exactly four items:** `submission.csv` (matches one selected Kaggle submission), `notebook.ipynb`, `README.md`, `.env` (empty if no keys) |
| ZIP name | **Unresolved:** rulebook v1.3 says `NeuraHack2026_<TeamName>_<TeamCode>.zip`; problem statement says `NeuraHack_Preli_<TeamName>.zip`. Check the guidance email before building |
| Notebook must begin with | Markdown declaration: libraries/versions (or install cell); all external datasets, pretrained models, APIs with sources and licences/providers; method and validation strategy; random seeds and reproduction instructions |
| README must contain | Team name, approach summary, external resources, seeds, run order, approximate runtime, API usage notes |
| `.env` | Every API key the notebook uses. Notebook reads them via `os.getenv` / `python-dotenv`. Dedicated competition key, spending limit, funded until verification ends; revoke afterwards |
| Verification | Clean re-run on Kaggle/Colab with internet; reproduced private score within **0.01** accepted; larger gap triggers review. Pipeline may also be run on **additional unseen evaluation data** without manual changes |

### A5. Submission file format
```
id,label,techniques,action,confidence
20001,1,T1|T2,1,0.94
20002,0,NONE,0,0.98
```
- Every test `id` exactly once, `id` unchanged, exact column order of `sample_submission.csv`.
- `label`, `action`: integers 0–5 (not `2.0`, not strings). `techniques`: codes T1–T6 joined by `|`, or `NONE` (never empty). `confidence`: number in [0, 1]. No blanks, no extra columns.
- Rulebook v1.3 §4.1 still describes a two-column file (`id,label`); the Kaggle overview and problem statement say five. **`sample_submission.csv` decides.**

### A6. Compliance checklist (run before every final submission)
- [ ] Every external model/dataset/API appears in the declaration cell (name, source, licence/provider, purpose).
- [ ] No API key in any file except `.env`; `.env` is not in git.
- [ ] No test text or test label appears in any training, rule or augmentation list.
- [ ] Hand-written examples live in `data/extra/` and are embedded in the notebook.
- [ ] Pseudo-labeling (if used) is fully programmatic and declared.
- [ ] No rule keys off specific test strings or ids (looks like hardcoding).
- [ ] The notebook produces the submitted file end-to-end; the file in the ZIP equals one of the two selected Kaggle submissions (byte-compare).
- [ ] Notebook runs on new text via `predict_df` with no manual edits.

---

## Part B — Data leakage rules

1. **Group before you split.** Near-duplicates, shared templates and augmented/synthetic variants carry their source's `group_id` and never straddle train/validation.
2. **Validate on real labeled rows only.** Synthetic and augmented rows enter training folds only.
3. **Fit preprocessing inside the fold** whenever it uses labels. Unsupervised fits (TF-IDF vocabulary/IDF on train+test text) are allowed and declared; label-dependent steps (thresholds, calibration, class weights, pseudo-label selection) must be cross-fitted.
4. **Held-out corruption operators.** Training augmentation uses operator set A; robustness evaluation uses operator set B plus A.
5. **Never tune on the public leaderboard.** Public LB is a sanity check (format works, no collapse), not an objective.
6. **Look at test text only for unsupervised purposes** (noise statistics, vocabulary, language mix). Never craft examples or rules from specific test rows.

---

## Part C — Engineering rules

### C1. Workflow
- One repo, `main` always runnable. Short-lived branches `n/<topic>` and `t/<topic>`; merge via quick pull request or at least a fast diff review. Pull before you push; commit small and often.
- Commit messages: `<area>: <what>` (e.g. `preproc: strip zero-width, keep ZWJ in Bangla`).
- **Interfaces in `ARCHITECTURE.md` §4 are frozen.** If you must change one, tell the other person and log it in `MEMORY.md`.
- Never hand-edit `notebooks/notebook.ipynb`. It is generated.

### C2. Code
- Python 3.x, type hints on public functions, short docstrings stating inputs/outputs.
- No global mutable state; pass `rng`/`seed` explicitly. All randomness derives from `config.SEED`.
- No silent fallbacks: log a warning when an optional component is skipped.
- Pure functions in `preprocessing`/`augment`; no printing inside them.
- Keep `requirements.txt` current; pin versions that matter.

### C3. Experiments
- **One change at a time**, compared on the *same* saved folds and the same repeats.
- Log every run in `outputs/validation_results/experiments.csv` (config, git hash, metrics, slices).
- Promotion gate: accept a change only if the surrogate composite improves on repeated grouped CV **and** no slice drops by more than about 3 macro-F1 points. Prefer simpler models when scores are within noise (n is tiny; differences of one or two points on 86 rows are noise).
- Look at errors (`error_analysis.py`) after every accepted change: especially hard-legit false alarms and confusions among classes 1/2/3/4/5.

### C4. Submissions
- Only submit files that were produced by the pipeline (`prediction.main()`), run through `tools/validate_submission.py` first.
- Name files `outputs/predictions/<exp-id>_<hhmm>.csv`; record each Kaggle submission in the `MEMORY.md` submission log (id, public score, what changed).
- **Final pair:** one conservative (best CV, simplest) and one that adds the riskier components (e.g. pseudo-labeling or LLM data) *only if* it was CV-positive. Select both manually before 18:00 on 27 Nov.
- Keep a spare submission slot for the final packaged output check if the rules allow.

### C5. Security
- `.env` and keys: never commit, never print, never paste into chats. `.gitignore` must include `.env`, `data/`, `outputs/cache/`, `.venv/`.
- Use a competition-only API key with a spending cap. The `.env` in the ZIP holds that key (required for reproduction); revoke after verification.

### C6. Working with AI assistants
- Allowed, but every generated line is ours to understand and defend in the notebook review.
- Don't paste test rows or labels into any chat or external service to get "answers"; that risks violating A3.2.
- Ask assistants for code against the frozen contracts; verify with tests before merging.

### C7. Time rules
- At 17:30 Fri: **code freeze.** Only fixes for validated bugs afterwards.
- By 18:00 Fri: both finals selected on Kaggle.
- By 21:30 Fri: ZIP built, validated, uploaded (30-minute buffer before 22:00).
- Take a rest block overnight; late-night bugs in a submission validator cost more than the hours saved.
