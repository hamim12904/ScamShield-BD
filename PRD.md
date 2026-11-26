# PRD — ScamShield BD (NeuraHack 2026 Preliminary)

> Product requirements for our competition entry. Source of truth for *what* we build and *how we know it worked*.
> Companion docs: `ARCHITECTURE.md` (structure), `DESIGN.md` (ML design), `RULES.md` (constraints), `TASKS.md` (plan), `MEMORY.md` (running log).
> Times are BST (UTC+6) unless stated.

---

## 1. Summary

Build a system that reads an SMS or chat message, or a conversation of up to 12 messages separated by `|||`, in English, Bangla, Banglish or mixed text, and outputs one prediction row per record:

`id,label,techniques,action,confidence`

The system must generalise to unseen brands and scenarios, survive adversarial text corruption, and stay honest about its own uncertainty. We have very little labeled data (86 labeled examples, per our own notes) and a 500,000-row unlabeled test set, so *how we extend supervision* is the core of the challenge.

## 2. Goals

| # | Goal | Measure |
|---|------|---------|
| G1 | Rank in the **Top 60** on the private leaderboard (shortlist for verification) | Private composite score S |
| G2 | Survive **verification** and be one of the **Top 35** | Reproducible notebook, clean validation, generalisation analysis |
| G3 | Be **robust**, not leaderboard-tuned | Small gap between clean and corrupted CV; stable across grouped CV folds |
| G4 | Produce a **valid, reproducible ZIP** on time | Passes `tools/validate_submission.py`; clean-run reproduces score within 0.01 |

**Non-goals:** beating the public leaderboard at the cost of robustness; building a UI; latency optimisation beyond "500k rows finish comfortably"; hand-labeling anything in the test set (prohibited).

## 3. Scoring (what we optimise)

```
S = 0.40·F_A + 0.15·F_B + 0.10·F_C + 0.10·F_domain + 0.10·F_robust + 0.05·Calibration + 0.10·F_novel
```

| Component | Weight | Definition | Implication for us |
|---|---|---|---|
| Task A macro-F1 | 40% | 6 classes, equal weight | Rare classes matter as much as common ones; do not ignore class 5 / class 3 |
| Task B macro-F1 | 15% | T1–T6, computed over **all** test rows | Six binary problems; legit rows are negatives; tune a threshold per technique |
| Task C macro-F1 | 10% | 6 action labels | Needs its own head; derive consistency from Task A |
| Domain shift | 10% | Task A macro-F1 on hidden subset | Grouped CV, avoid template memorisation |
| Robustness | 10% | Task A macro-F1 on corrupted-text subset | Normalisation + corruption augmentation + char n-grams |
| Calibration | 5% | 1 − Brier(confidence, correctness of Task A) | Confidence must be calibrated, never a default 0.99 |
| Novel scenarios | 10% | Task A macro-F1 on hidden subset | Semantic models, LLM-generated scenario coverage, minimal-pair data |

Roughly **30% of the score is Task A on hidden, shifted or corrupted data.** Hidden subsets are unmarked in `test.csv` and may overlap. The public leaderboard is 30% of rows and may drop components with no rows, so it is a weak guide. The private leaderboard (70%) decides rank. The 0.01 reproduction tolerance applies to the *private* score.

## 4. Problem definition

**Input.** `test.csv`: `id`, `text`. `text` is one message or up to 12 messages joined by `|||`, in chronological order, possibly with speaker markers. Links are defanged (`example[.]com`).

**Outputs (per row).**

| Task | Column | Type | Values |
|---|---|---|---|
| A | `label` | int | 0 Legit, 1 MFS/OTP, 2 Prize/lottery, 3 Fake job/investment, 4 Delivery phishing, 5 Promo spam |
| B | `techniques` | `\|`-joined codes | any of T1–T6, or `NONE` (never empty) |
| C | `action` | int | 0 none, 1 share info, 2 pay/transfer, 3 click link, 4 install/join/task, 5 contact sender |
| D | `confidence` | float | in [0, 1], probability that `label` is correct |

**Label priority (first match wins):** 1 → 2 → 4 → 3 → 5 → 0. Intent beats keywords.

**Hard legitimate cases that must stay class 0 (or 5 where marketing):** genuine OTP notices that say "do not share"; transaction confirmations; real delivery tracking; bill reminders; service notifications with links; genuine marketing and cashback (class 5); a friend *warning* about a scam (class 0).

## 5. Functional requirements

| ID | Requirement | Priority |
|---|---|---|
| FR-1 | Produce `submission.csv` with exactly the columns/order in `sample_submission.csv`, every test `id` exactly once | P0 |
| FR-2 | Task A classifier over the whole conversation (never the last message only) | P0 |
| FR-3 | Task B multi-label output with per-technique thresholds; `NONE` when no technique applies | P0 |
| FR-4 | Task C six-class action output, consistent with Task A | P0 |
| FR-5 | Task D calibrated confidence, fit on out-of-fold predictions | P0 |
| FR-6 | Light + skeleton text normalisation that survives obfuscation without destroying signal | P0 |
| FR-7 | Leakage-safe grouped validation, saved to disk | P0 |
| FR-8 | Programmatic corruption augmentation (label-preserving) | P0 |
| FR-9 | Frozen multilingual embedding model + small classifier (Model 2) | P1 |
| FR-10 | LLM-generated synthetic data incl. hard legitimate and minimal-pair examples (declared, programmatic) | P1 |
| FR-11 | Ensemble of Models 1 and 2 using OOF predictions | P1 |
| FR-12 | Pseudo-labeling of test text (programmatic, declared), adopted only if CV improves | P2 |
| FR-13 | Fine-tuned XLM-R / BanglaBERT | P2 (gated, see DESIGN §5) |
| FR-14 | `predict_df(text_df) -> submission_df` usable on new unseen text without manual changes (organisers may re-run on extra data) | P0 |
| FR-15 | `tools/validate_submission.py` and `tools/make_zip.py` enforce format and ZIP contents | P0 |

## 6. Non-functional requirements

| ID | Requirement |
|---|---|
| NFR-1 **Reproducibility** | Fixed seeds; notebook generated from `src/`; clean re-run reproduces private score within 0.01 |
| NFR-2 **Compliance** | Every external model, dataset and API declared; no hardcoded keys; no hand-labeled or hardcoded test predictions |
| NFR-3 **Leakage-free** | No test text or label copied into training; synthetic variants stay in the same fold as their source |
| NFR-4 **Runtime** | Full 500k inference finishes in a reasonable single Kaggle session; embeddings batched, length-sorted, cached |
| NFR-5 **Cost** | API spend capped with a competition-only key; key stays funded through verification |
| NFR-6 **Clarity** | Notebook is judged by humans: clear declaration section, validation story, error analysis |

## 7. Constraints and context

- **Time:** prelims 26 Nov 18:00 → 27 Nov 18:00. Final ZIP due 27 Nov 22:00.
- **Data:** about 86 labeled rows (confirm in EDA), 500,000 unlabeled test rows. Final class mix and hidden subsets undisclosed.
- **Team:** 2–3 undergraduates. Teammate owns inspection, validation and preprocessing. Nabila owns modelling, augmentation, calibration, submission and packaging.
- **Environment:** local Python venv (pandas, numpy, scipy, scikit-learn, lightgbm, sentence-transformers, torch, transformers, matplotlib, seaborn, jupyter). Final run on Kaggle or Colab with internet access.
- **Submissions:** 10 per day. Pick 2 finals manually; otherwise Kaggle picks the best two by *public* score.

## 8. Success criteria by milestone

| Milestone | Target time | Done when |
|---|---|---|
| **M0** foundations | Fri 00:30 | Scaffolding, contracts, validator, metrics exist; EDA first pass shared |
| **M1** first valid submission | Fri ~01:30 | Baseline TF-IDF+LR pipeline produces a Kaggle-accepted file |
| **M2** multi-model OOF | Fri ~12:00 | Models 1 and 2 plus corruption augmentation have grouped OOF probabilities and logged CV |
| **M3** candidate finals | Fri ~15:00 | Ensemble + calibration chosen on CV; two candidate finals trained on all data |
| **M4** freeze + submit | Fri ~17:30 | Two final Kaggle submissions selected manually; code frozen |
| **M5** package | Fri ~21:30 | Clean-run notebook reproduces the submission; ZIP built and validated; 30-min buffer kept |

Quality gates (hypotheses, to be tightened after the first CV):
- Every model change is accepted only if it improves the surrogate composite on repeated grouped CV and no slice (language, convo vs single, corrupted) drops by more than about 3 macro-F1 points.
- Clean-vs-corrupted macro-F1 gap on CV: aim for under about 5 points.
- Calibration: Brier on OOF must beat the constant-confidence baseline.

## 9. Risks

| ID | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | Overfit to 86 examples / template leakage inflates CV | High | High | Grouped folds, repeated CV, leave-cluster-out slice |
| R2 | Keyword over-reliance fails hard-legit cases | High | High | Minimal-pair synthetic data, hard-legit slice in CV |
| R3 | Corruption robustness weak | Med | High | Skeleton view, char n-grams, corruption augmentation with held-out operators for evaluation |
| R4 | Class-prior shift in test | Med | Med | Class-balanced weights, optional prior-shift correction, avoid class-frequency priors in rules |
| R5 | LLM-synthetic data distribution mismatch | Med | Med | Low sample weight, round-trip label check, validate on real rows only |
| R6 | Pseudo-labeling confirmation bias | Med | Med | Adopt only if CV on real data improves; cap per class |
| R7 | Notebook does not reproduce (non-deterministic LLM, time-out, missing key) | Med | **Critical** | Freeze generated data inside the notebook, seeds, caching, early clean-run rehearsal |
| R8 | Format / ZIP mistakes | Med | **Critical** | Validators, rule discrepancies resolved (see `MEMORY.md` §3) |
| R9 | Fatigue errors overnight | High | Med | Rest block, small PRs, validator on every file |
| R10 | Kaggle quota/limits | Low | Med | Submission budget plan in `TASKS.md` |

## 10. Open questions (resolve from EDA / organisers)

1. What label, techniques and action do **noise-only** records get in `sample.csv`?
2. Is class 5 (promo spam) always `NONE` / action 0 (or 3)? Check the label cross-tab.
3. Single message vs conversation ratio; how are speaker markers formatted?
4. Class distribution in the sample, and how many near-duplicate templates exist?
5. Which submission columns does Kaggle expect: the two-column rulebook format or the five-column format? See `MEMORY.md` §3.
6. ZIP file name: rulebook v1.3 vs problem statement conflict. Check the guidance email.
7. May LLM-generated data be embedded as a frozen cell in the notebook (with the generation code kept)? Ask in the official channel or check the guidance email.
