# DESIGN — ScamShield BD (modelling and data design)

> The technical decisions behind the pipeline. Structure and interfaces are in `ARCHITECTURE.md`. Everything here is a *plan with tunables*; results go to `MEMORY.md`.

---

## 1. Label semantics: the decision procedure

Labels follow **intent and behaviour, not keywords**. Priority when several categories apply (stop at the first match): **1 → 2 → 4 → 3 → 5 → 0**.

| Class | Fires when… | Typical near-miss that must NOT fire |
|---|---|---|
| 1 MFS/OTP | asks for PIN/OTP/password/NID/card details; claims a mistaken transfer and asks for money back; support impersonation to gain access | Genuine OTP delivery that says "do not share" → 0 |
| 2 Prize/lottery | unearned prize/gift/cashback/reward needing payment, click, details | Genuine cashback promo with a code → 5 |
| 4 Delivery phishing | unexpected parcel/customs/fee/address/tracking-link request | Real tracking update "out for delivery" → 0 |
| 3 Fake job/investment | unrealistic income, upfront recruitment fee, task earnings, money doubling, crypto | Real job posting without fee/lure → 0 |
| 5 Promo spam | marketing/discount/app promo, spam-like but not deceptive | Fraudulent claim → move up the priority list |
| 0 Legit | transactional, informational, personal, service notices | A friend warning about a scam → 0 |

**Why it matters for modelling:** the same surface words (OTP, link, cashback, parcel, urgent) appear across classes. The model must learn *requests* and *claims*, so training data must include **minimal pairs** (legit vs scam variants of one template) and hard negatives (see §7).

## 2. Text representation: three views

Never overwrite the original. `preprocessing.build_views` adds:

| View | Purpose | Operations |
|---|---|---|
| `text_raw` | structural cues (caps ratio, punctuation, emoji, spacing) | untouched |
| `text_norm` (light, safe) | embeddings, word TF-IDF | NFKC; remove U+200B, U+2060, U+FEFF; keep `\|\|\|` and speaker markers; collapse whitespace; Unicode lookalikes → Latin **only inside Latin-context tokens**; Bangla digits → ASCII; re-fuse defanged links (`hxxp`, `[.]`, `(dot)`) into URL form; keep emojis, numbers, casing |
| `text_skel` (aggressive, for char models) | robustness to obfuscation | lowercase; de-leetspeak (`0→o, 1→i/l, 3→e, 4→a, 5→s, @→a, $→s`) **only in tokens that mix letters and those characters**; collapse 3+ repeats to 2; re-join spaced-out letters (`v e r i f y`, `v.e.r.i.f.y`); remove **all** zero-width characters; replace noise tokens by `<NOISE>`; digit runs → shape tokens (`<N6>` for a 6-digit code) while keeping small amounts' magnitude (`<N3>` + original in raw) |

**Care points**
- U+200C/U+200D are legitimate in Bangla conjuncts. Strip them in `text_skel` only; in `text_norm` keep them when adjacent to Bangla characters.
- Don't transliterate Bangla ↔ Banglish; it's lossy. Char n-grams and a multilingual encoder cover both scripts.
- Preserve `|||`. Segment by it for structure features, never delete it.
- Emojis stay (they are signal: 🎁💰). Numbers stay (amounts, 6-digit codes, "within 1 hour").
- URLs: keep the host/TLD tokens as features (`url_tld_xyz`), plus `has_url`. Defanged form is the norm in this dataset.
- Idempotence: `f(f(x)) == f(x)` (tested).

**Structure features** (scaled, stacked with TF-IDF): `n_msgs`, mean/max message length, `has_url`, `n_urls`, digit ratio, uppercase ratio, `!`/`?` counts, emoji count, Bangla-script ratio, Latin ratio, `noise_ratio`, `has_otp_like_number` (6-digit), `asks_question`, `first_msg_len`.

**Conversation handling.** Always classify the whole text. Feature blocks: (1) full conversation, (2) first message, (3) last message, (4) concatenated middle. Delayed requests are caught by block 1 and 4. Never train on the last message alone.

**Noise-only / empty records.** `noise_ratio ≈ 1` or no linguistic tokens → route to a dedicated rule/small classifier; what label to emit comes from `sample.csv` (open question; hypothesis: class 0 / `NONE` / action 0 with low confidence). Confidence is lowered in either case (§9).

## 3. Model 1: TF-IDF + linear models (baseline, ships first)

- **Features:** `word` TF-IDF 1–2 grams on `text_norm` (token pattern covers Bangla Unicode); `char_wb` 2–5 grams on `text_skel`; optionally `char` 1–4 on `text_raw`; plus structure features; plus first/last-message TF-IDF blocks. `sublinear_tf=True`, `min_df=1` (tiny data), `max_features` capped for speed.
- **Fit vocabulary/IDF on train + test text** (unsupervised; helps unseen brands and n-grams; declared in the notebook).
- **Classifiers:** `LogisticRegression` (multinomial, `class_weight='balanced'`, `C` searched in {0.5, 1, 2, 5, 10, 20}) as the primary; `LinearSVC` as a comparison. For SVM probabilities use softmax over scaled decision values with a temperature fitted on OOF.
- **Why:** fast, inspectable, strong on small data and obfuscation (char n-grams), and gives a benchmark before any transformer.

## 4. Model 2: frozen multilingual embeddings + small classifier

- **Encoder candidates (verify availability and licence; declare):** `paraphrase-multilingual-MiniLM-L12-v2` (fast baseline); `intfloat/multilingual-e5-base` (use the `query: ` prefix); `sentence-transformers/LaBSE`; `BAAI/bge-m3` (strongest but heavy).
- **Inputs:** `text_norm`, truncated to `MAX_LEN` (128–256 tokens); optionally add a separate embedding of the *last message* and concatenate.
- **Classifier:** `LogisticRegression` on L2-normalised embeddings, optionally a small MLP; heads for A, B, C.
- **Why:** captures semantic similarity across languages and phrasing, helping unseen scenarios. Expect it to be weaker on heavy Banglish obfuscation, so it complements Model 1 in the ensemble.
- **Efficiency:** sort by length, dynamic padding, fp16 on GPU, batch 256–512, cache `.npy`. Time 10k rows first, extrapolate to 500k, and switch to a smaller encoder if the estimate is too long.

## 5. Model 3 (optional): fine-tuned transformer

Fine-tuning XLM-R or BanglaBERT on 86 rows is high-variance and may lose to TF-IDF. **Gate:** attempt only if (a) a GPU is available, (b) at least ~2–3k cleaned synthetic + augmented examples exist, (c) Models 1–2 and calibration are done and (d) there is ≥ 3 hours left. Evaluate on real-only validation. Inference on 500k rows must fit the time budget. If it does not clearly beat the ensemble on grouped CV *and* the corrupted slice, drop it.

## 6. Task heads and consistency

| Task | Head | Notes |
|---|---|---|
| A | multinomial LR (per model) → ensemble probs | primary; class-balanced weights |
| B | six independent binary LRs (one-vs-rest), probability + **per-technique threshold** tuned on OOF | macro-F1 over rows: legit rows are negatives; thresholds usually < 0.5 for rare techniques |
| C | multinomial LR | 6 actions; may use A's OOF probs as extra features |
| D | calibration model (§9) | uses A's ensemble output |

**Consistency post-processing (`postprocess.py`)** is learned from data, not assumed. Build the cross-tab `label × techniques × action` from `sample.csv` (+ verified synthetic). Enforce a rule only if it holds in ≥ ~95% of real rows:

| Hypothesis (verify!) | Rule if confirmed |
|---|---|
| label ∈ {0, 5} ⇒ techniques = `NONE` | zero out Task B |
| label = 0 ⇒ action = 0 | force action 0 |
| label ∈ {1,2,3,4} ⇒ at least one technique | if none above threshold, emit argmax technique |
| label 1 ⇒ T2 likely; 2 ⇒ T4; 3 ⇒ T6; 4 ⇒ T1/T3 | use as soft priors (feature or probability reweighting), not hard rules |

## 7. Data expansion (how we beat the 86-row limit)

All expansion is **programmatic, seeded, declared**, and every synthetic row inherits its source's `group_id`.

### 7.1 Programmatic corruption (P0)
Operators (set A for training, set B held out for evaluation):
- A: character substitution (leet/lookalikes: `Verify your OTP → Ver1fy your 0TP`), repetition, random insertion/deletion, spacing changes, random capitalisation, `!!!`/emoji suffixes, zero-width insertion.
- B (evaluation only): different homoglyph tables, word-boundary removal, transliteration-style respellings, embedded noise strings (`wkrjfejejrn`), heavy mixed corruption.
Levels: light/medium/heavy; 2–5 variants per row. **Label-preserving constraint:** never corrupt so heavily that no decisive cue survives; keep `|||`.

### 7.2 Hand-written extras (P1)
A small, curated set in `data/extra/handwritten.py` covering the hard-legit families (real OTP + warning; transaction confirmations; delivery tracking; bill reminders; genuine promo/cashback; a friend warning about scams). Written without looking at test rows. Embedded in the notebook.

### 7.3 LLM-generated data (P1, optional but high-value)
- **Scenario matrix:** 6 classes × {English, Bangla script, Banglish, mixed} × {single, 2–4 messages, 5–12 messages with delayed intent} × scenario seeds (MFS agent/support, mistaken transfer, prize, job/task/crypto, parcel/customs/address, promo; plus hard-legit seeds) × brand/operator lists (several MFS, couriers, telcos, e-commerce).
- **Minimal pairs:** for every scam template, also generate a legitimate sibling that shares vocabulary (e.g. real OTP vs OTP-harvesting). This directly trains against keyword reliance.
- **Prompting:** include the label definitions and the **priority order**; request strict JSON `{messages[], label, techniques[], action, language}`; vary writing styles; forbid real phone numbers/PII; links defanged.
- **Quality filter:** (1) schema check; (2) dedup by normalised hash + char-n-gram cosine > 0.9; (3) **round-trip label check**: an independent labeling call (or the current model) must agree with the intended label, else drop; (4) drop anything near-identical to a test row (guards against accidental copying).
- **Weighting:** real = 1.0; corruption variants ≈ 0.5–1.0; LLM rows ≈ 0.3–0.5; tune by real-only CV.
- **Reproducibility:** API output is non-deterministic. Save the filtered set as `data/extra/synth_v1.jsonl`, embed it **frozen** in the notebook, keep the generator behind `REGENERATE_SYNTH`. Declare the provider/model, purpose and cost. Keep the key funded through verification. Confirm with organisers that a frozen embedded copy is acceptable.

### 7.4 Pseudo-labeling and domain adaptation (P2)
- Only programmatic, declared, using model predictions only.
- Take test rows where Model 1 and Model 2 agree and ensemble probability ≥ τ (start 0.9), class-capped, excluding noise-only rows. Up to 2 rounds.
- **Adopt only if** real-only grouped CV (with pseudo-labeled test rows added to training folds only) improves F_A and the corrupted slice. Watch for confirmation bias and class-prior shift.
- Unsupervised domain adaptation that is always on: fit TF-IDF vocabulary/IDF on train + test text.

## 8. Ensembling
- Weighted average of probabilities (or log-probabilities) from Model 1 and Model 2 (and 3 if it exists). Choose 1–2 weights on OOF; with n this small prefer **equal weights or a single mixing parameter** over stacking.
- Calibrate **after** blending.

## 9. Confidence (Task D) and calibration
- Target: 1 if Task A prediction is correct, else 0, on OOF predictions. Score = 1 − Brier.
- Inputs: max ensemble probability, top-2 margin, entropy, model disagreement, `noise_ratio`, and an **OOD signal** (max cosine similarity to training embeddings; low → unfamiliar scenario → lower confidence).
- Model: tiny logistic calibrator (2–4 parameters) or temperature scaling. Avoid isotonic unless OOF n is large (≥ several hundred, e.g. with verified synthetic). Cross-fit so calibration is not evaluated on its own training rows.
- Clip to [0.03, 0.97]. For noise-only rows, shrink toward the base accuracy on such rows.
- Baseline to beat: constant confidence equal to OOF accuracy.

## 10. Evaluation design

**Metrics (`metrics.py`):** Task A/C macro-F1; Task B macro-F1 over T1–T6 (binary per technique, averaged); calibration = 1 − Brier; composite surrogate
`S_hat = 0.40·F_A + 0.15·F_B + 0.10·F_C + 0.10·F_domain* + 0.10·F_robust* + 0.05·Cal + 0.10·F_novel*` (starred = proxies below).

**Validation:** `StratifiedGroupKFold`, 5 folds × 3 repeats (teammate's `folds.json`); report mean and spread. A class with too few rows for stratification falls back to fewer folds or leave-group-out.

**Proxies for hidden subsets (weak, but better than nothing):**
- *Robustness:* macro-F1 on **corrupted copies of held-out folds** using operator set B.
- *Domain/novel:* **leave-one-cluster-out** (template/brand clusters held out entirely); and, if LLM data exists, train on real + generation A, test on a differently prompted generation B (real rows still the main yardstick).

**Slices reported every time:** language (English / Bangla / Banglish / mixed), single vs conversation, hard-legit subset, corrupted subset, per-class F1, class-1 vs 2 vs 4 confusions.

**Error analysis (`error_analysis.py`):** confusion matrices; list of legit-predicted-as-scam and scam-predicted-as-legit; per-technique precision/recall with chosen thresholds; robustness curve (F1 vs corruption level).

**Promotion gate:** see `RULES.md` C3.

## 11. Efficiency (500k rows)
- TF-IDF transform and linear prediction are cheap; chunk to bound memory.
- Embeddings dominate: length-sorted batches, fp16, cached; estimate before committing.
- `predict_df` processes in chunks and writes partial outputs so a crash doesn't lose everything.
- Inference timing (avg and P95 per record) is logged for possible technical verification.

## 12. Failure modes and mitigations

| Failure | Detection | Mitigation |
|---|---|---|
| Template memorisation | CV much higher than leave-cluster-out | Grouped folds; regularise (smaller `C`); more diverse synthetic data |
| Keyword shortcuts | Hard-legit slice F1 drops | Minimal pairs; hard negatives; review top features |
| Overconfident errors | Brier worse than constant baseline | Stronger calibration regularisation; OOD signal |
| Class-prior shift | Predicted class histogram on test far from sample | Balanced weights; optional EM prior-shift estimate (unsupervised) |
| Noise-only mispredicted | Check predictions on high-`noise_ratio` rows | Dedicated route, low confidence |
| LLM data poisoning | Round-trip disagreement, real-only CV drops | Lower weight or drop LLM data |
| Notebook non-reproducibility | Clean-run score differs > 0.01 | Seeds, frozen data, deterministic ops, early rehearsal |
