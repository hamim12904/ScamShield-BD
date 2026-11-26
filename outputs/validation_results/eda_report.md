# EDA report — ScamShield BD (Teammate A, A1)

Date: 27 Nov 2026, ~02:45 BST. Seed `config.SEED = 42`. Sources: `data/sample.csv` (labeled, 86 rows), `data/test.csv` (unlabeled, 500,000 rows), `data/sample_submission.csv`.

Provenance: `python -m src.text_analysis` (sections 1–6), `python -m src.training.cross_validation`, plus direct computations with `build_views`, `parse_techniques`, `language_bucket` and `preprocess`. `text_analysis` section 7 aborted on a Windows cp1252 console error when printing a Cyrillic character; sections 1–6 completed and the remaining items were computed directly with UTF-8 output. Test text is used only for aggregate, unsupervised statistics — no individual test rows appear in this report.

## 1. Labeled row count and class counts

86 labeled rows.

| Label | Class | n | share |
|---|---|---|---|
| 0 | Legitimate | 32 | 37.2% |
| 1 | MFS/OTP scam | 12 | 14.0% |
| 2 | Prize/lottery scam | 10 | 11.6% |
| 3 | Fake job/investment | 11 | 12.8% |
| 4 | Delivery/parcel phishing | 9 | 10.5% |
| 5 | Promotional spam | 12 | 14.0% |

Actions: `0` 44, `1` 12, `2` 12, `3` 8, `4` 7, `5` 3.
Technique frequency (`NONE` 44): T1 21, T5 18, T3 13, T4 12, T6 11, T2 11.

## 2. Language mix

| Bucket | sample (86) | test (500,000) |
|---|---|---|
| english | 0.686 | 0.561 |
| banglish | 0.244 | 0.329 |
| mixed_bn_en | 0.000 | 0.104 |
| bangla_script | 0.070 | 0.006 |

Sample label × language: class 0 = 24 en / 7 banglish / 1 bangla; class 1 = 2 en / 9 banglish / 1 bangla; class 2 = 7/2/1; class 3 = 9/1/1; class 4 = 6/2/1; class 5 = 11/0/1.

Key shift: test has ~10.4% mixed Bangla+English and no sample row does; test Bangla-script share is lower (0.6% vs 7.0%). Banglish is the main scam language in the sample (9/12 class-1 rows).

## 3. Single messages vs conversations

Sample: 78 single, 8 conversations; `n_msgs` = {1: 78, 2: 1, 3: 5, 5: 2}, max 5.
Test: 392,752 single (78.6%), 107,248 conversations (21.5%); `n_msgs` = 1: 392,752; 2: 22,279; 3: 30,478; 4: 8,365; 5: 12,102; 6: 11,394; 7: 5,072; 8: 4,187; 9: 3,954; 10: 4,042; 11: 2,337; 12: 3,038. Max 12, as specified.

Sample label × conversation: each class has exactly 1 conversation (class 0 has 3).

## 4. Speaker markers

Raw exploratory regex `^\s*[A-Za-z]{1,10}\s*:` (`text_analysis.SPEAKER_RE`) matches 16 sample rows and 169,363 test rows, but over-matches defanged URL prefixes (`hxxps:`, `https:`) and ordinary words (`Task:`, `Paribahan:`, `Helpline:`).

The preprocessing whitelist (`SPEAKER_RE` + `SPEAKER_CANON`) matches turn-initial labels `S`, `R`, `A`, `B`, `SENDER`, `YOU`, `CALLER`, `RECEIVER` (optionally bracketed), e.g. `S: … ||| R: …`. Canonical counts: sample 8 rows (S 5, R 2, A 1); test 67,270 rows (S 25,841; A 16,797; SENDER 15,975; CALLER 8,460; B 71; YOU 63; R 63). All 8 sample conversations carry canonical markers.

## 5. Duplicates and template clusters

- Sample: 0 exact duplicates, 0 duplicates after normalisation; 0 char-TF-IDF near-duplicate pairs at cosine ≥ 0.85.
- Test: 0 exact duplicates, 139,741 normalized duplicates (27.95%), 360,259 unique normalized templates.
- 5 of 86 sample normalized texts also appear in test — a leakage risk, handled by grouped folds.
- Saved folds (`outputs/validation_results/folds.csv`): 86 rows, 85 groups (84 singletons + 1 pair), 3 clusters. Cluster × label: cluster 0 = {0:17, 1:6, 2:5, 3:5, 4:4, 5:5}; cluster 1 = {0:12, 1:3, 2:3, 3:4, 4:3, 5:5}; cluster 2 = {0:3, 1:3, 2:2, 3:2, 4:2, 5:2}. All classes present in every validation fold; leakage and leave-cluster-out checks pass.

## 6. Cross-tabs and rules holding at ≥95%

Label × action:

| label | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| 0 | 32 | 0 | 0 | 0 | 0 | 0 |
| 1 | 0 | 11 | 1 | 0 | 0 | 0 |
| 2 | 0 | 1 | 2 | 5 | 1 | 1 |
| 3 | 0 | 0 | 4 | 0 | 6 | 1 |
| 4 | 0 | 0 | 5 | 3 | 0 | 1 |
| 5 | 12 | 0 | 0 | 0 | 0 | 0 |

Label × number of techniques:

| label | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| 0 | 32 | 0 | 0 | 0 |
| 1 | 0 | 0 | 5 | 7 |
| 2 | 0 | 1 | 7 | 2 |
| 3 | 0 | 5 | 4 | 2 |
| 4 | 0 | 4 | 4 | 1 |
| 5 | 12 | 0 | 0 | 0 |

Rules at ≥95% (all 100% on these 86 rows):

| # | Rule | Count |
|---|---|---|
| 1 | label 0 → action 0 | 32/32 |
| 2 | label 5 → action 0 | 12/12 |
| 3 | label 0 → techniques NONE | 32/32 |
| 4 | label 5 → techniques NONE | 12/12 |
| 5 | label 2 → technique T4 present | 10/10 |
| 6 | label 3 → technique T6 present | 11/11 |
| 7 | technique T2 present → label 1 | 11/11 |
| 8 | technique T6 present → label 3 | 11/11 |

Not at 95%: label 1 → action 1 (11/12, 91.7%); label 1 → T2 (11/12, 91.7%); label 1 → T1 (8/12, 66.7%); label 4 → T1 (6/9, 66.7%) or T3 (5/9, 55.6%). Exact technique sets are diffuse for classes 1–4 (largest: `T1|T2|T5` 6/12 for class 1; `T6` 5/11 for class 3). Do not turn these into hard rules.

Handoff to B (F-04): rules 1–8 are the only candidates that clear the 95% bar, and the sample is small enough that they should be validated on grouped CV before adoption.

## 7. Noise-only rows (F-05)

`build_views(sample).noise_only` flags exactly 2 rows:

| id | text | label | techniques | action | noise_ratio |
|---|---|---|---|---|---|
| 10058 | `lkjsdfoiu xcvbnm` | 0 | NONE | 0 | 1.0 |
| 10081 | `zzqpwoeiru` | 0 | NONE | 0 | 1.0 |

Both are label 0 / NONE / action 0, so the current forced prediction is consistent with the labeled evidence, but n=2 is too small to treat as verified. Mark the noise-only rule as an assumption in the notebook (F-05), as previously planned.

## 8. Class 5 (promotional spam)

12/12 have techniques `NONE`; 12/12 have action 0. No other technique or action occurs in this class.

## 9. Test statistics (unsupervised)

Full-test counts from `text_analysis`: Cyrillic/Greek lookalikes 8.508%, zero-width characters 6.717%, URLs 17.357%, emojis 16.798%; mean text length 115.5 chars (sample 85.6).

Deterministic 50,000-row subsample (`test.iloc[::10]`) via `preprocess`:

- `noise_ratio` histogram (bins `[0, .01, .1, .25, .5, .75, .99, 1]`): 43,968 / 4,496 / 1,427 / 16 / 5 / 0 / 88. 87.9% of rows have no removed tokens; 0.176% (88/50,000) are `noise_only`.
- Language: english 56.09%, banglish 33.05%, mixed_bn_en 10.28%, bangla_script 0.57% (matches the full-test figures).
- Lookalikes 8.504%, zero-width 6.732%.

Noise is therefore rare but non-zero, and obfuscation flags are common enough that the robustness slice matters.

## 10. Hard legitimate cases (label 0)

Regex hits on labeled legitimate rows (ids only; counts overlap):

- OTP / code / PIN: 7 rows — 10016, 10017, 10036, 10052, 10075, 10085, 10086.
- Link / defanged domain: 2 rows — 10001, 10031.
- Cashback / discount / offer / promo: 0 rows.
- Parcel / delivery / tracking / order: 4 rows — 10001, 10004, 10021, 10048.
- Urgency words (`urgent`, `immediately`, `expire`, `today`, `now`): 2 rows — 10004, 10063.

Example of the hard pattern from labeled data: id 10085 is a multi-turn legitimate conversation about resetting a bKash PIN that explicitly says never to share it — keyword-only models would misclassify it. The sample contains no label-0 cashback row, while class 5 contains ordinary cashback promos, so that specific hard-legit family is unmeasured locally and should be watched in validation.
