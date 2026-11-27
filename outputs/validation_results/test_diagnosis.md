# Test-set class-0 skew diagnosis

Aggregate counts and percentages only. All test labels below are model predictions; no test ground truth exists. No individual test text or id is included.

## 1. Model 2 coverage (H1)

- M2 probability rows: 500,000
- Rows equal to the padded tail vector: **499,000 (99.80%)**
- Rows with a distinct M2 vector: 1,000 (0.20%)
- ood_sim values equal to the tail value: 499,000 (99.80%); distinct values in padded tail: 1
- Constant ood_sim tail: yes
- M2 covers all test rows: no

## 2. Predicted histograms vs labeled sample

| Class | Sample count | Sample % | Predicted count | Predicted % |
|---|---:|---:|---:|---:|
| 0 | 32 | 37.21% | 367,409 | 73.48% |
| 1 | 12 | 13.95% | 35,355 | 7.07% |
| 2 | 10 | 11.63% | 22,768 | 4.55% |
| 3 | 11 | 12.79% | 19,503 | 3.90% |
| 4 | 9 | 10.47% | 26,426 | 5.29% |
| 5 | 12 | 13.95% | 28,539 | 5.71% |

Predicted actions:

| Action | Count | % |
|---|---:|---:|
| 0 | 395,948 | 79.19% |
| 1 | 41,487 | 8.30% |
| 2 | 47,519 | 9.50% |
| 3 | 8,592 | 1.72% |
| 4 | 6,417 | 1.28% |
| 5 | 37 | 0.01% |

Predicted technique counts (a row may carry several codes):

| Technique | Count | % of rows |
|---|---:|---:|
| NONE | 395,948 | 79.19% |
| T1 | 68,909 | 13.78% |
| T2 | 20,989 | 4.20% |
| T3 | 8,546 | 1.71% |
| T4 | 7,113 | 1.42% |
| T5 | 34,638 | 6.93% |
| T6 | 9,352 | 1.87% |

## 3. Predicted-0 cue rates vs predicted 1-4 (H2/H3)

| Signal | All rows | Predicted 0 | Predicted 1-4 |
|---|---:|---:|---:|
| noise_only | 845 (0.17%) | 845 (0.23%) | 0 (0.00%) |
| has_url | 87,361 (17.47%) | 59,496 (16.19%) | 25,297 (24.31%) |
| otp_like | 27,861 (5.57%) | 27,648 (7.53%) | 184 (0.18%) |
| money | 264,622 (52.92%) | 182,424 (49.65%) | 77,298 (74.29%) |
| cred | 124,821 (24.96%) | 95,252 (25.93%) | 29,567 (28.42%) |
| urgency | 51,439 (10.29%) | 28,068 (7.64%) | 22,355 (21.48%) |
| reward | 72,768 (14.55%) | 43,527 (11.85%) | 22,650 (21.77%) |
| conversation | 107,248 (21.45%) | 81,515 (22.19%) | 17,344 (16.67%) |

## 4. Confidence by predicted class (H5)

| Predicted class | n | Mean | q10 | q50 | q90 |
|---|---:|---:|---:|---:|---:|
| all | 500,000 | 0.5968 | 0.4254 | 0.5688 | 0.8203 |
| 0 | 367,409 | 0.6019 | 0.4321 | 0.5755 | 0.8204 |
| 1 | 35,355 | 0.5868 | 0.4067 | 0.5603 | 0.8283 |
| 2 | 22,768 | 0.5595 | 0.4028 | 0.5346 | 0.7623 |
| 3 | 19,503 | 0.5717 | 0.3962 | 0.5224 | 0.8376 |
| 4 | 26,426 | 0.5687 | 0.4222 | 0.5358 | 0.7962 |
| 5 | 28,539 | 0.6165 | 0.4348 | 0.5821 | 0.8654 |

## 5. Language mix (H4)

| Bucket | Sample count | Sample % | Test count | Test % |
|---|---:|---:|---:|---:|
| english | 59 | 68.60% | 280,544 | 56.11% |
| banglish | 21 | 24.42% | 164,619 | 32.92% |
| bangla_script | 6 | 6.98% | 2,950 | 0.59% |
| mixed_bn_en | 0 | 0.00% | 51,887 | 10.38% |
| no_letters | 0 | 0.00% | 0 | 0.00% |

Predicted class mix per test language bucket (counts only):

| Bucket | n | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| english | 280,544 | 220,877 | 2,243 | 12,382 | 11,460 | 12,730 | 20,852 |
| banglish | 164,619 | 104,316 | 31,171 | 7,007 | 6,073 | 9,368 | 6,684 |
| bangla_script | 2,950 | 2,297 | 144 | 63 | 446 | 0 | 0 |
| mixed_bn_en | 51,887 | 39,919 | 1,797 | 3,316 | 1,524 | 4,328 | 1,003 |
| no_letters | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## 6. M1 vs M2 argmax disagreement

Cannot compute: M1 test probabilities are not saved.

## Interpretation (hypotheses only; no test labels exist)

- **H1 M2 probabilities mostly padded:** supported. 99.80% of rows match the tail vector.
- **H2 Many predicted-0 rows are noise-only:** not supported. Noise-only share is 0.23% among predicted 0 vs 0.17% overall.
- **H3 Predicted-0 rows contain scam cues:** cannot tell. Higher for predicted 0: otp_like; lower for predicted 0: has_url, money, cred, urgency, reward. Cue presence is not a test label.
- **H4 Language shift hurts the model:** supported for the shift itself (largest bucket difference 12.50 points); effect on errors cannot tell without labels.
- **H5 Low confidence concentrates in class 0:** not supported. Predicted-0 mean confidence is 0.6019 vs 0.5968 overall.
