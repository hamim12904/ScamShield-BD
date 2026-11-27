# Test-set class-0 skew diagnosis

Aggregate counts and percentages only. All test labels below are model predictions; no test ground truth exists. No individual test text or id is included.

## 1. Model 2 coverage (H1)

- M2 probability rows: 500,000
- Rows equal to the padded tail vector: **1 (0.00%)**
- Rows with a distinct M2 vector: 499,999 (100.00%)
- ood_sim values equal to the tail value: 1 (0.00%); distinct values in padded tail: 1
- Constant ood_sim tail: yes
- M2 covers all test rows: no

## 2. Predicted histograms vs labeled sample

| Class | Sample count | Sample % | Predicted count | Predicted % |
|---|---:|---:|---:|---:|
| 0 | 32 | 37.21% | 183,922 | 36.78% |
| 1 | 12 | 13.95% | 84,569 | 16.91% |
| 2 | 10 | 11.63% | 47,816 | 9.56% |
| 3 | 11 | 12.79% | 41,565 | 8.31% |
| 4 | 9 | 10.47% | 64,676 | 12.94% |
| 5 | 12 | 13.95% | 77,452 | 15.49% |

Predicted actions:

| Action | Count | % |
|---|---:|---:|
| 0 | 261,374 | 52.27% |
| 1 | 83,894 | 16.78% |
| 2 | 109,060 | 21.81% |
| 3 | 24,338 | 4.87% |
| 4 | 14,631 | 2.93% |
| 5 | 6,703 | 1.34% |

Predicted technique counts (a row may carry several codes):

| Technique | Count | % of rows |
|---|---:|---:|
| NONE | 261,374 | 52.27% |
| T1 | 145,290 | 29.06% |
| T2 | 59,770 | 11.95% |
| T3 | 95,007 | 19.00% |
| T4 | 55,104 | 11.02% |
| T5 | 87,888 | 17.58% |
| T6 | 37,645 | 7.53% |

## 3. Predicted-0 cue rates vs predicted 1-4 (H2/H3)

| Signal | All rows | Predicted 0 | Predicted 1-4 |
|---|---:|---:|---:|
| noise_only | 845 (0.17%) | 832 (0.45%) | 0 (0.00%) |
| has_url | 87,361 (17.47%) | 17,116 (9.31%) | 59,830 (25.07%) |
| otp_like | 27,861 (5.57%) | 27,179 (14.78%) | 588 (0.25%) |
| money | 264,622 (52.92%) | 92,789 (50.45%) | 153,927 (64.51%) |
| cred | 124,821 (24.96%) | 50,469 (27.44%) | 74,341 (31.15%) |
| urgency | 51,439 (10.29%) | 11,162 (6.07%) | 39,023 (16.35%) |
| reward | 72,768 (14.55%) | 9,202 (5.00%) | 41,757 (17.50%) |
| conversation | 107,248 (21.45%) | 35,900 (19.52%) | 54,188 (22.71%) |

## 4. Confidence by predicted class (H5)

| Predicted class | n | Mean | q10 | q50 | q90 |
|---|---:|---:|---:|---:|---:|
| all | 500,000 | 0.8489 | 0.8488 | 0.8488 | 0.8488 |
| 0 | 183,922 | 0.8491 | 0.8488 | 0.8488 | 0.8488 |
| 1 | 84,569 | 0.8488 | 0.8488 | 0.8488 | 0.8488 |
| 2 | 47,816 | 0.8488 | 0.8488 | 0.8488 | 0.8488 |
| 3 | 41,565 | 0.8488 | 0.8488 | 0.8488 | 0.8488 |
| 4 | 64,676 | 0.8488 | 0.8488 | 0.8488 | 0.8488 |
| 5 | 77,452 | 0.8488 | 0.8488 | 0.8488 | 0.8488 |

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
| english | 280,544 | 120,973 | 22,214 | 26,738 | 26,023 | 29,550 | 55,046 |
| banglish | 164,619 | 40,285 | 55,880 | 14,701 | 11,527 | 24,622 | 17,604 |
| bangla_script | 2,950 | 1,219 | 214 | 329 | 823 | 361 | 4 |
| mixed_bn_en | 51,887 | 21,445 | 6,261 | 6,048 | 3,192 | 10,143 | 4,798 |
| no_letters | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## 6. M1 vs M2 argmax disagreement

Cannot compute: M1 test probabilities are not saved.

## Interpretation (hypotheses only; no test labels exist)

- **H1 M2 probabilities mostly padded:** not supported. 0.00% of rows match the tail vector.
- **H2 Many predicted-0 rows are noise-only:** not supported. Noise-only share is 0.45% among predicted 0 vs 0.17% overall.
- **H3 Predicted-0 rows contain scam cues:** cannot tell. Higher for predicted 0: otp_like; lower for predicted 0: has_url, money, cred, urgency, reward. Cue presence is not a test label.
- **H4 Language shift hurts the model:** supported for the shift itself (largest bucket difference 12.50 points); effect on errors cannot tell without labels.
- **H5 Low confidence concentrates in class 0:** not supported. Predicted-0 mean confidence is 0.8491 vs 0.8489 overall.
