# Result Discussion (2026-03-07)

분석 대상:
- `result.csv` (현재 60행: `3 settings x 5 seeds x 4 methods`)
- 시각화 기준: `time_bar.png`의 `median_iqr`

## 1) 시간 스케일이 예측과 일치하는지

프로토콜의 이론적 기대:
- `abs_pearson`, `min_dbi`, `mi`는 대체로 `N`(row), `P`(feature)에 따라 증가
- `shap`은 가장 느리고 변동성이 큼

실측(각 shape, seed 5개 기준 `median [Q1, Q3]`, 단위: sec):

| shape | abs_pearson | min_dbi | mi | shap |
|---|---:|---:|---:|---:|
| 100 x 300 | 0.042 [0.040, 0.044] | 0.133 [0.126, 0.137] | 0.306 [0.301, 0.336] | 0.605 [0.587, 0.634] |
| 1000 x 300 | 0.040 [0.038, 0.042] | 0.189 [0.188, 0.195] | 1.283 [1.252, 1.323] | 40.990 [36.763, 49.968] |
| 1000 x 3000 | 0.595 [0.586, 0.598] | 2.795 [2.743, 2.817] | 20.806 [20.711, 20.863] | 72.578 [67.658, 73.413] |

해석:
- 방법 간 상대 속도 순서는 전 shape에서 `abs_pearson < min_dbi < mi < shap`로 유지되어 예측과 일치.
- 다만 스케일 증가율은 방법별로 비대칭:
  - Row 증가(100 -> 1000, feature=300): `abs_pearson` 0.95x, `min_dbi` 1.42x, `mi` 4.19x, `shap` 67.75x
  - Feature 증가(300 -> 3000, row=1000): `abs_pearson` 14.88x, `min_dbi` 14.79x, `mi` 16.22x, `shap` 1.77x
- 결론적으로 "순위 예측"은 맞지만, "row/feature 스케일 계수"는 구현/시스템 영향(특히 SHAP)으로 단순 선형 기대와는 차이가 있음.

## 2) relevant feature 추출 능력 비교

정답 기준: `selected_feature == the_ground_truth`

전체 정확도(15 datasets 기준):

| method | correct / total | hit rate |
|---|---:|---:|
| abs_pearson | 15 / 15 | 1.00 |
| min_dbi | 12 / 15 | 0.80 |
| mi | 15 / 15 | 1.00 |
| shap | 15 / 15 | 1.00 |

관찰:
- `abs_pearson`, `mi`, `shap`은 이번 실험에서 정답 피처를 항상 회수.
- `min_dbi`는 3회 miss:
  - `dataset_id=4` (100x300, seed=3)
  - `dataset_id=9` (1000x300, seed=3)
  - `dataset_id=11` (1000x3000, seed=0)
- 이는 DBI가 "target 상관"보다 "config 분리도"를 최적화하기 때문으로 해석 가능.

## 3) 5,000 x 20,000에서 예상 CPU time

추정 방법:
- 실측 median 기반으로 방법별 power-law 계수 추정
  - `time ~= k * rows^a * features^b`
- 기준점: 현재 관측된 3 shape (`100x300`, `1000x300`, `1000x3000`)

예상 시간(대략):

| method | predicted time (sec) | approx |
|---|---:|---:|
| abs_pearson | 5.32 | ~0.09 min |
| min_dbi | 32.88 | ~0.55 min |
| mi | 562.60 | ~9.38 min |
| shap | 2213.09 | ~36.9 min |

주의:
- SHAP는 변동성이 커서 불확실성이 가장 큼. (mean 기반 추정 시 ~7398 sec, 약 123 min까지 증가)
- 따라서 운영 계획은 `median_iqr` 기준으로 보고, SHAP는 여유 시간을 크게 잡는 것이 안전.
