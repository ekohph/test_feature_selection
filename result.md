# Result Discussion (2026-03-07, updated)

분석 데이터:
- `result.csv` (현재 120행: `3 settings x 5 seeds x 2 relationship modes x 4 methods`)
- relationship mode:
  - `default_nonlinear`
  - `non_monotonic_strong`

## 1) 시간 스케일이 예측과 일치하는지

`time_bar.png`(mode 통합 집계, `median_iqr`) 기준 중앙값:

| shape | abs_pearson | min_dbi | mi | shap |
|---|---:|---:|---:|---:|
| 100 x 300 | 0.0200 | 0.0690 | 0.1565 | 0.4920 |
| 1000 x 300 | 0.0240 | 0.1090 | 0.8000 | 48.0915 |
| 1000 x 3000 | 0.3605 | 1.6355 | 12.1720 | 77.6535 |

해석:
- 시간 순서는 일관적으로 `abs_pearson < min_dbi < mi < shap`.
- `N`(rows), `P`(features)가 커질수록 계산 시간이 증가.
- `shap`은 분산(IQR)이 상대적으로 커서, 동일 shape에서도 런타임 변동이 큰 편.

## 2) 비선형성 추가 후 정답률 비교

정답 기준: `selected_feature == the_ground_truth`

mode별 정답률:

| relationship_mode | abs_pearson | min_dbi | mi | shap |
|---|---:|---:|---:|---:|
| default_nonlinear | 1.000 | 0.800 | 1.000 | 1.000 |
| non_monotonic_strong | 0.133 | 0.933 | 1.000 | 0.800 |

전체(두 mode 통합) 정답률:

| method | correct / total | hit rate |
|---|---:|---:|
| abs_pearson | 17 / 30 | 0.567 |
| min_dbi | 26 / 30 | 0.867 |
| mi | 30 / 30 | 1.000 |
| shap | 27 / 30 | 0.900 |

핵심 관찰:
- `non_monotonic_strong`에서 `abs_pearson` 성능이 크게 하락(1.000 -> 0.133).
- `mi`는 두 mode 모두 1.000으로 안정적.
- `shap`은 비선형 모드에서 하락(1.000 -> 0.800)하지만 correlation 기반보다는 강건.

## 3) 계산 시간이 mode에 거의 영향받지 않는 이유 (+ 그림)

대부분 방법에서 시간 복잡도를 지배하는 것은 관계식 모양보다 `N`, `P`, 반복 횟수다.
- `abs_pearson`: 후보 feature별 상관계수 반복 계산
- `min_dbi`: 후보 feature별 DBI 계산 반복
- `mi`: 후보 feature별 MI 추정 반복

그래서 관계식을 바꿔도 위 3개는 동일 shape에서 시간이 거의 비슷하다.  
반면 `shap`은 내부 모델 학습/설명 단계가 포함되어, 타깃 함수 난이도 변화에 더 민감하다.

shape별 `non_monotonic_strong / default_nonlinear` 시간비(중앙값):

| shape | abs_pearson | min_dbi | mi | shap |
|---|---:|---:|---:|---:|
| 100 x 300 | 1.000 | 1.060 | 1.097 | 1.263 |
| 1000 x 300 | 1.000 | 0.964 | 1.022 | 2.461 |
| 1000 x 3000 | 1.031 | 1.033 | 0.958 | 2.084 |

그림(mode별 시간 민감도 ratio):

![Mode Time Ratio](visuals/time_mode_ratio.png)

## 4) 5,000 x 20,000 예상 CPU time (업데이트)

관측된 3개 shape에서 power-law(`time ~= k * rows^a * features^b`)로 추정.

mode 통합 기준 예상치:

| method | predicted time (sec) | approx |
|---|---:|---:|
| abs_pearson | 3.82 | ~0.06 min |
| min_dbi | 20.97 | ~0.35 min |
| mi | 358.69 | ~5.98 min |
| shap | 2835.52 | ~47.3 min |

mode별 참고(특히 SHAP):
- `default_nonlinear`: `shap ~1636 sec` (~27.3 min)
- `non_monotonic_strong`: `shap ~4738 sec` (~79.0 min)

결론:
- 비선형성 강화 후에도 `abs_pearson/min_dbi/mi` 시간은 mode 영향이 제한적.
- `shap`은 mode 민감도가 상대적으로 커서, 대규모 실험에서는 별도 시간 예산을 두는 것이 안전하다.
