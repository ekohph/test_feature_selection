# Result Discussion (2026-03-07, updated)

분석 데이터:
- `result.csv` (현재 120행: `3 settings x 5 seeds x 2 relationship modes x 4 methods`)
- relationship mode:
  - `default_nonlinear`
  - `non_monotonic_strong`

실행 환경(시간 측정 기준):
- CPU: `Intel(R) Core(TM) Ultra 7 155H` (16 cores, 22 logical processors, max 3.8GHz)
- RAM: 총 `31.59 GiB` (측정 시점 가용 `17.11 GiB`, 사용률 약 `45.8%`)
- 전원 모드: `SAMSUNG MODE` (`powercfg /getactivescheme`)
- 동시작업수(벤치마크 프로세스): `1` (단일 실험 프로세스 기준, dataset/selector 병렬 실행 없음)

## 1) 시간 스케일이 예측과 일치하는지

`time_bar.png`(mode 통합 집계, `median_iqr`) 기준 중앙값:

| shape | abs_pearson | min_dbi | mi | shap |
|---|---:|---:|---:|---:|
| 100 x 300 | 0.0110 | 0.0590 | 0.1585 | 0.4680 |
| 1000 x 300 | 0.0165 | 0.1355 | 1.4675 | 94.5675 |
| 1000 x 3000 | 0.5450 | 3.5915 | 31.4365 | 231.1870 |

해석:
- 시간 순서는 일관적으로 `abs_pearson < min_dbi < mi < shap`.
- `N`(rows), `P`(features)가 커질수록 계산 시간이 증가.
- `shap`은 분산(IQR)이 상대적으로 커서, 동일 shape에서도 런타임 변동이 큰 편.

## 2) 비선형성 추가 후 정답률 비교

정답 기준: `selected_feature == the_ground_truth`

mode별 정답률:

| relationship_mode | abs_pearson | min_dbi | mi | shap |
|---|---:|---:|---:|---:|
| monotonic | 1.000 | 0.600 | 1.000 | 1.000 |
| non_monotonic | 0.067 | 0.933 | 1.000 | 0.867 |

전체(두 mode 통합) 정답률:

| method | correct / total | hit rate |
|---|---:|---:|
| abs_pearson | 16 / 30 | 0.533 |
| min_dbi | 23 / 30 | 0.767 |
| mi | 30 / 30 | 1.000 |
| shap | 28 / 30 | 0.933 |

방법론별 accuracy 요약(mode/overall):

| method | monotonic | non_monotonic | overall |
|---|---:|---:|---:|
| abs_pearson | 15 / 15 (1.000) | 1 / 15 (0.067) | 16 / 30 (0.533) |
| min_dbi | 9 / 15 (0.600) | 14 / 15 (0.933) | 23 / 30 (0.767) |
| mi | 15 / 15 (1.000) | 15 / 15 (1.000) | 30 / 30 (1.000) |
| shap | 15 / 15 (1.000) | 13 / 15 (0.867) | 28 / 30 (0.933) |

핵심 관찰:
- `non_monotonic_strong`에서 `abs_pearson` 성능이 크게 하락(1.000 -> 0.067).
- `mi`는 두 mode 모두 1.000으로 안정적.
- `shap`은 비선형 모드에서 하락(1.000 -> 0.867)하지만 correlation 기반보다는 강건.

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
| 100 x 300 | 1.000 | 0.933 | 0.962 | 1.176 |
| 1000 x 300 | 1.167 | 1.185 | 0.941 | 2.282 |
| 1000 x 3000 | 1.011 | 1.018 | 0.854 | 2.368 |

그림(mode별 시간 민감도 ratio):

![Mode Time Ratio](visuals/time_mode_ratio.png)

## 4) 10,000 x 20,000 예상 CPU time (업데이트)

관측된 3개 shape에서 power-law(`time ~= k * rows^a * features^b`)로 추정.

mode 통합 기준 예상치:

| method | predicted time (sec) | approx |
|---|---:|---:|
| abs_pearson | 14.59 | ~0.24 min |
| min_dbi | 122.76 | ~2.05 min |
| mi | 3634.85 | ~60.6 min |
| shap | 97570.50 | ~1626.2 min |

mode별 참고(특히 SHAP):
- `default_nonlinear`: `shap ~49335 sec` (~822.2 min)
- `non_monotonic_strong`: `shap ~135175 sec` (~2252.9 min)

## 5) 추가 해석: 왜 MI가 SHAP보다 정확했고, 왜 SHAP 시간이 더 흔들리는가

### 5-1) MI가 SHAP보다 정확도가 높게 나온 가능한 이유

- 현재 데이터 생성에서는 `the_most_relevant`가 `target`의 직접 비선형 함수로 정의되어, "한 변수와 타깃의 직접 의존성"이 매우 강하다.
- `mi`는 각 후보와 타깃의 쌍별 의존성을 직접 점수화하므로, 이런 구조에서 top-1 식별이 유리하다.
- `shap`은 랜덤포레스트 예측모델을 먼저 학습한 뒤 중요도를 계산한다. 후보 간 상관(클러스터 latent 공유, config 효과)이 있으면 중요도가 유사 후보들로 분산될 수 있어 top-1 정확도가 떨어질 수 있다.
- 즉 본 결과의 해석은 "이 프로토콜 구조에서는 MI가 더 유리했다"이며, 일반적으로 항상 MI > SHAP를 의미하지는 않는다.

### 5-2) SHAP 시간이 mode에 따라 더 크게 변하는 이유

- `result.csv` 기준으로도 SHAP 런타임은 mode에 따라 큰 차이를 보인다.
- shape별 SHAP 중앙값 시간(초):

| shape | default_nonlinear | non_monotonic_strong | ratio (non/default) |
|---|---:|---:|---:|
| 100 x 300 | 0.423 | 0.519 | 1.227 |
| 1000 x 300 | 42.646 | 186.354 | 4.369 |
| 1000 x 3000 | 162.519 | 274.015 | 1.686 |

- 특히 `1000 x 300`에서 mode 전환 시 SHAP 시간이 약 4.37배 증가해, 함수 복잡도 변화에 민감하다는 점이 뚜렷하다.

결론:
- 정확도: 본 실험 구성에서는 MI가 가장 안정적으로 정답 feature를 찾았고, SHAP은 그 다음으로 강건했다.
- 시간: `abs_pearson/min_dbi/mi`는 주로 shape 영향, SHAP은 shape + 함수 복잡도(모드) 영향이 함께 크게 작동한다.
