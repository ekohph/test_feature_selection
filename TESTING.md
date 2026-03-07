# 테스트 전략: Feature Relevance 선택

## 목적

매 테스트 실행마다 **3개 데이터셋 설정 x 5개 seed x 2개 관계 모드로 생성한 데이터셋 30개**에서
feature-selection 동작을 평가합니다.

각 데이터셋에서 수행할 일:
- 알려진 `target`과 알려진 `the_most_relevant`(정답)로 데이터 생성
- 선택 알고리즘 실행
- 선택 품질과 실행 시간 기록

## 데이터셋 프로토콜 (랜덤 30개)

- 데이터셋 설정 3개:
  - (1) Row 100개, Feature 300개, Cluster 3개
  - (2) Row 1000개, Feature 300개, Cluster 3개
  - (3) Row 1000개, Feature 3000개, Cluster 30개
- 각 설정마다 seed `[0, 1, 2, 3, 4]`를 반복해 생성
- 각 `(setting, seed)` 조합에서 관계 모드 2개를 모두 생성:
  - `default_nonlinear` (기존)
  - `non_monotonic_strong` (신규)
- 총 30개 데이터셋 생성 (`3 settings x 5 seeds x 2 modes`)
- 위 3개 설정은 모두 `cluster당 feature 100개`를 만족 (`n_features / n_clusters = 100`)
- 각 데이터셋은 클러스터 제외 가정을 만족해야 함:
  - 선택된 feature는 `target`일 수 없음
  - 선택된 feature는 `target`과 같은 클러스터일 수 없음

생성 API:
- `generate_synthetic_dataset(...)`

각 생성 데이터셋 필수 포함 컬럼:
- `config`
- `target`
- `target`과 다른 클러스터에 있는 알려진 `the_most_relevant`

## 생성 수식 요약

아래 식은 `README.md`와 동일한 생성 로직 요약입니다.

```text
x_f = z_f + eps,   eps ~ N(0, 0.30^2)
target = 0.6 * x_target + 1.4 * s + eps_t,   eps_t ~ N(0, 0.10^2)

the_most_relevant =
    1.1 * tanh(1.2 * target)
  + 0.25 * target^3
  + 0.15 * x_most_relevant
  + 0.40 * cfg_effect(config)
  + eps_r,   eps_r ~ N(0, 0.03^2)

# non_monotonic_strong mode
the_most_relevant =
    1.20 * sin(2.4 * target)
  + 0.55 * cos(0.8 * target^2)
  + 0.20 * target^2
  + 0.12 * x_most_relevant
  + 0.45 * cfg_effect(config)
  + eps_r,   eps_r ~ N(0, 0.03^2)

# canonical normalization (before selection)
x_f = (x_f - mean(x_f)) / std(x_f)
```

요점:
- `the_most_relevant`는 `target`과 비선형 관계(`tanh`, `target^3`)를 갖도록 생성됨
- `non_monotonic_strong` 모드에서는 `sin`, `cos(target^2)`를 포함한 강한 비단조 비선형 관계를 사용함
- 클러스터 제외 규칙 하에서 가장 높은 관련성을 갖도록 설계됨
- `cfg_effect`로 config 기반 분리 가능성을 유지함
- feature selection 전에 모든 feature(`config` 제외)에 canonical normalization(z-score)을 적용함

## 데이터셋별 지표

선택된 feature 기준:
- `selected_feature`
- `config` 라벨 기준 `[selected_feature, target]`의 DBI
- `selected_feature`와 `target`의 Pearson 상관계수

정답 feature (`the_most_relevant`) 기준:
- feature 이름 자체 (`the_ground_truth`)
- `config` 라벨 기준 `[the_ground_truth, target]`의 DBI
- `the_ground_truth`와 `target`의 Pearson 상관계수

실행 시간:
- 해당 데이터셋에서 선택 실행에 걸린 총 시간
- `time.perf_counter()` 사용

## Selector 방법론

프로토콜은 selector(피처 선택 기준)를 바꿔 여러 방법론을 비교할 수 있습니다.
현재 구현된 방법론은 아래 4가지입니다.

1. `abs_pearson`
- 정의: 후보 피처와 `target`의 절대 Pearson 상관계수가 최대인 피처 선택
- 장점: 계산이 빠르고 해석이 단순함
- 한계: 비선형(단조/비단조) 관계를 충분히 반영하지 못할 수 있음

2. `min_dbi`
- 정의: `[candidate, target]` 2차원 공간에서 `config` 라벨 기준 Davies-Bouldin Index(DBI)가 최소인 피처 선택
- 장점: `config` 분리도 관점의 선택이 가능함
- 한계: `target`과의 직접 상관이 낮은 피처가 선택될 수 있음

3. `shap`
- 정의: 트리 기반 예측모델을 학습한 뒤 `import shap`의 `TreeExplainer`로 SHAP 값을 계산하고, 절대 SHAP 평균(Shapley index) 이 최대인 피처를 선택
- 예시 절차:
  - 모델 학습: `model.fit(X_candidates, y_target)`
  - 설명기 생성: `explainer = shap.TreeExplainer(model)`
  - SHAP 계산: `shap_values = explainer.shap_values(X_candidates)`
  - 피처별 index: `mean_abs_shap = mean(abs(shap_values), axis=0)`
  - 선택: `argmax(mean_abs_shap)`
- 장점: 비선형 및 상호작용 효과를 반영한 중요도 산출 가능
- 한계: 모델 학습 + SHAP 계산 비용이 커서 데이터가 커질수록 실행시간 증가

4. `mi`
- 정의: `mutual_info_regression`으로 후보 feature와 `target`의 mutual information을 계산해 최대값 선택
- 장점: 비선형 의존성을 반영 가능
- 한계: `abs_pearson` 대비 계산 비용이 증가함

참고:
- 모든 selector는 공통으로 클러스터 제외 규칙(타깃 본인/동일 클러스터 제외) 이후 후보군에서 동작합니다.
- 동점일 때는 피처 이름의 사전순으로 결정합니다.

## Selector 스케일링 예측

표기:
- `N`: row 수(샘플 수)
- `P`: 후보 feature 수(클러스터 제외 적용 후)
- `K`: `config` 라벨 수

예상 시간 복잡도(데이터셋 1개 기준):

1. `abs_pearson`
- 각 후보마다 `target`과 상관계수 1회 계산
- 대략 `O(P * N)`
- 해석: row 수가 2배면 시간도 거의 2배, 후보 feature 수가 2배여도 거의 2배

2. `min_dbi`
- 각 후보마다 DBI 계산(라벨별 centroid/scatter + 라벨쌍 비교)
- 대략 `O(P * (N + K^2))` (일반적으로 `N` 항이 지배적)
- 해석: 실무에서 보통 `K << N`이므로 거의 `O(P * N)`처럼 증가하지만, `K`가 커지면 추가 비용이 눈에 띔

3. `shap` (`TreeExplainer` 기준)
- 트리 모델 학습 + SHAP 계산으로 구성됨
- 대략 `O(train_tree(N, P) + explain_tree(N, P, T, D))`
  - `T`: 트리 개수, `D`: 트리 깊이(또는 리프 구조 복잡도)
- 해석: 일반적으로 `abs_pearson`/`min_dbi`보다 상수항과 구조 의존 비용이 커서, `N`, `P`, `T`, `D`가 커질수록 증가폭이 큼

4. `mi`
- MI 추정(`mutual_info_regression`)을 각 후보 feature에 대해 수행
- 대략 `O(P * N)`에 가까운 증가 경향
- 해석: 비선형 관계를 반영하지만 계산량은 `abs_pearson`보다 큼

요약:
- 네 방법 모두 feature 수 `P`에 대해 선형적으로 증가
- row 수 `N` 증가에도 대체로 선형 증가
- 예상 상대 속도(일반적): `abs_pearson` <= `min_dbi` <= `mi` < `shap(TreeExplainer)` (데이터 분포/모델 파라미터에 따라 변동 가능)

## 결과 테이블 형식 (필수)

결과 테이블은 **120행**이어야 하며(`3 settings x 5 seeds x 2 modes x 4 selectors`) 다음 컬럼을 포함해야 합니다.
- `selection_method` 
- `n_rows` 
- `n_features` 
- `n_clusters` 
- `selected_feature`
- `dbi_selected_feature`
- `corr_selected_feature`
- `computation_time_sec`
- `the_ground_truth`
- `dbi_ground_truth_feature`
- `corr_ground_truth_feature`

선택(권장) 추가 컬럼:
- `dataset_id`
- `seed`
- `target`
- `relationship_mode`

반올림 규칙:
- 결과 DataFrame에 `.round(3)` 적용 후 `result.csv` 저장

## 검증 체크리스트

- 결과 테이블 행 수가 정확히 120인가
- 각 `(n_rows, n_features, n_clusters)` 설정마다 seed가 정확히 5개인가 (`0,1,2,3,4`)
- 각 `(n_rows, n_features, n_clusters, seed)` 조합마다 관계 모드가 2개(`default_nonlinear`, `non_monotonic_strong`)인가
- 각 `(dataset setting, seed, relationship_mode)` 조합마다 selector 결과가 4행인가
- 필수 컬럼이 모두 존재하는가
- 필수 결과 컬럼에 null이 없는가
- `selected_feature`와 `the_ground_truth`가 유효한 데이터셋 컬럼인가
- 모든 행에서 실행 시간이 0 이상인가
- `result.csv` 값이 소수점 셋째 자리로 반올림되어 저장되는가
- 생성된 데이터셋에서 feature 컬럼(`config` 제외)이 canonical normalization 상태(평균≈0, 표준편차≈1)인지 확인
