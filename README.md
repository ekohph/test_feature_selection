# Feature Relevance 데이터셋 생성기

이 프로젝트는 feature relevance 실험을 위한 합성 데이터셋을 생성합니다.
피처는 클러스터 단위로 구성되며, 메타데이터 컬럼 `config`가 추가됩니다.

피처 이름 형식은 `f_{i}_c_{j}`입니다.
- `i`: 클러스터 내부 피처 인덱스
- `j`: 클러스터 인덱스

예: `f_2_c_3`는 3번 클러스터의 2번째 피처를 의미합니다.

생성 데이터셋 형태:
`n_rows x [1 ('config') + n_features]`

## 문제 설정

주어진 `target` 피처에 대해, 가장 관련성이 높은 단일 피처를 선택하는 것이 목표입니다.

이 생성기에서 `the_most_relevant`는 다음 성질을 갖도록 설계되어 있습니다.
- `target`과 강한 관련성
- `config` 값에 따른 충분한 변동성
- 관계 모드(`relationship_mode`)에 따라 단조/비단조 비선형 구조를 선택 가능

같은 클러스터에 속한 피처끼리는 의도적으로 상관이 생기며,
선택 문제에서는 보통 `target`과 같은 클러스터 피처를 제외합니다.

데이터셋은 빠른 I/O를 위해 [Feather](https://arrow.apache.org/docs/python/feather.html) 형식으로 저장할 수 있습니다.

## 생성 수식

기호:
- `z_f`: 피처 `f`의 클러스터 기반 잠재 신호 (`cluster_latent`)
- `x_f^(0)`: 재정의 전 피처 `f`의 기본값
- `target*`: 재정의 후 최종 `target`
- `mr*`: 재정의 후 최종 `the_most_relevant`
- `s`: `target`과 공유되는 행 단위 공통 신호 (`shared_signal`), `s ~ N(0, 1)`
- `eps`: 가우시안 잡음
- `cfg_effect(config)`: config 라벨별 랜덤 효과 함수
  - 각 config 라벨 `k`에 대해 `a_k ~ N(0, config_effect_scale^2)`를 먼저 샘플링
  - 각 행 `i`에서는 `cfg_effect(config_i) = a_{config_i}`로 할당
  - 기본값: `config_effect_scale = 3.0`

행(row) 단위 생성식은 아래와 같습니다.

기본 피처 (재정의 전):

```text
x_f^(0) = z_f + eps,   eps ~ N(0, 0.30^2)
```

타깃 피처 (최종값):

```text
target* = 0.6 * x_target^(0) + 1.4 * s + eps_t,   eps_t ~ N(0, 0.10^2)
```

가장 관련성 높은 피처 `the_most_relevant` (최종값):

1) `relationship_mode="default_nonlinear"` (기존):

```text
mr* =
    1.1 * tanh(1.2 * target*)
  + 0.25 * (target*)^3
  + 0.15 * x_most_relevant^(0)
  + 0.40 * cfg_effect(config)
  + eps_r,   eps_r ~ N(0, 0.03^2)
```

2) `relationship_mode="non_monotonic_strong"` (신규):

```text
mr* =
    1.20 * sin(2.4 * target*)
  + 0.55 * cos(0.8 * (target*)^2)
  + 0.20 * (target*)^2
  + 0.12 * x_most_relevant^(0)
  + 0.45 * cfg_effect(config)
  + eps_r,   eps_r ~ N(0, 0.03^2)
```

Canonical normalization (모든 feature, selection 이전):

```text
x_f = (x_f - mean(x_f)) / std(x_f)
```

해석:
- `z_f`는 코드의 `cluster_latent[:, cluster-1]`에 해당하며, `x_f^(0)`는 여기에 노이즈를 더한 값입니다.
- `target` 식의 좌변(`target*`)과 우변의 `x_target^(0)`은 같은 이름의 "동일 시점 변수"가 아니라, 재정의 전/후 단계가 다른 값입니다.
- `the_most_relevant` 식의 `x_most_relevant^(0)`은 의도된 항입니다. `x_target^(0)`으로 바꾸면 "자기 기본값" 기여가 사라져 생성 구조가 달라집니다.
- `s`는 모든 행에서 독립적으로 생성되지만(`N(0,1)`), `target`에 공통으로 주입되어 관련성을 강화합니다.
- `cfg_effect(config)`는 같은 config를 가진 행들이 동일한 효과값을 공유하도록 만들어 DBI 평가에 필요한 분리 가능성을 높입니다.
- `default_nonlinear`는 기존의 단조 비선형(`tanh`, `target^3`) 구조를 사용합니다.
- `non_monotonic_strong`는 `sin`, `cos(target^2)`를 포함한 강한 비단조 비선형 구조를 사용합니다.
- 비선형 결합 강도를 높여, 클러스터 제외 규칙 하에서 `the_most_relevant`가 가장 관련성 높은 후보가 되도록 설계했습니다.
- `cfg_effect` 항은 DBI 평가에 필요한 config 분리 가능성을 유지합니다.
- 최종 반환 직전 모든 feature에 canonical normalization(z-score, 평균 0/표준편차 1)을 적용합니다.

## 빠른 시작 (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install pytest
$env:PYTHONPATH = "src"

# 샘플 데이터 생성 및 저장
python -c "from dataset.generate import make_and_save; make_and_save('data/synthetic.feather')"
```

## 사용 예시

```python
from dataset.generate import generate_synthetic_dataset

df = generate_synthetic_dataset(
    n_features=200,
    n_clusters=10,
    n_rows=1000,
    target="f_1_c_1",
    the_most_relevant="f_1_c_3",
    seed=42,
)

df.to_feather("data/mydata.feather")
```

## 프로토콜 실행

`run_protocol.py`는 `TESTING.md` 프로토콜을 실행하는 스크립트입니다.

동작:
- 데이터 크기 설정 3개 각각에 대해 seed `0, 1, 2, 3, 4`를 모두 반복해 데이터셋 생성
- 각 `(dataset setting, seed)` 조합에서 관계 모드 2종(`default_nonlinear`, `non_monotonic_strong`)을 모두 생성
  - 총 데이터셋 수: `3 settings x 5 seeds x 2 modes = 30`
  - 설정: `(100, 300, 3)`, `(1000, 300, 3)`, `(1000, 3000, 30)`
  - 위 3개 설정은 모두 `cluster당 feature 100개`를 만족
- 생성 데이터셋을 `tests/tmp`에 저장
- 4개 선택 방법(`abs_pearson`, `min_dbi`, `shap`, `mi`)으로 데이터셋별 선택 피처와 정답 피처를 평가
- 120행 결과 테이블(3개 설정 x 5개 seed x 2개 mode x 4개 방법)을 생성해 `result.csv`로 저장
- 저장 전 `.round(3)` 적용

실행 명령:

```powershell
$env:PYTHONPATH = "src"
python run_protocol.py
```

출력:
- `tests/tmp`의 테스트 데이터셋 (`dataset_1_r100_f300_c3_seed_0.feather` 등)
- 프로젝트 루트의 `result.csv`

## 테스트 실행

```powershell
$env:PYTHONPATH = "src"
pytest -q
```

## 시각화 실행

`VISUAL.md`와 `visualize_results.py`를 사용해 결과를 시각화할 수 있습니다.

```powershell
$env:PYTHONPATH = "src"
python visualize_results.py --result-csv result.csv --datasets-dir tests/tmp --output-dir visuals
```

## 참고

- 현재 패키지 설정 파일(`pyproject.toml`/`setup.py`)이 없어 직접 실행 시 `PYTHONPATH=src`가 필요합니다.
- 테스트 파일: `tests/test_generate.py`, `tests/test_testing_protocol.py`

생성 옵션 전체는 `src/dataset/generate.py`를 참고하세요.
