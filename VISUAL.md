# 시각화 가이드

`run_protocol.py` 실행 후 생성된 결과(`result.csv`, `tests/tmp`)를 시각화합니다.

## 목적

1. 정답 피처/선택 피처와 `target`의 관계를 산점도로 확인
2. 데이터셋 shape 및 선택 방법별 계산 시간 분포를 비교
3. 관계 모드(`relationship_mode`) 추가 이후 시간 분포 해석 기준을 명확히 유지

## 입력

- 결과 테이블: `result.csv`
- 데이터셋 파일: `tests/tmp/dataset_<id>_r<n_rows>_f<n_features>_c<n_clusters>_seed_<seed>.feather`

## 출력

- `visuals/scatter_gt.png`
  - 각 `dataset_id`별 `the_ground_truth` vs `target` 산점도
  - 점 색상: `config` (기본 범주형 팔레트 `tab20`)
- `visuals/scatter_selected.png`
  - `selected_feature != the_ground_truth`인 경우만 `selected_feature` vs `target` 산점도
  - 불일치가 없으면 안내 문구 이미지를 저장
- `visuals/time_bar.png`
  - x축: dataset shape (`#rows x #features`)
  - 색상: `selection_method`
  - y축: `computation_time_sec` (log scale)
  - 기본 요약: `median_iqr` (center=median, error=IQR)
- `visuals/time_bar_default_nonlinear.png`
- `visuals/time_bar_non_monotonic_strong.png`
  - `relationship_mode`별로 분리 집계한 시간 막대 그래프

## 실행 방법 (PowerShell)

```powershell
$env:PYTHONPATH = "src"
python visualize_results.py --result-csv result.csv --datasets-dir tests/tmp --output-dir visuals
```

요약 방식을 명시적으로 지정하려면:

```powershell
python visualize_results.py --result-csv result.csv --datasets-dir tests/tmp --output-dir visuals --time-summary median_iqr
```

옵션:

- `--time-summary median_iqr` (기본)
- `--time-summary mean_std`

## 참고

- 현재 프로토콜(3 settings x 5 seeds x 2 relationship modes x 4 selectors) 기준 `result.csv`는 120행입니다.
- `result.csv` 필수 컬럼:
  - `selection_method`, `dataset_id`, `seed`, `n_rows`, `n_features`, `n_clusters`
  - `selected_feature`, `the_ground_truth`, `target`, `computation_time_sec`
- `relationship_mode` 컬럼이 포함되면, 현재 `time_bar.png`는 해당 모드들을 shape별로 함께 집계합니다.
