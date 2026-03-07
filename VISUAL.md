# 시각화 가이드

`run_protocol.py` 실행 후 생성된 결과를 시각화합니다.

## 목적

1. 선택된 피처와 타깃의 관계를 산점도로 확인
2. 선택 방법별 계산 시간을 막대그래프로 비교

## 입력

- 결과 테이블: `result.csv`
- 데이터셋 파일: `tests/tmp/dataset_<id>_r<n_rows>_f<n_features>_c<n_clusters>_seed_<seed>.feather`

## 출력

- `visuals/scatter.png`
  - 전체 데이터셋 x method를 한 figure의 subplot grid로 저장
  - x축: `selected_feature`, y축: `target`
  - 점 색상: `config`
- `visuals/time_bar.png`
  - x축: `dataset_id`
  - method별 계산시간(`computation_time_sec`) grouped bar

## 실행 방법 (PowerShell)

```powershell
$env:PYTHONPATH = "src"
python visualize_results.py --result-csv result.csv --datasets-dir tests/tmp --output-dir visuals
```

## 참고

- `result.csv`에는 최소한 다음 컬럼이 있어야 합니다:
  - `selection_method`, `dataset_id`, `seed`, `n_rows`, `n_features`, `n_clusters`
  - `selected_feature`, `target`, `computation_time_sec`
- 스크립트는 데이터셋 파일명을 위 규칙으로 찾습니다.
