# 🏠 House Prices - Advanced Regression Techniques

> Predict sales prices and practice feature engineering, RFs, and gradient boosting.
> Kaggle: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques

---

## 📌 1. Project Overview

- **Objective**: 79개의 설명 변수를 활용하여 아이오와주 에임스(Ames) 지역의 주택 가격을 예측하는 회귀 문제.
- **Data**: `train.csv`(label 포함), `test.csv`(label 없음)
- **Target**: `SalePrice`

---

## 🧪 Current Results (Local Validation)

> 현재는 `train.csv`만 사용한 hold-out 검증 결과이며, 리더보드 제출 전 단계입니다.

- **Metric (≈ Kaggle RMSLE)**: `0.13850` (RMSE on `log1p(SalePrice)` scale)
- **Reference (RMSE, original scale)**: `27121.50`
- **Validation**: `train_test_split(test_size=0.3, random_state=42)`
- **Model**: `RandomForestRegressor`
- **Preprocessing (high-level)**:
  - Target transform: `log1p(SalePrice)`
  - Missing (cat): `"None"` fill
  - Encoding: 일부 서열형 컬럼은 ordinal mapping, 나머지 범주형은 One-Hot
  - Missing (num): KNN imputation

➡️ 상세 실험/변경 로그: [`results/Experiment_Report.md`](./results/Experiment_Report.md)

---

## ⚙️ 2. Pipeline Architecture (Planned)

본 프로젝트는 `KYUN-Pipe` 코어 모듈을 활용하여 확장성 있는 머신러닝 파이프라인으로 구축될 예정입니다.

1.  **Data Loading**: `DataLoader`를 사용하여 대규모 피처를 포함한 데이터셋 로드.
2.  **Preprocessing**: `UniversalPreprocessor` 기반 파이프라인 구축.
    - **Missing Value Handling**: 각 변수의 특성에 맞는 전략(Median, None, Mode 등) 적용.
    - **Feature Engineering**: 다수의 컬럼을 결합하거나 새로운 의미를 추출하는 파생 변수 생성.
    - **Encoding**: 범주형 변수의 Ordinal/One-Hot 인코딩 처리.
3.  **Training**: `Trainer`를 통한 다양한 회귀 모델(XGBoost, LightGBM, CatBoost, Lasso 등) 실험 및 앙상블.
4.  **Evaluation**: `Evaluator`를 통한 RMSE 및 RMSLE 지표 분석.
5.  **Reporting**: `Reporter`를 활용한 실험 과정 및 결과의 시각화 및 기록.

## 🚀 3. How to Run

본 프로젝트는 `uv` 패키지 매니저를 사용합니다.

```bash
# 의존성 설치
uv sync

# 메인 실험 실행 (예정)
# python baseline_script.py
```

## 💡 Key Focus Points

- **High-Dimensional Data**: 컬럼이 상당히 많기 때문에, EDA를 통해 각 변수의 중요도를 파악하고 불필요한 노이즈를 제거하는 과정에 집중합니다.
- **Advanced Regression**: 선형 회귀부터 복잡한 부스팅 모델까지 폭넓게 적용하고 최적의 하이퍼파라미터를 탐색합니다.
- **Consistent Pipeline**: `KYUN-Pipe`의 구조를 유지하며 데이터 전처리부터 모델 평가까지 유출(Leakage) 없는 실험 환경을 보장합니다.

## 📊 Results & Artifacts

실험 결과물은 `results/` 폴더 내에 자동 아카이빙될 예정입니다.

| Artifact          | Description                                   | Format |
| :---------------- | :-------------------------------------------- | :----- |
| `metadata.json`   | 사용된 피처 리스트, 모델 하이퍼파라미터 정보  | JSON   |
| `metrics.json`    | RMSE, MAE, R2 Score 등 회귀 평가 지표         | JSON   |
| `predictions.csv` | Kaggle 제출을 위한 최종 테스트 세트 예측 결과 | CSV    |
| `model.pkl`       | 학습된 모델 객체                              | Joblib |
