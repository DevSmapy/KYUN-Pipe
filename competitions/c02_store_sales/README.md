# 🛒 Store Sales - Time Series Forecasting

> **Corporación Favorita의 데이터를 활용한 시계열 판매량 예측**
>
> 본 프로젝트는 에콰도르의 대형 식료품점인 Corporación Favorita의 데이터를 바탕으로 수천 개의 제품군에 대한 판매량을 예측하는 것을 목표로 합니다.

---

## 📅 Project Status: Phase 2 (OOP Refactoring & Pipeline Construction)

절차지향적인 `baseline_script.py`에서 벗어나, 재사용성과 확장성을 고려한 **OOP 아키텍처**로 리팩토링을 완료했습니다.

- [x] **OOP 리팩토링**: `UniversalPreprocessor` 및 `TimeSeriesTrainer` 도입
- [x] **전문적인 시계열 처리**: 유가 보간(Interpolation) 및 휴일 데이터 자동 병합 클래스 구현
- [x] **Target Engineering**: RMSLE 최적화를 위한 Log Transformation (`log1p` ↔ `expm1`) 자동화
- [x] **Validation Strategy**: `DataSplitter`를 통한 시간 기반 Hold-out 검증 구축
- [x] **Feature Engineering**: Lag(16, 30일), Rolling Mean(7일) 등 시계열 특징 추출

## 🏗 System Architecture (KYUN-Pipe)

본 프로젝트는 다음과 같은 모듈화된 구조로 실행됩니다:

1.  **DataLoader**: `train`, `test` 및 `context_data`(oil, stores 등)의 자동 로드 및 관리
2.  **UniversalPreprocessor**: Scikit-learn Pipeline 기반의 전처리 엔진
    - `HolidayChecker`: 공휴일 및 작업일 플래그 생성 및 병합
    - `OilPriceImputer`: 시계열 연속성 확보를 위한 유가 데이터 보간
    - `StoreStatsMerger`: 매장별 거래 통계 산출 및 병합
    - `TimeSeriesWindowFeaturizer`: Lag 및 Rolling Window 피처 생성
3.  **DataSplitter**: 시계열 누수(Data Leakage) 방지를 위한 날짜 기준 데이터 분할
4.  **TimeSeriesTrainer**:
    - 타겟 변수의 로그 스케일링 자동 관리
    - `LGBMRegressor`, `XGBRegressor` 등 다양한 모델과의 호환성 확보
    - Early Stopping 및 Validation 모니터링

## 🔍 Key Features Implemented

- **Temporal Features**: Year, Month, Day of week, Weekend flag
- **Window Features**: 16-day/30-day Lags, 7-day Rolling Mean of sales
- **External Factors**: Linear interpolated Oil Prices, Transferred holiday handling
- **Store Profiles**: Average/Std transactions per store

## 🛠 How to Run

```python
# main.py 실행 시 전체 파이프라인이 순차적으로 동작합니다.
python main.py
```

## 🔗 Competition Info

- [Kaggle: Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)

### 💡 수정 포인트:

- **Phase 변경**: `Phase 1 (EDA)`에서 `Phase 2 (Refactoring & Pipeline)`으로 격상시켰습니다.
- **핵심 컴포넌트 강조**: 우리가 만든 `HolidayChecker`, `OilPriceImputer` 등의 클래스 이름을 명시하여 전문성을 높였습니다.
- **TimeSeriesTrainer 특장점**: 로그 변환 자동화(`log1p` ↔ `expm1`) 부분을 강조하여 시계열 예측에 특화된 프로젝트임을 보여주었습니다.

이 README를 통해 프로젝트의 완성도가 한눈에 들어올 거예요! (웃음)
