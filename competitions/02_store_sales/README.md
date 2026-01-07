# 🛒 Store Sales - Time Series Forecasting

> **Corporación Favorita의 데이터를 활용한 시계열 판매량 예측**
>
> 본 프로젝트는 에콰도르의 대형 식료품점인 Corporación Favorita의 데이터를 바탕으로 수천 개의 제품군에 대한 판매량을 예측하는 것을 목표로 합니다.

---

## 📅 Project Status: Phase 1 (EDA & Feature Exploration)

현재 **절차지향적 접근(Procedural Approach)**을 통해 데이터를 분석하고 가용한 피처들을 탐색하고 있는 초기 단계입니다.

- [x] 데이터 기초 통계 확인 (`train`, `test`, `stores`, `transactions`, `oil`, `holidays_events`)
- [ ] 시계열 특성 파악 (Trend, Seasonality, Holidays)
- [ ] 외부 요인(유가, 공휴일)과 판매량의 상관관계 분석
- [ ] 가공 가능한 파생 피처(Derived Features) 목록화
- [ ] 베이스라인 모델 구축 및 검증

## 🔍 Key Data Components

분석 중인 주요 데이터 포인트는 다음과 같습니다:

1.  **Sales Data**: `store_nbr`, `family`, `onpromotion` 등의 정보를 포함한 핵심 판매 기록
2.  **Stores**: 상점의 위치(City, State), 타입(Type), 클러스터 정보
3.  **Oil Prices**: 에콰도르 경제에 밀접한 영향을 미치는 유가 데이터 (시계열 외생 변수)
4.  **Holidays & Events**: 공휴일, 이벤트, 그리고 급여일(15일, 말일) 등의 일정 정보
5.  **Transactions**: 각 상점의 일별 트랜잭션 수 (Sales와 밀접한 상관관계)

## 🛠 Roadmap

1.  **Exploratory Data Analysis (Current)**: 데이터의 분포를 살피고 결측치 처리 전략 수립
2.  **Feature Engineering**: 시계열 특징(Lag, Rolling mean), 공휴일 플래그, 유가 보간법 등 적용
3.  **Modeling**: XGBoost, LightGBM 또는 Prophet/NeuralProphet을 활용한 예측
4.  **Refactoring (OOP)**: `KYUN-Pipe` 구조에 맞춰 `DataLoader`, `Preprocessor`, `Trainer`로 모듈화

## 🔗 Competition Info

- [Kaggle: Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)
