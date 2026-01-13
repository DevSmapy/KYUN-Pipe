# 🏛️ KYUN-Pipe

> **"Code is read much more often than it is written."**
> Kaggle 풀이를 **재사용이 가능한 OOP 기반 파이프라인**으로 리팩토링하여 아카이빙합니다.

---

## 🎯 Core Focus

- **OOP-Centric**: 절차지향 코드를 `DataLoader`, `Preprocessor`, `Trainer` 클래스로 구조화
- **Scalability**: 시계열, 분류, 회귀 등 다양한 태스크에 즉시 대응 가능한 모듈형 설계
- **Robustness**: 데이터 누수(Leakage) 방지 및 Target Engineering 자동화 캡슐화

## 🛠 Stack

- **Environment**: Python 3.13 / uv (Package Manager)
- **Libraries**: Scikit-learn, Pandas, NumPy, LightGBM, XGBoost
- **Engineering**: OOP, Time-series Validation, Data-driven Design

## 📁 Repository Structure

```text
.
├── core/                # 공통 Base Class 및 Utility (DataLoader, Trainer, Preprocessor 등)
├── competitions/        # Competition-specific Scripts
│   ├── c01_spaceship_titanic/ # Classification: Spaceship Titanic
│   └── c02_store_sales/       # Time Series Regression: Store Sales Forecasting
└── README.md
```

## 🚀 Current Milestone

- [x] Phase 1: Titanic 데이터셋을 통한 OOP 기초 파이프라인 구축 (`c01_spaceship_titanic`)
- [x] Phase 2: 시계열(Time-series) 전문 트레이너 및 전처리 모듈 확장 (`c02_store_sales`)
- [ ] Phase 3: 정형 데이터 competition 3종 리팩토링 및 파이프라인 안정화
- [ ] Phase 4: 실시간 데이터 시뮬레이터 및 ETL 파이프라인 연결

---

**Author: DevSmapy(Kyun)**
