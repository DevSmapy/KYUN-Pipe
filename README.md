# 🏛️ KYUN-Pipe

> **"Code is read much more often than it is written."**
> Kaggle 풀이를 **재사용 가능한 OOP 기반 파이프라인**으로 리팩토링하고, 실험 결과(메타데이터/메트릭/모델)를 **일관된 구조로 아카이빙**합니다.

---

## 🎯 Core Focus

- **OOP-Centric**: 절차지향 코드를 `DataLoader`, `Preprocessor`, `Trainer`, `Reporter` 중심으로 구조화
- **Scalability**: 분류 / 회귀 / 시계열 등 다양한 태스크에 대응 가능한 모듈형 설계
- **Reproducibility**: 실행 단위 결과 폴더(`results/<run_id>/`)에 메트릭/메타데이터/모델 아티팩트 고정 기록
- **Robustness**: 데이터 누수(Leakage) 방지, 검증 전략(holdout / time split) 분리

## 🛠 Stack

- **Environment**: Python 3.13 / `uv` (Package Manager)
- **Lint & Format**: `ruff` + `pre-commit` (+ GitHub Actions)
- **Type Check**: `mypy`
- **Libraries**: pandas, scikit-learn (프로젝트별: LightGBM / XGBoost 등)

## 📁 Repository Structure

```text
.
├── core/                # 공통 Base Class 및 Utility (DataLoader, Trainer, Preprocessor 등)
├── competitions/        # Competition-specific Scripts
│   ├── c01_spaceship_titanic/ # Classification: Spaceship Titanic
│   ├── c02_store_sales/       # Time Series Regression: Store Sales Forecasting
│   └── c03_house_prices/      # Regression: House Prices
└── README.md
```

## 📦 Result Artifacts (Archived)

각 실행은 `results/<run_id>/` 폴더로 자동 아카이빙되는 것을 목표로 합니다.

예시(competition 내부):

- `competitions/<competition>/results/<run_id>/metadata.json`
- `competitions/<competition>/results/<run_id>/metrics.json`
- `competitions/<competition>/results/<run_id>/<model>.pkl`
- `competitions/<competition>/results/<run_id>/submission_<model>.csv` (생성되는 경우)

> 모델 파일(`.pkl`)은 용량이 커질 수 있으니, 레포 정책에 따라 추적 방식(Git LFS 등)을 정하는 것을 권장합니다.

## 🚀 Current Milestone

- [x] Phase 1: OOP 파이프라인 기초 구축 (`c01_spaceship_titanic`)
- [x] Phase 2: 시계열(Time-series) 전처리/트레이너 확장 (`c02_store_sales`)
- [x] Phase 3: 정형 데이터 회귀 파이프라인 실험 결과 아카이빙 추가 (`c03_house_prices`)
- [ ] Phase 4: 파이프라인 안정화 (실험/제출 메타데이터 스키마 표준화 + 공통 CLI 도입)
- [ ] Phase 5: 실시간 데이터 시뮬레이터 및 ETL 파이프라인 연결

---

**Author: DevSmapy(Kyun)**
