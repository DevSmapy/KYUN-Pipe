# 🏛️ KYUN-Pipe

> **"Code is read much more often than it is written."**
> Kaggle 풀이를 **재사용이 가능한 OOP 기반 파이프라인**으로 리팩토링하여 아카이빙합니다.

---

## 🎯 Core Focus

- **OOP-Centric**: 절차지향 코드를 `DataLoader`, `Preprocessor`, `Model` 클래스로 구조화
- **Scalability**: 새로운 Competition에도 즉시 적용 가능한 모듈형 파이프라인 설계
- **Code Archive**: 라이브러리 공식 문서 수준으 ㅣ정교한 주석과 리팩토링된 코드 자산 구축

## 🛠 Stack

- **Environment**: Python 3.x/PyCharm
- **Libraries**: Scikit-learn, Pandas, NumPy, XGBoost/LightGBM
- **Engineering**: OOP, Data-driven Design, (Future: Kafka, PySpark)

## 📁 Repository Structure

```text
.
├── core/                # 공통 Base Class 및 Utility (DataLoader, Trainer 등)
├── competitions/        # Competitions
│   ├── 01_titanic/      # Titanic: Machine Learning from Disaster
│   └── 02_house_prices/ # House Prices: Advanced Regression Techniques
└── README.md
```

## 🚀 Current Milestone

- [ ] Phase 1: Titanic 데이터셋을 통한 OOP 기초 파이프라인 구축
- [ ] Phase 2: 정형 데이터 competition 3종 리팩토링 완료
- [ ] Phase 3: 실시간 데이터 시뮬레이터 및 ETL 파이프라인 연결

---

**Author: DevSmapy(Kyun)**
