# 🧾 Experiment Report — House Prices (c03)

## Summary

- **Metric (≈ Kaggle RMSLE)**: `0.13850` (RMSE on `log1p(SalePrice)` scale)
- **RMSE (original scale)**: `27121.50`
- **Validation**: hold-out (`test_size=0.3`, `random_state=42`)
- **Model**: RandomForestRegressor
- **Status**: Baseline solid. Next: KFold CV + boosting models.

---

## Experiment: RF Baseline + Ordinal/OneHot + KNN

### Dataset

- `train.csv` only (no submission yet)

### Target

- `y = log1p(SalePrice)`
- Evaluation aligned to Kaggle: RMSE on log-scale (≈ RMSLE)

### Preprocessing

- Categorical missing: `"None"`
- Ordinal encoding: column-wise mapping (domain-driven, strict validation)
- Remaining categorical: One-Hot encoding
- Numeric missing: KNNImputer

### Notes / Risks

- Single hold-out score can be optimistic/pessimistic → move to KFold.
- KNNImputer cost/behavior should be compared vs median imputation under CV.

---

## Next Experiments (Planned)

1. KFold(5) CV for stability (mean/std)
2. LightGBM / XGBoost baseline
3. Outlier handling comparison (rule-based)
