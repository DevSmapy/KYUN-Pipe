import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

SCALE_QUAL = {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
SCALE_EXPOSURE = {"None": 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4}
SCALE_FINISH = {
    "None": 0,
    "Unf": 1,
    "LwQ": 2,
    "Rec": 3,
    "BLQ": 4,
    "ALQ": 5,
    "GLQ": 6,
}

# apply_quality_value_encoding이 실제로 지원하는 ordinal 컬럼 정의(핵심 수정 포인트)
ORDINAL_MAPS: dict[str, dict[str, int]] = {
    "ExterQual": SCALE_QUAL,
    "ExterCond": SCALE_QUAL,
    "BsmtQual": SCALE_QUAL,
    "BsmtCond": SCALE_QUAL,
    "HeatingQC": SCALE_QUAL,
    "KitchenQual": SCALE_QUAL,
    "FireplaceQu": SCALE_QUAL,
    "GarageQual": SCALE_QUAL,
    "GarageCond": SCALE_QUAL,
    "PoolQC": SCALE_QUAL,
    "Fence": {"None": 0, "MnWw": 1, "GdWo": 2, "MnPrv": 3, "GdPrv": 4},
    "LandSlope": {"Sev": 0, "Mod": 1, "Gtl": 2},
    "LotShape": {"Reg": 0, "IR1": 1, "IR2": 2, "IR3": 3},
    "BsmtExposure": SCALE_EXPOSURE,
    "BsmtFinType1": SCALE_FINISH,
    "BsmtFinType2": SCALE_FINISH,
    "CentralAir": {"N": 0, "Y": 1},
}


def apply_quality_value_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """
    Column-wise ordinal encoding with:
    - safe handling for missing columns (skip if not present)
    - normalization of missing tokens (NaN/'NA'/'' -> 'None')
    - strict validation: raises if unknown labels appear for a mapped column
    """
    out = df.copy()

    none_tokens = {"NA", "nan", ""}

    for col, ordinal_map in ORDINAL_MAPS.items():
        if col not in out.columns:
            continue

        s = (
            out[col]
            .fillna("None")
            .astype(str)
            .str.strip()
            .replace({tok: "None" for tok in none_tokens})
        )

        mapped = s.map(ordinal_map)

        if mapped.isna().any():
            unknown = sorted(set(s[mapped.isna()].unique()))
            raise ValueError(
                f"[apply_quality_value_encoding] {col}: unknown labels: {unknown}"
            )

        out[col] = mapped.astype("int")

    return out


if __name__ == "__main__":
    # split data by categorical, numeric
    train = pd.read_csv(
        "/kaggle/input/house-prices-advanced-regression-techniques/train.csv"
    )
    df = train.copy()

    y = np.log1p(df["SalePrice"])
    X = df.drop("SalePrice", axis=1)

    cat_columns = X.select_dtypes("object").columns
    num_columns = X.select_dtypes("number").columns

    train_X, valid_X, train_y, valid_y = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    train_cat = train_X[cat_columns].fillna("None")
    valid_cat = valid_X[cat_columns].fillna("None")

    # (핵심) "Qual" 포함 여부가 아니라, ORDINAL_MAPS에 정의된 컬럼만 ordinal로 처리
    ordinal_cols = pd.Index([c for c in cat_columns if c in ORDINAL_MAPS])

    train_cat.loc[:, ordinal_cols] = apply_quality_value_encoding(
        train_cat.loc[:, ordinal_cols]
    )
    valid_cat.loc[:, ordinal_cols] = apply_quality_value_encoding(
        valid_cat.loc[:, ordinal_cols]
    )

    # one-hot은 ordinal_cols를 제외한 나머지 범주형에만 적용
    onehotencoder = OneHotEncoder(
        sparse_output=False, handle_unknown="ignore"
    ).set_output(transform="pandas")
    diff_cols = cat_columns.difference(ordinal_cols)

    train_oh_encoded = onehotencoder.fit_transform(train_cat.loc[:, diff_cols])
    valid_oh_encoded = onehotencoder.transform(valid_cat.loc[:, diff_cols])

    train_cat = pd.concat([train_oh_encoded, train_cat.loc[:, ordinal_cols]], axis=1)
    valid_cat = pd.concat([valid_oh_encoded, valid_cat.loc[:, ordinal_cols]], axis=1)

    knn_imputer = KNNImputer(n_neighbors=5)
    train_num = pd.DataFrame(
        knn_imputer.fit_transform(train_X[num_columns]),
        columns=num_columns,
        index=train_X.index,
    )
    valid_num = pd.DataFrame(
        knn_imputer.transform(valid_X[num_columns]),
        columns=num_columns,
        index=valid_X.index,
    )

    train_X_preprocessed = pd.concat([train_num, train_cat], axis=1)
    valid_X_preprocessed = pd.concat([valid_num, valid_cat], axis=1)

    # change data type from object to float
    train_X_preprocessed = train_X_preprocessed.astype(float)
    valid_X_preprocessed = valid_X_preprocessed.astype(float)

    # train model (재현성 확보)
    RF = RandomForestRegressor(random_state=42)
    RF.fit(train_X_preprocessed, train_y)

    # evaluate model
    preds_log = RF.predict(valid_X_preprocessed)

    rmse_log = np.sqrt(mean_squared_error(valid_y, preds_log))
    print("RMSE on log1p scale (≈ RMSLE):", rmse_log)

    valid_y_orig = np.expm1(valid_y)
    preds_orig = np.expm1(preds_log)
    rmse_orig = np.sqrt(mean_squared_error(valid_y_orig, preds_orig))
    print("RMSE on original scale:", rmse_orig)

    # LightGBM
    lgbm = LGBMRegressor(
        n_estimators=3000,
        learning_rate=0.03,
        random_state=42,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
    )

    lgbm.fit(
        train_X_preprocessed,
        train_y,
        eval_set=[(valid_X_preprocessed, valid_y)],
        eval_metric="rmse",
        callbacks=[],
    )
    preds_log_lgbm = lgbm.predict(valid_X_preprocessed)
    rmse_log_lgbm = float(np.sqrt(mean_squared_error(valid_y, preds_log_lgbm)))
    print(f"[LightGBM] RMSE on log1p scale (≈ RMSLE): {rmse_log_lgbm:.4f}")

    # XGBoost
    xgb = XGBRegressor(
        n_estimators=5000,
        learning_rate=0.03,
        random_state=42,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        n_jobs=-1,
        tree_method="hist",
    )

    xgb.fit(
        train_X_preprocessed,
        train_y,
        eval_set=[(valid_X_preprocessed, valid_y)],
        verbose=False,
    )

    preds_log_xgb = xgb.predict(valid_X_preprocessed)
    rmse_log_xgb = float(np.sqrt(mean_squared_error(valid_y, preds_log_xgb)))
    print(f"[XGBoost] RMSE on log1p scale (≈ RMSLE): {rmse_log_xgb:.4f}")
