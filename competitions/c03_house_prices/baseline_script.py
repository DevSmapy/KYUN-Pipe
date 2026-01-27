import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


def apply_quality_value_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """
    Column-wise ordinal encoding with:
    - safe handling for missing columns (skip if not present)
    - normalization of missing tokens (NaN/'NA'/'' -> 'None')
    - strict validation: raises if unknown labels appear for a mapped column
    """
    out = df.copy()

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

        # If any values weren't mapped, fail loudly so you can fix the mapping once (DE-friendly)
        if mapped.isna().any():
            unknown = sorted(set(s[mapped.isna()].unique()))
            raise ValueError(
                f"[apply_quality_value_encoding] {col}: unknown labels: {unknown}"
            )

        out[col] = mapped.astype("int8")

    return out


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

## impute categorical data (NaN is NA, so fillna "No")
train_cat = train_X[cat_columns].fillna("None")
valid_cat = valid_X[cat_columns].fillna("None")


## encode categorical data with LabelEncoder
# label_encoder = LabelEncoder()
qual_cols = train_cat.filter(like="Qual").columns

"""
for i in qual_cols:
    train_cat[i] = label_encoder.fit_transform(train_cat[i])
    valid_cat[i] = label_encoder.transform(valid_cat[i])
"""
train_cat[qual_cols] = apply_quality_value_encoding(train_cat[qual_cols])
valid_cat[qual_cols] = apply_quality_value_encoding(valid_cat[qual_cols])

## encode categorical data with OneHotEncoder
onehotencoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore").set_output(
    transform="pandas"
)
diff_cols = cat_columns.difference(train_cat.filter(like="Qual").columns)
train_oh_encoded = onehotencoder.fit_transform(train_cat[diff_cols])
valid_oh_encoded = onehotencoder.transform(valid_cat[diff_cols])

### concatenate
train_cat = pd.concat([train_oh_encoded, train_cat[qual_cols]], axis=1)
valid_cat = pd.concat([valid_oh_encoded, valid_cat[qual_cols]], axis=1)


## impute numeric data by using KNN
knn_imputer = KNNImputer()
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

# train model
RF = RandomForestRegressor()
RF.fit(train_X_preprocessed, train_y)

# evaluate model
preds_log = RF.predict(valid_X_preprocessed)

# 1) Kaggle 지표와 정렬되는 값: log-space RMSE (≈ RMSLE)
rmse_log = np.sqrt(mean_squared_error(valid_y, preds_log))
print("RMSE on log1p scale (≈ RMSLE):", rmse_log)

# 2) 참고용: 원 스케일 RMSE (원하면 같이 보기)
valid_y_orig = np.expm1(valid_y)
preds_orig = np.expm1(preds_log)
rmse_orig = np.sqrt(mean_squared_error(valid_y_orig, preds_orig))
print("RMSE on original scale:", rmse_orig)
