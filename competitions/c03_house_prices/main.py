import logging
import sys

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

module_path = "/kaggle/input/pipe-core"
if module_path not in sys.path:
    sys.path.append(module_path)

from DataLoader import DataLoader  # noqa: E402
from Preprocessor import DataSplitter, UniversalPreprocessor  # noqa: E402
from Reporter import ResultReporter  # noqa: E402
from Trainer import ModelTrainer  # noqa: E402
from Transformers import (  # noqa: E402
    NumericCaster,
    OrdinalMapper,
    ValueConverter,
    ValueEncoder,
)

logger = logging.getLogger(__name__)

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


class NoneFiller(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.fillna("None")


def evaluate_log_rmse(
    y_valid_log: pd.Series, preds_log: np.ndarray, model_name: str
) -> dict[str, float]:
    rmse_log = float(np.sqrt(mean_squared_error(y_valid_log, preds_log)))
    print(f"[{model_name}] RMSE on log1p scale (≈ RMSLE): {rmse_log:.5f}")

    y_valid_orig = np.expm1(y_valid_log)
    preds_orig = np.expm1(preds_log)
    rmse_orig = float(np.sqrt(mean_squared_error(y_valid_orig, preds_orig)))
    print(f"[{model_name}] RMSE on original scale: {rmse_orig:.2f}")

    return {"rmse_log": rmse_log, "rmse_orig": rmse_orig}


if __name__ == "__main__":
    # 0) Reporter (artifacts)
    reporter = ResultReporter(base_path=".")

    # 1. Load and get data
    DATA_DIR = "/kaggle/input/house-prices-advanced-regression-techniques"
    dataloader = DataLoader(
        train_path=f"{DATA_DIR}/train.csv", test_path=f"{DATA_DIR}/test.csv"
    )
    dataloader.load_data()
    train, test = dataloader.get_data()

    # 2. Split data into train and validation sets(for protection dataleakage)
    train_set, valid_set = DataSplitter.split_random(
        train, test_size=0.3, random_state=42
    )

    # 3. Target transform (baseline_script.py와 동일)
    y_train = np.log1p(train_set["SalePrice"])
    y_valid = np.log1p(valid_set["SalePrice"])

    X_train_df = train_set.drop(columns=["SalePrice"])
    X_valid_df = valid_set.drop(columns=["SalePrice"])
    X_test_df = test

    # 4. Set up preprocessing pipeline (baseline_script.py 흐름 반영)
    steps = [
        ("value_converter", ValueConverter(fill_object_na_with="None")),
        ("ordinal_encoder", OrdinalMapper(ORDINAL_MAPS)),
        (
            "onehot_encoder",
            ValueEncoder(
                encoder=OneHotEncoder(
                    sparse_output=False,
                    handle_unknown="ignore",
                ).set_output(transform="pandas")
            ),
        ),
        ("numeric_caster", NumericCaster(dtype="float32")),
    ]

    preprocessor = UniversalPreprocessor(steps)

    # train: fit_transform, test: transform
    X_train, X_test = preprocessor.run(X_train_df, X_test_df)

    # valid: transform only (run()을 다시 호출하면 valid로 fit이 다시 발생함)
    X_valid = preprocessor.get_pipeline().transform(X_valid_df)

    # 5) Train & Evaluate (OOP)
    models = [
        ("RandomForest", RandomForestRegressor(random_state=42)),
        (
            "LightGBM",
            LGBMRegressor(
                n_estimators=3000,
                learning_rate=0.03,
                random_state=42,
                subsample=0.8,
                colsample_bytree=0.8,
                n_jobs=-1,
            ),
        ),
        (
            "XGBoost",
            XGBRegressor(
                n_estimators=5000,
                learning_rate=0.03,
                random_state=42,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                n_jobs=-1,
                tree_method="hist",
            ),
        ),
    ]

    metrics: dict[str, dict[str, float]] = {}
    best_name: str | None = None
    best_trainer: ModelTrainer | None = None
    best_rmse_log = float("inf")

    for name, model in models:
        trainer = ModelTrainer(model=model, model_name=name)

        fit_params: dict[str, object] = {}
        if name == "LightGBM":
            fit_params = {
                "eval_metric": "rmse",
            }
        elif name == "XGBoost":
            fit_params = {
                "verbose": False,
            }

        trainer.train(X_train, y_train, X_valid=X_valid, y_valid=y_valid, **fit_params)

        preds_log = trainer.predict(X_valid).to_numpy()
        metrics[name] = evaluate_log_rmse(y_valid, preds_log, name)

        if metrics[name]["rmse_log"] < best_rmse_log:
            best_rmse_log = metrics[name]["rmse_log"]
            best_name = name
            best_trainer = trainer

    # 6) Save artifacts (metrics / metadata / best model / submission)
    reporter.save_metrics(metrics)

    metadata = {
        "competition": "c03_house_prices",
        "target": "SalePrice",
        "target_transform": "log1p",
        "validation": {"type": "holdout", "test_size": 0.3, "random_state": 42},
        "preprocessing_steps": [name for name, _ in steps],
        "train_shape": [int(X_train.shape[0]), int(X_train.shape[1])],
        "valid_shape": [int(X_valid.shape[0]), int(X_valid.shape[1])],
        "best_model": best_name,
        "best_rmse_log": float(best_rmse_log),
    }
    reporter.save_metadata(metadata)

    if best_trainer is not None and best_name is not None:
        reporter.save_model(best_trainer.get_model(), name=best_name)

        if X_test is not None and test is not None:
            preds_test_log = best_trainer.predict(X_test).to_numpy()
            preds_test = np.expm1(preds_test_log)

            submission = pd.DataFrame(
                {
                    "Id": test["Id"].to_numpy(),
                    "SalePrice": preds_test,
                }
            )
            reporter.save_predictions(submission, name=f"submission_{best_name}")
