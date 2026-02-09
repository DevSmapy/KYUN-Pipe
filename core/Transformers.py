# core/Transformers.py
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import OneHotEncoder


class FeatureDropper(BaseEstimator, TransformerMixin):
    """Drops specified columns from the DataFrame (ignores missing ones)."""

    def __init__(self, columns: list[str]):
        self.columns = columns

    def fit(self, X: pd.DataFrame, y: Any = None):
        return self

    def transform(self, X: pd.DataFrame):
        X_out = X.copy()
        cols = [c for c in self.columns if c in X_out.columns]
        return X_out.drop(columns=cols)


class MissingValueFiller(BaseEstimator, TransformerMixin):
    """
    Imputes missing values:
    - numeric: KNN imputation (or any sklearn imputer you pass)
    - categorical: most_frequent (or any sklearn imputer you pass)

    Note:
    - fit() learns statistics ONLY from training set
    - transform() applies learned imputers (no refit)
    """

    def __init__(
        self,
        numeric_imputer: Any | None = None,
        categorical_imputer: Any | None = None,
    ):
        self.numeric_imputer = numeric_imputer or KNNImputer(n_neighbors=5)
        self.categorical_imputer = categorical_imputer or SimpleImputer(
            strategy="most_frequent"
        )

    def fit(self, X: pd.DataFrame, y: Any = None):
        self.num_columns_ = X.select_dtypes(include=["number"]).columns
        self.cat_columns_ = X.select_dtypes(include=["object", "category"]).columns

        if len(self.num_columns_) > 0:
            self.numeric_imputer.fit(X[self.num_columns_])
        if len(self.cat_columns_) > 0:
            self.categorical_imputer.fit(X[self.cat_columns_])

        return self

    def transform(self, X: pd.DataFrame):
        X_out = X.copy()

        if len(self.num_columns_) > 0:
            X_out.loc[:, self.num_columns_] = self.numeric_imputer.transform(
                X_out[self.num_columns_]
            )
        if len(self.cat_columns_) > 0:
            X_out.loc[:, self.cat_columns_] = self.categorical_imputer.transform(
                X_out[self.cat_columns_]
            )

        return X_out


class ValueConverter(BaseEstimator, TransformerMixin):
    """
    Utility transformer:
    - fill missing tokens for categorical columns (e.g., NaN -> "None")
    - optionally cast selected dtype columns to a target dtype (e.g., object -> category)
    """

    def __init__(
        self,
        fill_object_na_with: str | None = None,
        cast_object_to_category: bool = False,
        none_tokens: Iterable[str] = ("NA", "nan", ""),
    ):
        self.fill_object_na_with = fill_object_na_with
        self.cast_object_to_category = cast_object_to_category
        self.none_tokens = set(none_tokens)

    def fit(self, X: pd.DataFrame, y: Any = None):
        return self

    def transform(self, X: pd.DataFrame):
        X_out = X.copy()

        obj_cols = X_out.select_dtypes(include=["object"]).columns
        if len(obj_cols) > 0 and self.fill_object_na_with is not None:
            for c in obj_cols:
                s = (
                    X_out[c]
                    .fillna(self.fill_object_na_with)
                    .astype(str)
                    .str.strip()
                    .replace(
                        {tok: self.fill_object_na_with for tok in self.none_tokens}
                    )
                )
                X_out[c] = s

        if self.cast_object_to_category and len(obj_cols) > 0:
            X_out.loc[:, obj_cols] = X_out.loc[:, obj_cols].astype("category")

        return X_out


class OrdinalMapper(BaseEstimator, TransformerMixin):
    """
    Domain-driven ordinal encoding for specific columns.
    - maps: dict[column_name, dict[label, int]]
    - unknown_policy: "error" or "use_default"
    """

    def __init__(
        self,
        maps: dict[str, dict[str, int]],
        unknown_policy: str = "error",
        default_value: int = 0,
    ):
        self.maps = maps
        self.unknown_policy = unknown_policy
        self.default_value = default_value

    def fit(self, X: pd.DataFrame, y: Any = None):
        return self

    def transform(self, X: pd.DataFrame):
        X_out = X.copy()

        for col, mapping in self.maps.items():
            if col not in X_out.columns:
                continue

            s = X_out[col].fillna("None").astype(str).str.strip()
            mapped = s.map(mapping)

            if mapped.isna().any():
                if self.unknown_policy == "use_default":
                    mapped = mapped.fillna(self.default_value)
                else:
                    unknown = sorted(set(s[mapped.isna()].unique()))
                    raise ValueError(
                        f"[OrdinalMapper] {col}: unknown labels: {unknown}"
                    )

            X_out[col] = mapped.astype("int16")

        return X_out


class ValueEncoder(BaseEstimator, TransformerMixin):
    """
    Encodes categorical features with OneHotEncoder and returns a DataFrame.
    - Detects object/category columns at fit time and encodes them.
    """

    def __init__(self, encoder: Any | None = None):
        self.encoder = encoder or OneHotEncoder(
            sparse_output=False, handle_unknown="ignore"
        )
        self.feature_names_: np.ndarray | None = None
        self.cat_columns_: pd.Index | None = None

    def fit(self, X: pd.DataFrame, y: Any = None):
        X_out = X.copy()
        self.cat_columns_ = X_out.select_dtypes(include=["object", "category"]).columns

        if len(self.cat_columns_) == 0:
            return self

        # ✅ mixed type(str/float) 방지: categorical 입력을 항상 string으로 통일
        X_cat = X_out.loc[:, self.cat_columns_].fillna("None").astype("string")

        self.encoder.fit(X_cat)

        if hasattr(self.encoder, "get_feature_names_out"):
            self.feature_names_ = self.encoder.get_feature_names_out(self.cat_columns_)
        else:
            self.feature_names_ = None

        return self

    def transform(self, X: pd.DataFrame):
        X_out = X.copy()

        if self.cat_columns_ is None or len(self.cat_columns_) == 0:
            return X_out

        # ✅ fit과 동일한 전처리 적용
        X_cat = X_out.loc[:, self.cat_columns_].fillna("None").astype("string")
        encoded = self.encoder.transform(X_cat)

        if self.feature_names_ is not None:
            encoded_df = pd.DataFrame(
                encoded, columns=self.feature_names_, index=X_out.index
            )
        else:
            encoded_df = pd.DataFrame(
                encoded, columns=self.cat_columns_, index=X_out.index
            )

        X_out = X_out.drop(columns=self.cat_columns_)
        return pd.concat([X_out, encoded_df], axis=1)


class NumericCaster(BaseEstimator, TransformerMixin):
    """Force all columns to numeric dtype (useful for LGBM/XGB stability)."""

    def __init__(self, dtype: str = "float32"):
        # mypy/pandas stubs friendly: store parsed dtype instead of raw str
        self.dtype = np.dtype(dtype)

    def fit(self, X: pd.DataFrame, y: Any = None):
        return self

    def transform(self, X: pd.DataFrame):
        X_out = X.copy()
        X_out = X_out.replace([np.inf, -np.inf], np.nan).fillna(0)
        return X_out.astype(self.dtype)
