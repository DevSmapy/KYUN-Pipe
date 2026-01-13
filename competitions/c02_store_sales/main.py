import logging
import sys

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

module_path = "/kaggle/input/pipe-core"
if module_path not in sys.path:
    sys.path.append(module_path)

from DataLoader import DataLoader  # noqa: E402
from Preprocessor import UniversalPreprocessor  # noqa: E402

logger = logging.getLogger(__name__)


class HolidayChecker(BaseEstimator, TransformerMixin):
    """
    Definition is_holiday column & is_workday column
    """

    def __init__(self, holidays_df):
        self.holidays_df = holidays_df
        self.holiday_types = ["Holiday", "Transfer", "Additional", "Bridge"]
        self.processed_holidays = None

    def fit(self, X, y=None):
        he = self.holidays_df.copy()
        he["date"] = pd.to_datetime(he["date"])
        he["is_holiday"] = (
            (he["type"].isin(self.holiday_types)) & (~he["transferred"])
        ).astype(int)
        he["is_workday"] = (he["type"] == "Work Day").astype(int)
        self.processed_holidays = (
            he.groupby("date")[["is_holiday", "is_workday"]].max().reset_index()
        )
        return self

    def transform(self, df):
        X = df.copy()
        X["date"] = pd.to_datetime(X["date"])

        cols_to_drop = [c for c in ["is_holiday", "is_workday"] if c in X.columns]
        if cols_to_drop:
            X = X.drop(columns=cols_to_drop)

        X = pd.merge(X, self.processed_holidays, on="date", how="left")
        X[["is_holiday", "is_workday"]] = X[["is_holiday", "is_workday"]].fillna(0)
        return X


class OilPriceImputer(BaseEstimator, TransformerMixin):
    """
    Oil price preprocessing: Reindexing to fill missing dates and Linear interpolation.
    """

    def __init__(self, oil_df):
        self.oil_df = oil_df
        self.processed_oil = None

    def fit(self, X, y=None):
        oil = self.oil_df.copy()
        oil["date"] = pd.to_datetime(oil["date"])
        all_dates = pd.date_range(
            start=oil["date"].min(), end=oil["date"].max(), freq="D"
        )
        oil = oil.set_index("date").reindex(all_dates).rename_axis("date").reset_index()
        oil["dcoilwtico"] = (
            oil["dcoilwtico"]
            .interpolate(method="linear", limit_direction="both")
            .ffill()
            .bfill()
        )
        self.processed_oil = oil
        return self

    def transform(self, df):
        X = df.copy()
        X["date"] = pd.to_datetime(X["date"])

        if "dcoilwtico" in X.columns:
            X = X.drop(columns=["dcoilwtico"])

        return pd.merge(X, self.processed_oil, on="date", how="left")


class StoreStatsMerger(BaseEstimator, TransformerMixin):
    """
    Store statistics preprocessing: Average and standard deviation of sales per store.
    """

    def __init__(self, stores_df, transactions_df):
        self.stores_df = stores_df
        self.transactions_df = transactions_df
        self.store_profile = None

    def fit(self, X, y=None):
        store_stats = (
            self.transactions_df.groupby("store_nbr")["transactions"]
            .agg(["mean", "std"])
            .reset_index()
        )
        store_stats.columns = pd.Index(
            ["store_nbr", "avg_transactions", "std_transactions"]
        )
        self.store_profile = pd.merge(
            self.stores_df, store_stats, on="store_nbr", how="left"
        )
        return self

    def transform(self, df):
        X = df.copy()

        profile_cols = [c for c in self.store_profile.columns if c != "store_nbr"]
        X = X.drop(columns=[c for c in profile_cols if c in X.columns])

        return pd.merge(X, self.store_profile, on="store_nbr", how="left")


class DateTimeFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extract date-time features from the 'date' column.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, df):
        X = df.copy()
        X["date"] = pd.to_datetime(X["date"])

        # day_of_week 이름 통일 (기존 baseline_script 스타일)
        X["year"] = X["date"].dt.year
        X["month"] = X["date"].dt.month
        X["dayofweek"] = X["date"].dt.dayofweek
        X["is_weekend"] = (X["dayofweek"] >= 5).astype(int)

        return X


class TimeSeriesWindowFeaturizer(BaseEstimator, TransformerMixin):
    """
    Create time series window features for each store.
    """

    def __init__(self, lags=[16, 30], window_size=7):
        self.lags = lags
        self.window_size = window_size

    def fit(self, X, y=None):
        return self

    def transform(self, df):
        X = df.copy()

        if "sales" not in X.columns:
            return X

        X = X.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)

        for lag in self.lags:
            X[f"sales_lag_{lag}"] = X.groupby(["store_nbr", "family"])[
                "sales"
            ].transform(lambda x: x.shift(lag))

        # Rolling Mean: Captures the average sales trend over the last 7 days (shifted by 16)
        X[f"sales_roll_mean_{self.window_size}"] = X.groupby(["store_nbr", "family"])[
            "sales"
        ].transform(lambda x: x.shift(16).rolling(window=7).mean())

        # Fill NaNs caused by shifting with 0
        return X.fillna(0)


# main.py execution block example
if __name__ == "__main__":
    # 1. Initialize and Load All Data
    DATA_DIR = "/kaggle/input/store-sales-time-series-forecasting"

    dataloader = DataLoader(
        train_path=f"{DATA_DIR}/train.csv", test_path=f"{DATA_DIR}/test.csv"
    )
    dataloader.load_data()

    # Load oil, holidays_events, stores, transactions automatically
    dataloader.load_additional_data(directory_path=DATA_DIR)

    train, test = dataloader.get_data()
    context_data = (
        dataloader.get_additional_data()
    )  # Returns dict: {'oil': df, 'stores': df, ...}
    print(context_data.keys())

    steps = [
        ("holiday_checker", HolidayChecker(context_data["holidays_events"])),
        ("oil_price_imputer", OilPriceImputer(context_data["oil"])),
        (
            "store_stats_merger",
            StoreStatsMerger(context_data["stores"], context_data["transactions"]),
        ),
        ("datetime_feature_extractor", DateTimeFeatureExtractor()),
        ("time_series_window_featurizer", TimeSeriesWindowFeaturizer()),
    ]

    preprocessor = UniversalPreprocessor(steps)
    train_X, test_X = preprocessor.run(train, test)

    print(train_X.head())
    print(train_X.info())

    print(test_X.head())
    print(test_X.info())
