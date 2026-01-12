import os
from functools import reduce

import lightgbm as lgb
import numpy as np
import pandas as pd
import seaborn as sns
from lightgbm import LGBMRegressor  # Scikit-learn API for LightGBM
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error

if __name__ == "__main__":
    # ==========================================
    # 1. Data Loading
    # ==========================================
    data_dict = {}
    input_path = "/kaggle/input/store-sales-time-series-forecasting"

    # Walk through the directory and load all .csv files into a dictionary
    for dirname, _, filenames in os.walk(input_path):
        for filename in filenames:
            if filename.endswith(".csv"):
                name = filename.split(".")[0]
                data_dict[name] = pd.read_csv(os.path.join(dirname, filename))

    # ==========================================
    # 2. Date Preprocessing & Holiday Events
    # ==========================================
    # Convert 'date' column to datetime objects for the main training set
    train = data_dict["train"]
    train["date"] = pd.to_datetime(train["date"])

    # Preprocess Holidays and Events
    he = data_dict["holidays_events"]
    he["date"] = pd.to_datetime(he["date"])

    # Define significant holiday types that affect sales
    holiday_types = ["Holiday", "Transfer", "Additional", "Bridge"]

    # Generate Holiday Flag: specific types that were NOT transferred
    he["is_holiday"] = ((he["type"].isin(holiday_types)) & (~he["transferred"])).astype(
        int
    )
    # Generate Work Day Flag: weekends that are officially designated as working days
    he["is_workday"] = (he["type"] == "Work Day").astype(int)

    # Aggregate by date: Take the maximum flag value in case of multiple events on the same day
    he = he.groupby("date")[["is_holiday", "is_workday"]].max().reset_index()
    data_dict["holidays_events"] = he

    # ==========================================
    # 3. Oil Price Imputation (Interpolation)
    # ==========================================
    oil = data_dict["oil"]
    oil["date"] = pd.to_datetime(oil["date"])

    # Ensure time series continuity by reindexing with a full date range (fills missing weekends)
    all_dates = pd.date_range(start=oil["date"].min(), end=oil["date"].max(), freq="D")
    oil = oil.set_index("date").reindex(all_dates).rename_axis("date").reset_index()

    # Apply Linear Interpolation to fill gaps in oil prices based on temporal flow
    oil["dcoilwtico"] = oil["dcoilwtico"].interpolate(
        method="linear", limit_direction="both"
    )
    # Use forward/backward fill for remaining NaNs at the boundaries
    oil["dcoilwtico"] = oil["dcoilwtico"].ffill().bfill()
    data_dict["oil"] = oil

    # ==========================================
    # 4. Store Profiling & Statistics
    # ==========================================
    transactions = data_dict["transactions"]
    stores = data_dict["stores"]

    # Calculate average and standard deviation of transactions per store (indicates store scale/activity)
    store_stats = (
        transactions.groupby("store_nbr")["transactions"]
        .agg(["mean", "std"])
        .reset_index()
    )
    store_stats.columns = pd.Index(
        ["store_nbr", "avg_transactions", "std_transactions"]
    )

    # Merge store metadata (location, type) with statistical features
    store_profile = pd.merge(stores, store_stats, on="store_nbr", how="left")

    # ==========================================
    # 5. Final Training Data Merging
    # ==========================================
    # Use reduce to merge train, oil, and holidays sequentially on 'date'
    train_merged = reduce(
        lambda x, y: pd.merge(x, y, on="date", how="left"),
        [train, data_dict["oil"], data_dict["holidays_events"]],
    )

    # Merge store-specific information
    final_train = pd.merge(train_merged, store_profile, on="store_nbr", how="left")

    # Fill NaNs for holiday flags and generate date-derived features
    final_train["is_holiday"] = final_train["is_holiday"].fillna(0)
    final_train["is_workday"] = final_train["is_workday"].fillna(0)
    final_train["year"] = final_train.date.dt.year
    final_train["month"] = final_train.date.dt.month
    final_train["dayofweek"] = final_train.date.dt.dayofweek
    final_train["is_weekend"] = (final_train["dayofweek"] >= 5).astype(int)

    # ==========================================
    # 6. Test Data Preprocessing
    # ==========================================
    test = data_dict["test"]
    test["date"] = pd.to_datetime(test["date"])

    # Mirror the training data preprocessing logic for the test set
    test_merged = reduce(
        lambda x, y: pd.merge(x, y, on="date", how="left"),
        [test, data_dict["oil"], data_dict["holidays_events"]],
    )
    final_test = pd.merge(test_merged, store_profile, on="store_nbr", how="left")

    # Process test set date features and fill NaNs
    final_test["is_holiday"] = final_test["is_holiday"].fillna(0)
    final_test["is_workday"] = final_test["is_workday"].fillna(0)
    final_test["dayofweek"] = final_test.date.dt.dayofweek
    final_test["is_weekend"] = (final_test["dayofweek"] >= 5).astype(int)

    # ==========================================
    # 7. Time Series Feature Engineering (Lag & Rolling)
    # ==========================================
    # Sort data to ensure correct shifting of time series features
    final_train = final_train.sort_values(["store_nbr", "family", "date"])

    # Create Lag Features: 16-day shift is used as a safety margin for the 16-day forecast period
    final_train["sales_lag_16"] = final_train.groupby(["store_nbr", "family"])[
        "sales"
    ].transform(lambda x: x.shift(16))
    final_train["sales_lag_30"] = final_train.groupby(["store_nbr", "family"])[
        "sales"
    ].transform(lambda x: x.shift(30))

    # Rolling Mean: Captures the average sales trend over the last 7 days (shifted by 16)
    final_train["sales_roll_mean_7"] = final_train.groupby(["store_nbr", "family"])[
        "sales"
    ].transform(lambda x: x.shift(16).rolling(window=7).mean())

    # Fill NaNs caused by shifting with 0
    final_train = final_train.fillna(0)

    # ==========================================
    # 8. Modeling Preparation
    # ==========================================
    # Convert object columns to 'category' type for LightGBM's native handling
    cat_cols = final_train.select_dtypes("object").columns
    for col in cat_cols:
        final_train[col] = final_train[col].astype("category")

    # Time-based Validation Split (Using the last 15 days as the validation set)
    split_date = "2017-08-01"
    train_data = final_train[final_train["date"] < split_date]
    valid_data = final_train[final_train["date"] >= split_date]

    # Select final features for training
    features = [
        "store_nbr",
        "family",
        "onpromotion",
        "dcoilwtico",
        "is_holiday",
        "is_workday",
        "dayofweek",
        "sales_lag_16",
        "sales_roll_mean_7",
        "city",
        "type",
        "cluster",
    ]

    # Target Variable Transformation: Apply Log transformation to optimize for RMSLE
    X_train = train_data[features]
    y_train = np.log1p(train_data["sales"])
    X_valid = valid_data[features]
    y_valid = np.log1p(valid_data["sales"])

    # ==========================================
    # 9. Model Training & Evaluation
    # ==========================================
    model = LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        random_state=42,
        importance_type="gain",  # Use 'gain' to measure the total information gain of a feature
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="rmse",
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=50
            ),  # Stop if validation score doesn't improve for 50 rounds
            lgb.log_evaluation(period=100),  # Print results every 100 iterations
        ],
    )

    # ==========================================
    # 10. Visualization (Feature Importance)
    # ==========================================
    feature_imp = pd.DataFrame(
        {"Value": model.feature_importances_, "Feature": features}
    )
    plt.figure(figsize=(10, 7))
    sns.barplot(
        x="Value",
        y="Feature",
        data=feature_imp.sort_values(by="Value", ascending=False),
    )
    plt.title("LightGBM Feature Importance (Gain)")
    plt.tight_layout()
    plt.show()

    # Final Validation Score
    y_pred_log = model.predict(X_valid)
    rmsle = np.sqrt(mean_squared_error(y_valid, y_pred_log))
    print(f"\n[Process Complete] Validation RMSLE: {rmsle:.4f}")
