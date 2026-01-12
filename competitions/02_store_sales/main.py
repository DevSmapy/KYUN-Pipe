import sys

import logging

module_path = "/kaggle/input/pipe-core"
if module_path not in sys.path:
    sys.path.append(module_path)

from DataLoader import DataLoader  # noqa: E402

logger = logging.getLogger(__name__)

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
