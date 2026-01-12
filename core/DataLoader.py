import pandas as pd
from typing import Optional, Tuple, Dict, List
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class DataLoader:
    """
    A universal data loader for Kaggle competitions.
    Handles standard CSV loading and additional context tables.
    """

    def __init__(self, train_path: str, test_path: str | None = None):
        self.train_path = Path(train_path)
        self.test_path = Path(test_path) if test_path else None
        self.train_df: Optional[pd.DataFrame] = None
        self.test_df: Optional[pd.DataFrame] = None
        self.additional_data: Dict[str, pd.DataFrame] = {}

    def load_data(self, **kwargs) -> None:
        """Loads core training and test data."""
        if not self.train_path.exists():
            raise FileNotFoundError(f"Train file not found at: {self.train_path}")

        self.train_df = self._read_file(self.train_path, **kwargs)

        if self.test_path and self.test_path.exists():
            self.test_df = self._read_file(self.test_path, **kwargs)
            logger.info(
                f"Loaded Core: Train {self.train_df.shape}, Test {self.test_df.shape}"
            )

    def load_additional_data(
        self, directory_path: str, exclude: List[str] = None
    ) -> None:
        """
        Scans a directory and loads all CSV files except core files into a dictionary.
        """
        dir_path = Path(directory_path)
        exclude = exclude or ["train", "test", "sample_submission"]

        logger.info(f"Scanning directory: {directory_path} for additional tables...")
        for file in dir_path.glob("*.csv"):
            name = file.stem
            if name not in exclude:
                self.additional_data[name] = self._read_file(file)
                logger.info(
                    f"Loaded additional table: {name} ({self.additional_data[name].shape})"
                )

    def get_data(
        self, copy: bool = True
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        if self.train_df is None:
            raise RuntimeError("Call load_data() before getting data.")
        return (
            (self.train_df.copy(), self.test_df.copy())
            if copy
            else (self.train_df, self.test_df)
        )

    def get_additional_data(self) -> Dict[str, pd.DataFrame]:
        """Returns the dictionary of loaded context tables."""
        return self.additional_data

    def _read_file(self, path: Path, **kwargs) -> pd.DataFrame:
        if path.suffix == ".csv":
            return pd.read_csv(path, **kwargs)
        elif path.suffix in [".parquet", ".pqt"]:
            return pd.read_parquet(path, **kwargs)
        else:
            raise ValueError(f"Unsupported format: {path.suffix}")
