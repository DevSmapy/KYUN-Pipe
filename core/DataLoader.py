import pandas as pd
from typing import Optional, Tuple
import logging
from pathlib import Path

# 모듈별 로거 생성
logger = logging.getLogger(__name__)


class DataLoader:
    """
    A universal data loader for Kaggle competitions.
    Handles standard CSV loading and basic path validation.
    """

    def __init__(self, train_path: str, test_path: Optional[str] = None):
        self.train_path = Path(train_path)
        self.test_path = Path(test_path) if test_path else None
        self.train_df: Optional[pd.DataFrame] = None
        self.test_df: Optional[pd.DataFrame] = None

    def load_data(self, **kwargs) -> None:
        """
        Loads training and test data from CSV files.
        """
        if not self.train_path.exists():
            logger.error(f"Train file missing: {self.train_path}")
            raise FileNotFoundError(f"Train file not found at: {self.train_path}")

        self.train_df = self._read_file(self.train_path, **kwargs)

        if self.test_path:
            if self.test_path.exists():
                self.test_df = self._read_file(self.test_path, **kwargs)
                logger.info(
                    f"Successfully loaded Train {self.train_df.shape} and Test {self.test_df.shape}"
                )
            else:
                logger.warning(
                    f"Test path provided but file not found: {self.test_path}"
                )
                logger.info(f"Only Train data loaded: {self.train_df.shape}")
        else:
            logger.info(f"Train data loaded: {self.train_df.shape}")

    def _read_file(self, path: Path, **kwargs) -> pd.DataFrame:
        """Helper to read files based on extension."""
        logger.debug(f"Reading file: {path.name}")
        if path.suffix == ".csv":
            return pd.read_csv(path, **kwargs)
        elif path.suffix in [".parquet", ".pqt"]:
            return pd.read_parquet(path, **kwargs)
        else:
            raise ValueError(f"Unsupported format: {path.suffix}")

    def get_data(
        self, copy: bool = True
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        if self.train_df is None:
            logger.error("Attempted to get data before loading.")
            raise RuntimeError("Data not loaded. Call load_data() first.")

        if copy:
            return self.train_df.copy(), (
                self.test_df.copy() if self.test_df is not None else None
            )
        return self.train_df, self.test_df
