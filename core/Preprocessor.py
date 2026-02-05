import logging
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


class UniversalPreprocessor:
    """
    A modular preprocessor wrapper that leverages sklearn.pipeline.Pipeline.
    This design focuses on reusability and preventing data leakage by
    separating fitting (train) and transformation (test).
    """

    def __init__(self, steps: list[tuple[str, Any]]):
        """
        Args:
            steps (List[Tuple[str, Any]]): List of (name, transformer) tuples
                compatible with sklearn's Pipeline. Transformers should
                implement fit() and transform() methods.
        """
        self.pipeline = Pipeline(steps)

    def run(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame | None = None
    ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        """
        Executes the preprocessing pipeline.

        Args:
            train_df (pd.DataFrame): Training data to fit and transform.
            test_df (Optional[pd.DataFrame]): Test data to transform using parameters from train_df.

        Returns:
            Tuple[pd.DataFrame, Optional[pd.DataFrame]]: Preprocessed (train, test) DataFrames.
        """
        logger.info(
            f"Initiating preprocessing pipeline with {len(self.pipeline.steps)} steps..."
        )

        # Train: fit + transfrom
        train_processed = self.pipeline.fit_transform(train_df)
        logger.info(f"Train data processed. Final shape: {train_processed.shape}")

        test_processed = None
        if test_df is not None:
            # Test: only transform
            logger.info(
                "Transforming test data using statistics learned from training set."
            )
            test_processed = self.pipeline.transform(test_df)
            logger.info(f"Test data processed. Final shape: {test_processed.shape}")

        logger.info("Universal Preprocessor execution completed.")
        return train_processed, test_processed

    def get_pipeline(self) -> Pipeline:
        """Returns the underlying sklearn Pipeline object."""
        return self.pipeline


class DataSplitter:
    """
    Provides various splitting strategies for train-test splits.
    """

    @staticmethod
    def split_by_date(
        df: pd.DataFrame, split_date: str
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Used for Time Series: keeps the order of time
        """
        logger.info(f"Splitting data by date: {split_date}")
        train = df[df.date < split_date].copy()
        valid = df[df.date >= split_date].copy()
        return train, valid

    @staticmethod
    def split_random(
        df: pd.DataFrame, test_size: float = 0.3, random_state: int = 42
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Used for general tasks
        """
        logger.info(f"Splitting data randomly with test size: {test_size}")
        train, valid = train_test_split(
            df, test_size=test_size, random_state=random_state
        )
        return train, valid
