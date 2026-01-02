import logging
from typing import List, Optional, Tuple, Any
import pandas as pd
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


class UniversalPreprocessor:
    """
    A modular preprocessor wrapper that leverages sklearn.pipeline.Pipeline.
    This design focuses on reusability and preventing data leakage by
    separating fitting (train) and transformation (test).
    """

    def __init__(self, steps: List[Tuple[str, Any]]):
        """
        Args:
            steps (List[Tuple[str, Any]]): List of (name, transformer) tuples
                compatible with sklearn's Pipeline. Transformers should
                implement fit() and transform() methods.
        """
        self.pipeline = Pipeline(steps)

    def run(
        self, train_df: pd.DataFrame, test_df: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
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
