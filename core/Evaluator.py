import logging
from typing import Dict, Any, List
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Handles evaluation of machine learning models.
    Provides metrics calculation and model analysis tools.
    """

    def __init__(self, task_type: str = "classification"):
        """
        Args:
            task_type (str): Type of ML task ('classification' or 'regression').
        """
        self.task_type = task_type
        self.results: Dict[str, Any] = {}

    def evaluate(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """
        General evaluation method based on the task type.
        """
        if self.task_type == "classification":
            return self._evaluate_classification(y_true, y_pred)
        else:
            logger.warning(f"Task type '{self.task_type}' is not yet implemented.")
            return {}

    def _evaluate_classification(
        self, y_true: pd.Series, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculates detailed classification metrics.
        """
        logger.info("Calculating classification metrics...")

        self.results = {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1_score": f1_score(y_true, y_pred, average="binary"),
            "precision": precision_score(y_true, y_pred, average="binary"),
            "recall": recall_score(y_true, y_pred, average="binary"),
        }

        # make summary report
        logger.info("--- [Classification Report] ---")
        for name, value in self.results.items():
            logger.info(f"{name:10}: {value:.4f}")

        # make detailed report
        logger.debug(f"\n{classification_report(y_true, y_pred)}")

        return self.results

    def get_confusion_matrix(self, y_true: pd.Series, y_pred: np.ndarray) -> np.ndarray:
        """Returns the confusion matrix."""
        return confusion_matrix(y_true, y_pred)

    def analyze_feature_importance(
        self, model: Any, feature_names: List[str], top_n: int = 10
    ):
        """
        Extracts and logs feature importance from the trained model.
        Works with tree-based models (RandomForest, XGBoost, etc.)
        """
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            feature_imp = pd.DataFrame(
                {"Feature": feature_names, "Importance": importances}
            ).sort_values(by="Importance", ascending=False)

            logger.info(f"--- [Top {top_n} Feature Importance] ---")
            for i, row in feature_imp.head(top_n).iterrows():
                logger.info(f"{row['Feature']:20}: {row['Importance']:.4f}")

            return feature_imp
        else:
            logger.warning("The provided model does not support feature_importances_.")
            return None
