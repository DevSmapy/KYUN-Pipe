import logging
from typing import Any, Dict, Optional
import pandas as pd
from sklearn.model_selection import GridSearchCV

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    A universal trainer that supports standard training and hyperparameter tuning.
    """

    def __init__(self, model: Any, model_name: str = "BaseModel"):
        self.model = model
        self.model_name = model_name
        self.best_params: Optional[Dict[str, Any]] = None
        self.is_tuned = False

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        param_grid: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Trains the model. If param_grid is provided, it performs GridSearchCV.
        """
        if param_grid:
            logger.info(
                f"[{self.model_name}] Starting Hyperparameter Tuning (GridSearch)..."
            )
            grid_search = GridSearchCV(
                self.model, param_grid, cv=5, scoring="f1", n_jobs=-1
            )
            grid_search.fit(X_train, y_train)

            self.model = grid_search.best_estimator_
            self.best_params = grid_search.best_params_
            self.is_tuned = True

            logger.info(f"[{self.model_name}] Best Tuning Completed! ✅")
            logger.info(f"[{self.model_name}] Best Parameters: {self.best_params}")
        else:
            logger.info(f"[{self.model_name}] Training with default/set parameters...")
            self.model.fit(X_train, y_train)
            logger.info(f"[{self.model_name}] Training completed.")

    def get_model_info(self) -> Dict[str, Any]:
        """
        Provides model information, including whether it was tuned and its current parameters.
        """
        info = {
            "model_name": self.model_name,
            "is_tuned": self.is_tuned,
            "current_params": self.model.get_params(),
        }
        if self.is_tuned:
            info["best_params"] = self.best_params

        logger.info(f"--- Model Info: {self.model_name} ---")
        logger.info(
            f"Tuning applied: {'Yes (Best Tuning! 🚀)' if self.is_tuned else 'No (Default)'}"
        )
        return info

    def predict(self, X: pd.DataFrame) -> pd.Series:
        logger.info(f"[{self.model_name}] Generating predictions...")
        preds = self.model.predict(X)
        return pd.Series(preds)

    def get_model(self) -> Any:
        return self.model
