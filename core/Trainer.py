import logging
from typing import Any

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
        self.best_params: dict[str, Any] | None = None
        self.is_tuned = False

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.Series | None = None,
        param_grid: dict[str, Any] | None = None,
        scoring: str = "neg_root_mean_squared_error",
        cv: Any = None,  # None이면 Hold-out, 숫자가 들어오면 K-Fold, TimeSeriesSplit 객체도 가능
        **fit_params,  # LGBM의 early_stopping 등을 위한 추가 인자
    ) -> None:
        """
        Trains the model with support for Hold-out or CV.
        """
        if param_grid and cv is not None:
            # 시계열 CV (TimeSeriesSplit 등) 또는 일반 CV 수행
            logger.info(
                f"[{self.model_name}] Starting Hyperparameter Tuning with CV..."
            )
            grid_search = GridSearchCV(
                self.model, param_grid, cv=cv, scoring=scoring, n_jobs=-1
            )
            grid_search.fit(X_train, y_train, **fit_params)
            self.model = grid_search.best_estimator_
            self.best_params = grid_search.best_params_
            self.is_tuned = True
        else:
            # Hold-out 방식 (baseline_script 방식)
            logger.info(f"[{self.model_name}] Training with Hold-out validation...")
            if X_valid is not None and y_valid is not None:
                # LGBM의 early stopping 등을 활용하기 위해 eval_set 전달
                self.model.fit(
                    X_train, y_train, eval_set=[(X_valid, y_valid)], **fit_params
                )
            else:
                self.model.fit(X_train, y_train, **fit_params)

        logger.info(f"[{self.model_name}] Training completed.")

    def get_model_info(self) -> dict[str, Any]:
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
