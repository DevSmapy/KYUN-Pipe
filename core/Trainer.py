import inspect
import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
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
            logger.info(f"[{self.model_name}] Training with Hold-out validation...")

            fit_sig = inspect.signature(self.model.fit)
            accepts_kwargs = any(
                p.kind == inspect.Parameter.VAR_KEYWORD
                for p in fit_sig.parameters.values()
            )
            allowed_keys = set(fit_sig.parameters.keys())

            safe_fit_params: dict[str, Any] = {}
            for k, v in fit_params.items():
                if accepts_kwargs or k in allowed_keys:
                    safe_fit_params[k] = v

            # eval_set은:
            # - 사용자가 fit_params로 넣었으면 그대로 사용
            # - 아니면 X_valid/y_valid가 있을 때, 모델이 받는 경우에만 자동 주입
            if (
                X_valid is not None
                and y_valid is not None
                and "eval_set" not in safe_fit_params
                and (accepts_kwargs or "eval_set" in allowed_keys)
            ):
                safe_fit_params["eval_set"] = [(X_valid, y_valid)]

            self.model.fit(X_train, y_train, **safe_fit_params)

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


class TimeSeriesTrainer(ModelTrainer):
    """
    Specialized trainer for Time Series models.
    """

    def __init__(
        self, model: Any, model_name: str = "TSModel", target_col: str = "sales"
    ):
        super().__init__(model, model_name)
        self.target_col = target_col
        self.features = None

    def train_with_log(self, train_df, valid_df, features, **fit_params):
        """
        Automatically handles log1p transformation of the target.
        """
        self.features = features
        X_train = train_df[features]
        y_train = np.log1p(train_df[self.target_col])
        X_valid = valid_df[features]
        y_valid = np.log1p(valid_df[self.target_col])

        super().train(X_train, y_train, X_valid=X_valid, y_valid=y_valid, **fit_params)

        preds_log = self.model.predict(X_valid)
        rmsle = np.sqrt(mean_squared_error(y_valid, preds_log))
        logger.info(f"[{self.model_name}] Validation RMSLE: {rmsle:.4f}")
        return rmsle

    def predict_original_scale(self, df: pd.DataFrame) -> np.ndarray:
        """
        Returns predictions reversed from log scale (expm1).
        """
        preds_log = self.model.predict(df[self.features])
        return np.expm1(preds_log)
