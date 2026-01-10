import json
import logging
import os
from datetime import datetime
from typing import Any

import joblib
import pandas as pd

logger = logging.getLogger(__name__)


class ResultReporter:
    """
    Handles saving experiment results, metadata, and model artifacts.
    """

    def __init__(self, base_path: str):
        self.base_path = base_path
        self.run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.output_dir = os.path.join(self.base_path, "results", self.run_id)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            logger.info(f"Created output directory: {self.output_dir}")

    def _save_json(self, data: dict[str, Any], filename: str):
        """Internal helper to save dictionary as JSON."""
        path = os.path.join(self.output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        logger.info(f"Saved: {path}")

    def save_metadata(self, metadata: dict[str, Any]):
        """Saves experiment metadata."""
        self._save_json(metadata, "metadata.json")

    def save_metrics(self, metrics: dict[str, Any]):
        """Saves evaluation metrics."""
        self._save_json(metrics, "metrics.json")

    def save_model(self, model: Any, name: str):
        """Saves trained model using joblib."""
        path = os.path.join(self.output_dir, f"{name}.pkl")
        joblib.dump(model, path)
        logger.info(f"Model saved: {path}")

    def save_predictions(self, df: pd.DataFrame, name: str = "submission"):
        """Saves model predictions as CSV."""
        path = os.path.join(self.output_dir, f"{name}.csv")
        df.to_csv(path, index=False)
        logger.info(f"Predictions saved: {path}")

    def save_plot(self, fig, name: str):
        """Saves matplotlib figure."""
        path = os.path.join(self.output_dir, f"{name}.png")
        fig.savefig(path)
        logger.info(f"Figure saved: {path}")
