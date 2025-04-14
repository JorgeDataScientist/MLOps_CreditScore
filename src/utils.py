import logging
import mlflow
from pathlib import Path
from omegaconf import DictConfig
import pandas as pd
from typing import Optional

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class MLflowLogger:
    """Clase para manejar logging en MLflow y DAGsHub."""
    def __init__(self, config: DictConfig):
        self.tracking_uri = config.mlflow.tracking_uri
        self.experiment_name = config.mlflow.experiment_name
        self.use_dagshub = config.mlflow.use_dagshub
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)

    def start_run(self, run_name: Optional[str] = None):
        return mlflow.start_run(run_name=run_name)

    def log_params(self, params: dict):
        mlflow.log_params(params)

    def log_metrics(self, metrics: dict):
        mlflow.log_metrics(metrics)

    def log_artifact(self, file_path: str):
        mlflow.log_artifact(file_path)

    def log_model(self, model, artifact_path: str):
        mlflow.sklearn.log_model(model, artifact_path)

def ensure_dir(file_path: str):
    """Crea directorios si no existen."""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)

def validate_data(df: pd.DataFrame, expected_columns: list) -> bool:
    """Valida que el DataFrame tenga las columnas esperadas."""
    missing = [col for col in expected_columns if col not in df.columns]
    if missing:
        logger.error(f"Faltan columnas: {missing}")
        return False
    return True
