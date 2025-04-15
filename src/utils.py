"""Módulo de utilidades para el pipeline de Credit Scoring.

Contiene clases y funciones para logging de experimentos y cálculo de métricas,
usadas en entrenamiento, evaluación y predicciones.

Dependencias:
    - mlflow: Para rastrear experimentos.
    - dagshub: Para integración con DAGsHub.
    - sklearn: Para cálculo de métricas.
"""

import mlflow
from dagshub import DAGsHubLogger
from sklearn.metrics import accuracy_score, f1_score
import logging

# Configura el logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseLogger:
    """Clase para manejar el logging de métricas, parámetros y modelos.

    Registra datos en MLflow y opcionalmente en DAGsHub según configuración.

    Args:
        tracking_uri: URI para el servidor de MLflow.
        use_dagshub: Indica si usar DAGsHub para logging.
    """

    def __init__(self, tracking_uri: str = None, use_dagshub: bool = False):
        """Inicializa el logger con configuración de MLflow y DAGsHub."""
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
            logger.info(f"MLflow configurado con tracking_uri: {tracking_uri}")
        self.use_dagshub = use_dagshub
        if use_dagshub:
            try:
                self.dagshub_logger = DAGsHubLogger()
                logger.info("DAGsHub logger inicializado")
            except Exception as e:
                logger.error(f"Error inicializando DAGsHub logger: {e}")
                self.dagshub_logger = None
        else:
            self.dagshub_logger = None

    def log_metrics(self, metrics: dict):
        """Registra métricas en MLflow y, si está habilitado, en DAGsHub.

        Args:
            metrics: Diccionario con nombres de métricas y sus valores.
        """
        try:
            mlflow.log_metrics(metrics)
            logger.info(f"Métricas registradas en MLflow: {metrics}")
            if self.use_dagshub and self.dagshub_logger:
                self.dagshub_logger.log_metrics(metrics)
                logger.info(f"Métricas registradas en DAGsHub: {metrics}")
        except Exception as e:
            logger.error(f"Error al registrar métricas: {e}")

    def log_params(self, params: dict):
        """Registra parámetros en MLflow y, si está habilitado, en DAGsHub.

        Args:
            params: Diccionario con nombres de parámetros y sus valores.
        """
        try:
            mlflow.log_params(params)
            logger.info(f"Parámetros registrados en MLflow: {params}")
            if self.use_dagshub and self.dagshub_logger:
                self.dagshub_logger.log_hyperparams(params)
                logger.info(f"Parámetros registrados en DAGsHub: {params}")
        except Exception as e:
            logger.error(f"Error al registrar parámetros: {e}")

    def log_model(self, model, artifact_path: str):
        """Registra un modelo en MLflow.

        Args:
            model: Modelo entrenado a registrar.
            artifact_path: Ruta dentro de MLflow para guardar el modelo.
        """
        try:
            mlflow.sklearn.log_model(model, artifact_path)
            logger.info(f"Modelo registrado en MLflow en: {artifact_path}")
        except Exception as e:
            logger.error(f"Error al registrar modelo: {e}")


def compute_metrics(y_true, y_pred, config, prefix: str = "test") -> dict:
    """Calcula métricas de evaluación según la configuración.

    Args:
        y_true: Etiquetas reales.
        y_pred: Predicciones del modelo.
        config: Configuración con lista de métricas a calcular.
        prefix: Prefijo para nombres de métricas (ej. 'test', 'train').

    Returns:
        Diccionario con métricas calculadas (ej. test_accuracy, test_f1_macro).
    """
    logger.info(f"Calculando métricas con prefijo: {prefix}")
    metrics = {}
    for metric in config.model_config.metrics:
        try:
            if metric == "accuracy":
                score = accuracy_score(y_true, y_pred)
                metrics[f"{prefix}_accuracy"] = score
            elif metric == "f1_macro":
                score = f1_score(y_true, y_pred, average="macro")
                metrics[f"{prefix}_f1_macro"] = score
            elif metric == "f1_per_class":
                f1_scores = f1_score(y_true, y_pred, average=None)
                for i, class_name in enumerate(config.process.target_classes):
                    metrics[f"{prefix}_f1_{class_name.lower()}"] = f1_scores[i]
            else:
                logger.warning(f"Métrica {metric} no soportada, ignorada")
        except Exception as e:
            logger.error(f"Error calculando métrica {metric}: {e}")
    return metrics