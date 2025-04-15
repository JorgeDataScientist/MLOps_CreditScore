"""Evalúa el modelo entrenado con métricas detalladas y visualizaciones.

Carga datos de prueba y el modelo, calcula métricas, genera matriz de confusión,
guarda resultados en metrics/ y gráficos en graphics/, y registra en MLflow/DAGsHub.

Dependencias:
    - pandas: Para cargar datos.
    - sklearn: Para métricas y reportes.
    - joblib: Para cargar modelos.
    - hydra: Para configuraciones.
    - mlflow: Para rastreo.
    - matplotlib, seaborn: Para visualizaciones.
    - utils: Para logging y métricas.
    - pathlib: Para rutas.
    - logging: Para registro de eventos.
"""

import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, classification_report
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import os
import mlflow
from utils import BaseLogger, compute_metrics

# Configura el logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(config: DictConfig) -> tuple:
    """Carga datos de prueba desde archivos especificados.

    Args:
        config: Configuración con rutas a datos de prueba.

    Returns:
        Tupla con X_test y y_test.
    """
    base_path = Path(get_original_cwd())
    X_test = pd.read_csv(base_path / config.processed.X_test.path)
    y_test = pd.read_csv(base_path / config.processed.y_test.path)
    logger.info("Datos de prueba cargados")
    return X_test, y_test.values.ravel()


def load_model(config: DictConfig):
    """Carga el modelo entrenado desde un archivo.

    Args:
        config: Configuración con ruta al modelo.

    Returns:
        Modelo cargado.
    """
    model_name = config.model_config._name
    model_path = Path(get_original_cwd()) / config.model.dir / model_name / config.model.name
    logger.info(f"Cargando modelo desde {model_path}")
    return joblib.load(model_path)


def save_metrics(metrics: dict, config: DictConfig):
    """Guarda métricas en un archivo CSV.

    Args:
        metrics: Diccionario con métricas calculadas.
        config: Configuración con ruta para métricas.
    """
    metrics_dir = Path(get_original_cwd()) / config.metrics.dir
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / config.metrics.name
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    logger.info(f"Métricas guardadas en {metrics_path}")


def save_confusion_matrix(y_test, y_pred, config: DictConfig):
    """Genera y guarda la matriz de confusión como imagen.

    Args:
        y_test: Etiquetas reales.
        y_pred: Predicciones del modelo.
        config: Configuración con clases y ruta para gráficos.
    """
    graphics_dir = Path(get_original_cwd()) / config.graphics.dir
    graphics_dir.mkdir(parents=True, exist_ok=True)
    cm_path = graphics_dir / config.graphics.confusion_matrix.name
    
    cm = confusion_matrix(y_test, y_pred, labels=config.process.target_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=config.process.target_classes, yticklabels=config.process.target_classes)
    plt.xlabel("Predicho")
    plt.ylabel("Real")
    plt.title(f"Matriz de Confusión - {config.model_config._name}")
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Matriz de confusión guardada en {cm_path}")


def evaluate(config: DictConfig):
    """Evalúa el modelo con métricas detalladas y registra resultados.

    Args:
        config: Configuración con rutas y parámetros de evaluación.

    Raises:
        FileNotFoundError: Si los archivos de datos o modelo no existen.
    """
    # Configurar credenciales de DAGsHub/MLflow
    try:
        versioning_config = OmegaConf.load(Path(get_original_cwd()) / "config/versioning_dagshub.yaml")
        os.environ["MLFLOW_TRACKING_USERNAME"] = versioning_config.dagshub.username
        os.environ["MLFLOW_TRACKING_PASSWORD"] = versioning_config.dagshub.token
        logger.info("Credenciales de MLflow configuradas desde versioning_dagshub.yaml")
    except Exception as e:
        logger.warning(f"No se pudieron cargar las credenciales: {e}. Continuando sin DAGsHub.")

    logger.info("Iniciando evaluación")
    logger_instance = BaseLogger(
        config.mlflow.tracking_uri, use_dagshub=config.mlflow.use_dagshub
    )
    X_test, y_test = load_data(config)
    model = load_model(config)
    
    y_pred = model.predict(X_test)
    metrics = compute_metrics(y_test, y_pred, config)
    for metric_name, score in metrics.items():
        logger.info(f"{metric_name}: {score:.3f}")
    
    save_metrics(metrics, config)
    save_confusion_matrix(y_test, y_pred, config)
    
    class_report = classification_report(
        y_test, y_pred, target_names=config.process.target_classes, labels=config.process.target_classes
    )
    logger.info("\nReporte de clasificación:\n" + class_report)
    
    try:
        logger_instance.log_metrics(metrics)
        with open("class_report.txt", "w") as f:
            f.write(class_report)
        mlflow.log_artifact("class_report.txt")
        logger.info("Reporte de clasificación registrado en MLflow")
    except Exception as e:
        logger.warning(f"No se pudo registrar en MLflow: {e}")


if __name__ == "__main__":
    """Punto de entrada para ejecutar la evaluación directamente."""
    import hydra

    @hydra.main(version_base=None, config_path="../config", config_name="main")
    def main(config: DictConfig):
        evaluate(config)

    main()