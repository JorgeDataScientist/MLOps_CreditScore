"""Entrena un modelo RandomForestClassifier con optimización de hiperparámetros.

Carga datos procesados, construye un pipeline con escalado y modelo, optimiza
hiperparámetros con RandomizedSearchCV, calcula métricas, genera gráficas
(matriz de confusión, curva ROC, barras de métricas) y registra resultados
en MLflow/DAGsHub. Guarda el modelo, parámetros y gráficas en models/model_X/
y graphics/model_X/.

Dependencias:
    - pandas: Para manipulación de datos.
    - sklearn: Para modelo, pipeline, métricas y gráficas.
    - hydra: Para configuraciones.
    - mlflow: Para rastreo de experimentos.
    - joblib: Para guardar modelos.
    - matplotlib, seaborn: Para visualizaciones.
    - utils: Para logging y métricas.
    - pathlib: Para manejo de rutas.
    - logging: Para registro de eventos.
    - os: Para configurar variables de entorno.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
import joblib
import json
from pathlib import Path
import logging
import os
import matplotlib.pyplot as plt
import seaborn as sns
from utils import BaseLogger, compute_metrics

# Configura el logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(config: DictConfig) -> tuple:
    """Carga datos procesados desde archivos especificados.

    Args:
        config: Configuración con rutas a datos procesados.

    Returns:
        Tupla con X_train, X_test, y_train, y_test.
    """
    base_path = Path(get_original_cwd())
    X_train = pd.read_csv(base_path / config.processed.X_train.path)
    X_test = pd.read_csv(base_path / config.processed.X_test.path)
    y_train = pd.read_csv(base_path / config.processed.y_train.path)
    y_test = pd.read_csv(base_path / config.processed.y_test.path)
    if isinstance(y_train, pd.DataFrame):
        y_train = y_train["Puntaje_Credito"].values.ravel() if "Puntaje_Credito" in y_train.columns else y_train.iloc[:, 0].values.ravel()
    if isinstance(y_test, pd.DataFrame):
        y_test = y_test["Puntaje_Credito"].values.ravel() if "Puntaje_Credito" in y_test.columns else y_test.iloc[:, 0].values.ravel()
    logger.info("Datos procesados cargados")
    return X_train, X_test, y_train, y_test

def build_pipeline(config: DictConfig) -> Pipeline:
    """Construye un pipeline con escalado y modelo RandomForest.

    Args:
        config: Configuración con parámetros del modelo.

    Returns:
        Pipeline con StandardScaler y RandomForestClassifier.
    """
    logger.info("Construyendo pipeline")
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                RandomForestClassifier(**config.model_config.params),
            ),
        ]
    )
    return pipeline

def save_confusion_matrix(y_test, y_pred, config: DictConfig):
    """Genera y guarda la matriz de confusión.

    Args:
        y_test: Etiquetas reales.
        y_pred: Predicciones.
        config: Configuración con clases y rutas.
    """
    model_name = config.model_config._name
    graphics_dir = Path(get_original_cwd()) / "graphics" / model_name
    graphics_dir.mkdir(parents=True, exist_ok=True)
    cm_path = graphics_dir / "confusion_matrix.png"
    
    cm = confusion_matrix(y_test, y_pred, labels=config.process.target_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=config.process.target_classes, yticklabels=config.process.target_classes)
    plt.xlabel("Predicho")
    plt.ylabel("Real")
    plt.title(f"Matriz de Confusión - {model_name}")
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Matriz de confusión guardada en {cm_path}")

def save_metrics_bar(metrics, config: DictConfig):
    """Genera y guarda un gráfico de barras de métricas.

    Args:
        metrics: Diccionario con métricas.
        config: Configuración con rutas.
    """
    model_name = config.model_config._name
    graphics_dir = Path(get_original_cwd()) / "graphics" / model_name
    graphics_dir.mkdir(parents=True, exist_ok=True)
    bar_path = graphics_dir / "metrics_bar.png"
    
    plt.figure(figsize=(10, 6))
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())
    sns.barplot(x=metric_values, y=metric_names, palette="Blues")
    plt.xlabel("Valor")
    plt.ylabel("Métrica")
    plt.title(f"Métricas de Evaluación - {model_name}")
    plt.savefig(bar_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Gráfico de barras de métricas guardado en {bar_path}")

def save_roc_curve(y_test, y_pred_proba, config: DictConfig):
    """Genera y guarda la curva ROC para cada clase.

    Args:
        y_test: Etiquetas reales.
        y_pred_proba: Probabilidades predichas.
        config: Configuración con clases y rutas.
    """
    model_name = config.model_config._name
    graphics_dir = Path(get_original_cwd()) / "graphics" / model_name
    graphics_dir.mkdir(parents=True, exist_ok=True)
    roc_path = graphics_dir / "roc_curve.png"
    
    y_bin = label_binarize(y_test, classes=config.process.target_classes)
    n_classes = y_bin.shape[1]
    
    plt.figure(figsize=(8, 6))
    for i, class_name in enumerate(config.process.target_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_pred_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title(f'Curva ROC - {model_name}')
    plt.legend(loc="lower right")
    plt.savefig(roc_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Curva ROC guardada en {roc_path}")

def train(config: DictConfig) -> RandomForestClassifier:
    """Entrena un modelo RandomForest con optimización de hiperparámetros.

    Args:
        config: Configuración con rutas, parámetros y espacio de búsqueda.

    Returns:
        Modelo entrenado y optimizado.

    Raises:
        FileNotFoundError: Si los archivos de datos no existen.
    """
    # Configurar credenciales de DAGsHub/MLflow
    try:
        versioning_config = OmegaConf.load(Path(get_original_cwd()) / "config/versioning_dagshub.yaml")
        os.environ["MLFLOW_TRACKING_USERNAME"] = versioning_config.dagshub.username
        os.environ["MLFLOW_TRACKING_PASSWORD"] = versioning_config.dagshub.token
        logger.info("Credenciales de MLflow configuradas desde versioning_dagshub.yaml")
    except Exception as e:
        logger.warning(f"No se pudieron cargar las credenciales: {e}. Continuando sin DAGsHub.")

    logger.info("Iniciando entrenamiento")
    logger_instance = BaseLogger(
        config.mlflow.tracking_uri, use_dagshub=True
    )
    X_train, X_test, y_train, y_test = load_data(config)
    
    # Validar columnas
    expected_cols = config.process.features
    if not all(col in X_train.columns for col in expected_cols):
        logger.error("Faltan columnas esperadas en los datos")
        raise ValueError("Faltan columnas esperadas en los datos")
    
    pipeline = build_pipeline(config)
    search_space = OmegaConf.to_container(config.model_config.search_space, resolve=True)
    param_distributions = {
        f"model__{key}": value for key, value in search_space.items()
    }
    
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_distributions,
        n_iter=config.model_config.optimization.n_iter,
        cv=config.model_config.cv.folds,
        scoring=config.model_config.cv.scoring,
        n_jobs=-1,
        random_state=config.model_config.params.random_state,
    )
    
    logger.info("Optimizando hiperparámetros...")
    search.fit(X_train, y_train)
    best_params = search.best_params_
    logger.info(f"Mejores hiperparámetros: {best_params}")
    logger.info(f"Mejor CV {config.model_config.cv.scoring}: {search.best_score_:.3f}")
    
    y_pred = search.predict(X_test)
    y_pred_proba = search.predict_proba(X_test)
    metrics = compute_metrics(y_test, y_pred, config)
    # Calcular ROC AUC
    metrics["test_roc_auc"] = roc_auc_score(y_test, y_pred_proba, multi_class="ovr")
    for metric_name, score in metrics.items():
        logger.info(f"{metric_name}: {score:.3f}")
    
    # Generar gráficas
    save_confusion_matrix(y_test, y_pred, config)
    save_metrics_bar(metrics, config)
    save_roc_curve(y_test, y_pred_proba, config)
    
    # Guardar modelo y parámetros
    model_name = config.model_config._name
    model_dir = Path(get_original_cwd()) / config.model.dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / config.model.name
    params_path = model_dir / config.model.params_name
    
    joblib.dump(search.best_estimator_, model_path)
    logger.info(f"Modelo guardado en {model_path}")
    
    with open(params_path, "w") as f:
        json.dump(best_params, f, indent=4)
    logger.info(f"Parámetros guardados en {params_path}")
    
    # Registrar en MLflow
    try:
        logger_instance.log_params({k.replace("model__", ""): v for k, v in best_params.items()})
        logger_instance.log_metrics({f"cv_{config.model_config.cv.scoring}": search.best_score_, **metrics})
        logger_instance.log_model(search.best_estimator_, f"model_{model_name}")
    except Exception as e:
        logger.warning(f"No se pudo registrar en MLflow: {e}")
    
    return search.best_estimator_

if __name__ == "__main__":
    """Punto de entrada para ejecutar el entrenamiento directamente."""
    import hydra

    @hydra.main(version_base=None, config_path="../config", config_name="main")
    def main(config: DictConfig):
        try:
            train(config)
        except Exception as e:
            logger.error(f"Error en la ejecución: {e}")
            raise

    main()