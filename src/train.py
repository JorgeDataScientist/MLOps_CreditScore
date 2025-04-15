# """Entrena un modelo RandomForestClassifier con optimización de hiperparámetros.

# Carga datos procesados, construye un pipeline con escalado y modelo, optimiza
# hiperparámetros con RandomizedSearchCV, calcula métricas y registra resultados
# en MLflow/DAGsHub. Guarda el modelo y parámetros en models/model_X/.

# Dependencias:
#     - pandas: Para manipulación de datos.
#     - sklearn: Para modelo, pipeline y métricas.
#     - hydra: Para configuraciones.
#     - mlflow: Para rastreo de experimentos.
#     - joblib: Para guardar modelos.
#     - utils: Para logging y métricas.
#     - pathlib: Para manejo de rutas.
#     - logging: Para registro de eventos.
# """

# import pandas as pd
# import numpy as np
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import RandomizedSearchCV
# from hydra.utils import get_original_cwd
# from omegaconf import DictConfig, OmegaConf
# import joblib
# import json
# from pathlib import Path
# import logging
# from utils import BaseLogger, compute_metrics

# # Configura el logger
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)


# def load_data(config: DictConfig) -> tuple:
#     """Carga datos procesados desde archivos especificados.

#     Args:
#         config: Configuración con rutas a datos procesados.

#     Returns:
#         Tupla con X_train, X_test, y_train, y_test.
#     """
#     base_path = Path(get_original_cwd())
#     X_train = pd.read_csv(base_path / config.processed.X_train.path)
#     X_test = pd.read_csv(base_path / config.processed.X_test.path)
#     y_train = pd.read_csv(base_path / config.processed.y_train.path)
#     y_test = pd.read_csv(base_path / config.processed.y_test.path)
#     logger.info("Datos procesados cargados")
#     return X_train, X_test, y_train.values.ravel(), y_test.values.ravel()


# def build_pipeline(config: DictConfig) -> Pipeline:
#     """Construye un pipeline con escalado y modelo RandomForest.

#     Args:
#         config: Configuración con parámetros del modelo.

#     Returns:
#         Pipeline con StandardScaler y RandomForestClassifier.
#     """
#     logger.info("Construyendo pipeline")
#     pipeline = Pipeline(
#         [
#             ("scaler", StandardScaler()),
#             (
#                 "model",
#                 RandomForestClassifier(**config.model_config.params),
#             ),
#         ]
#     )
#     return pipeline


# def train(config: DictConfig) -> RandomForestClassifier:
#     """Entrena un modelo RandomForest con optimización de hiperparámetros.

#     Args:
#         config: Configuración con rutas, parámetros y espacio de búsqueda.

#     Returns:
#         Modelo entrenado y optimizado.

#     Raises:
#         FileNotFoundError: Si los archivos de datos no existen.
#     """
#     logger.info("Iniciando entrenamiento")
#     logger_instance = BaseLogger(
#         config.mlflow.tracking_uri, use_dagshub=config.mlflow.use_dagshub
#     )
#     X_train, X_test, y_train, y_test = load_data(config)
    
#     # Validar columnas
#     expected_cols = config.process.features
#     if not all(col in X_train.columns for col in expected_cols):
#         logger.error("Faltan columnas esperadas en los datos")
#         raise ValueError("Faltan columnas esperadas en los datos")
    
#     pipeline = build_pipeline(config)
#     search_space = OmegaConf.to_container(config.model_config.search_space, resolve=True)
#     param_distributions = {
#         f"model__{key}": value for key, value in search_space.items()
#     }
    
#     search = RandomizedSearchCV(
#         pipeline,
#         param_distributions=param_distributions,
#         n_iter=config.model_config.optimization.n_iter,
#         cv=config.model_config.cv.folds,
#         scoring=config.model_config.cv.scoring,
#         n_jobs=-1,
#         random_state=config.model_config.params.random_state,
#     )
    
#     logger.info("Optimizando hiperparámetros...")
#     search.fit(X_train, y_train)
#     best_params = search.best_params_
#     logger.info(f"Mejores hiperparámetros: {best_params}")
#     logger.info(f"Mejor CV {config.model_config.cv.scoring}: {search.best_score_:.3f}")
    
#     y_pred = search.predict(X_test)
#     metrics = compute_metrics(y_test, y_pred, config)
#     for metric_name, score in metrics.items():
#         logger.info(f"{metric_name}: {score:.3f}")
    
#     # Guardar modelo y parámetros
#     model_name = config.model_config._name  # Ej. model_1
#     model_dir = Path(get_original_cwd()) / config.model.dir / model_name
#     model_dir.mkdir(parents=True, exist_ok=True)
#     model_path = model_dir / config.model.name
#     params_path = model_dir / config.model.params_name
    
#     joblib.dump(search.best_estimator_, model_path)
#     logger.info(f"Modelo guardado en {model_path}")
    
#     with open(params_path, "w") as f:
#         json.dump(best_params, f, indent=4)
#     logger.info(f"Parámetros guardados en {params_path}")
    
#     # Registrar en MLflow/DAGsHub
#     logger_instance.log_params({k.replace("model__", ""): v for k, v in best_params.items()})
#     logger_instance.log_metrics({f"cv_{config.model_config.cv.scoring}": search.best_score_, **metrics})
#     logger_instance.log_model(search.best_estimator_, f"model_{model_name}")
    
#     return search.best_estimator_


# if __name__ == "__main__":
#     """Punto de entrada para ejecutar el entrenamiento directamente."""
#     import hydra

#     @hydra.main(version_base=None, config_path="../config", config_name="main")
#     def main(config: DictConfig):
#         train(config)

#     main()


"""Entrena un modelo RandomForestClassifier con optimización de hiperparámetros.

Carga datos procesados, construye un pipeline con escalado y modelo, optimiza
hiperparámetros con RandomizedSearchCV, calcula métricas y registra resultados
en MLflow/DAGsHub. Guarda el modelo y parámetros en models/model_X/.

Dependencias:
    - pandas: Para manipulación de datos.
    - sklearn: Para modelo, pipeline y métricas.
    - hydra: Para configuraciones.
    - mlflow: Para rastreo de experimentos.
    - joblib: Para guardar modelos.
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
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
import joblib
import json
from pathlib import Path
import logging
import os
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
    logger.info("Datos procesados cargados")
    return X_train, X_test, y_train.values.ravel(), y_test.values.ravel()


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
        config.mlflow.tracking_uri, use_dagshub=True  # Habilitar DAGsHub
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
    metrics = compute_metrics(y_test, y_pred, config)
    for metric_name, score in metrics.items():
        logger.info(f"{metric_name}: {score:.3f}")
    
    # Guardar modelo y parámetros
    model_name = config.model_config._name  # Ej. model_1
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