"""Evalúa el modelo de Credit Scoring con métricas y un informe EDA.

Carga el modelo entrenado, calcula métricas de rendimiento, genera un reporte de
clasificación, guarda métricas y reporte en metrics/<model_name>/, y produce un
informe EDA en informe/<model_name>/.

Dependencias:
    - pandas: Para manipulación de datos.
    - sklearn: Para métricas y reportes.
    - joblib: Para cargar el modelo.
    - ydata-profiling: Para informe EDA.
    - hydra: Para configuraciones.
    - pathlib: Para rutas.
    - logging: Para registro de eventos.
"""

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
import joblib
from ydata_profiling import ProfileReport
from pathlib import Path
import logging
import numpy as np
from omegaconf import DictConfig
import hydra

# Configura el logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data_and_model(config: DictConfig):
    """Carga datos procesados y el modelo entrenado.

    Args:
        config: Configuración con rutas a datos y modelo.

    Returns:
        Tupla con X_test, y_test y el modelo.
    """
    base_path = Path(hydra.utils.get_original_cwd())
    X_test = pd.read_csv(base_path / "data/processed/X_test.csv")
    y_test = pd.read_csv(base_path / "data/processed/y_test.csv")
    if isinstance(y_test, pd.DataFrame):
        y_test = y_test["Puntaje_Credito"].values.ravel() if "Puntaje_Credito" in y_test.columns else y_test.iloc[:, 0].values.ravel()
    model_name = config.model_config._name
    model_path = base_path / "models" / model_name / "rf_model.pkl"
    model = joblib.load(model_path)
    logger.info(f"Datos y modelo cargados desde {model_path}")
    return X_test, y_test, model

def evaluate_model(X_test, y_test, model, config: DictConfig):
    """Evalúa el modelo y calcula métricas.

    Args:
        X_test: Características de prueba.
        y_test: Etiquetas de prueba.
        model: Modelo entrenado.
        config: Configuración con clases.

    Returns:
        Diccionario con métricas y predicciones.
    """
    logger.info("Evaluando modelo...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "roc_auc": roc_auc_score(y_test, y_pred_proba, multi_class="ovr")
    }
    logger.info(f"Métricas: {metrics}")
    return metrics, y_pred

def save_metrics(metrics, config: DictConfig):
    """Guarda métricas en un archivo CSV.

    Args:
        metrics: Diccionario con métricas.
        config: Configuración con ruta para métricas.
    """
    model_name = config.model_config._name
    metrics_dir = Path(hydra.utils.get_original_cwd()) / "metrics" / model_name
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / "metrics.csv"
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    logger.info(f"Métricas guardadas en {metrics_path}")

def save_class_report(y_test, y_pred, config: DictConfig):
    """Guarda el reporte de clasificación como archivo de texto.

    Args:
        y_test: Etiquetas reales.
        y_pred: Predicciones.
        config: Configuración con clases y rutas.
    """
    model_name = config.model_config._name
    metrics_dir = Path(hydra.utils.get_original_cwd()) / "metrics" / model_name
    metrics_dir.mkdir(parents=True, exist_ok=True)
    report_path = metrics_dir / f"class_report_{model_name}.txt"
    
    class_report = classification_report(
        y_test, y_pred, target_names=config.process.target_classes, labels=config.process.target_classes
    )
    with open(report_path, "w") as f:
        f.write(class_report)
    logger.info(f"Reporte de clasificación guardado en {report_path}")

def generate_eda_report(X_train, y_train, config: DictConfig):
    """Genera un informe EDA con los datos de entrenamiento.

    Args:
        X_train: Características de entrenamiento.
        y_train: Etiquetas de entrenamiento.
        config: Configuración con rutas.
    """
    logger.info("Generando informe EDA...")
    model_name = config.model_config._name
    report_dir = Path(hydra.utils.get_original_cwd()) / "informe" / model_name
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "informe.html"
    
    df = X_train.copy()
    df["Puntaje_Credito"] = y_train["Puntaje_Credito"] if isinstance(y_train, pd.DataFrame) else y_train
    numeric_columns = df.select_dtypes(include=["int64", "float64"]).columns
    df_numeric = df[numeric_columns]
    
    profile = ProfileReport(df_numeric, title=f"Informe EDA Credit Scoring - {model_name}", explorative=True)
    profile.to_file(report_path)
    logger.info(f"Informe EDA guardado en {report_path}")

def main(config: DictConfig):
    """Ejecuta la evaluación del modelo y genera el informe EDA.

    Args:
        config: Configuración con rutas y parámetros.
    """
    X_test, y_test, model = load_data_and_model(config)
    base_path = Path(hydra.utils.get_original_cwd())
    X_train = pd.read_csv(base_path / "data/processed/X_train.csv")
    y_train = pd.read_csv(base_path / "data/processed/y_train.csv")
    if isinstance(y_train, pd.DataFrame):
        y_train = y_train["Puntaje_Credito"].values.ravel() if "Puntaje_Credito" in y_train.columns else y_train.iloc[:, 0].values.ravel()
    
    metrics, y_pred = evaluate_model(X_test, y_test, model, config)
    save_metrics(metrics, config)
    save_class_report(y_test, y_pred, config)
    generate_eda_report(X_train, y_train, config)

if __name__ == "__main__":
    @hydra.main(version_base=None, config_path="../config", config_name="main")
    def run(config: DictConfig):
        main(config)
    run()