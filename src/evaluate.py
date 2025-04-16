"""Evalúa el modelo entrenado con métricas detalladas y visualizaciones.

Carga datos de prueba y el modelo, calcula métricas, genera múltiples gráficos
(matriz de confusión, histograma de clases, barras de métricas, curva ROC,
dispersión de características, importancia de características),
guarda resultados, gráficos y reporte en metrics/ y graphics/<model_name>/.

Dependencias:
    - pandas: Para cargar datos.
    - sklearn: Para métricas y reportes.
    - joblib: Para cargar modelos.
    - hydra: Para configuraciones.
    - matplotlib, seaborn: Para visualizaciones.
    - utils: Para métricas.
    - pathlib: Para rutas.
    - logging: Para registro de eventos.
"""

import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import numpy as np
from utils import compute_metrics

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
    model_name = config.model_config._name
    graphics_dir = Path(get_original_cwd()) / config.graphics.dir / model_name
    graphics_dir.mkdir(parents=True, exist_ok=True)
    cm_path = graphics_dir / config.graphics.confusion_matrix.name
    
    cm = confusion_matrix(y_test, y_pred, labels=config.process.target_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=config.process.target_classes, yticklabels=config.process.target_classes)
    plt.xlabel("Predicho")
    plt.ylabel("Real")
    plt.title(f"Matriz de Confusión - {model_name}")
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Matriz de confusión guardada en {cm_path}")

def save_class_distribution(y_test, config: DictConfig):
    """Genera y guarda un histograma de la distribución de clases.

    Args:
        y_test: Etiquetas reales.
        config: Configuración con ruta para gráficos.
    """
    model_name = config.model_config._name
    graphics_dir = Path(get_original_cwd()) / config.graphics.dir / model_name
    graphics_dir.mkdir(parents=True, exist_ok=True)
    hist_path = graphics_dir / f"class_distribution_{model_name}.png"
    
    plt.figure(figsize=(8, 6))
    sns.countplot(x=y_test, order=config.process.target_classes, palette="Blues")
    plt.xlabel("Puntaje Crediticio")
    plt.ylabel("Frecuencia")
    plt.title(f"Distribución de Clases - {model_name}")
    plt.savefig(hist_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Histograma de clases guardado en {hist_path}")

def save_metrics_bar(metrics: dict, config: DictConfig):
    """Genera y guarda un gráfico de barras de métricas.

    Args:
        metrics: Diccionario con métricas calculadas.
        config: Configuración con ruta para gráficos.
    """
    model_name = config.model_config._name
    graphics_dir = Path(get_original_cwd()) / config.graphics.dir / model_name
    graphics_dir.mkdir(parents=True, exist_ok=True)
    bar_path = graphics_dir / f"metrics_bar_{model_name}.png"
    
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
        config: Configuración con clases y ruta para gráficos.
    """
    model_name = config.model_config._name
    graphics_dir = Path(get_original_cwd()) / config.graphics.dir / model_name
    graphics_dir.mkdir(parents=True, exist_ok=True)
    roc_path = graphics_dir / f"roc_curve_{model_name}.png"
    
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

def save_feature_scatter(X_test, y_test, config: DictConfig):
    """Genera y guarda un gráfico de dispersión de dos características.

    Args:
        X_test: Datos de prueba.
        y_test: Etiquetas reales.
        config: Configuración con ruta para gráficos.
    """
    model_name = config.model_config._name
    graphics_dir = Path(get_original_cwd()) / config.graphics.dir / model_name
    graphics_dir.mkdir(parents=True, exist_ok=True)
    scatter_path = graphics_dir / f"feature_scatter_{model_name}.png"
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_test['Salario_Mensual'], y=X_test['Deuda_Pendiente'], hue=y_test, style=y_test, palette="deep")
    plt.xlabel("Salario Mensual")
    plt.ylabel("Deuda Pendiente")
    plt.title(f"Dispersión de Características - {model_name}")
    plt.legend(title="Puntaje Crediticio")
    plt.savefig(scatter_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Gráfico de dispersión guardado en {scatter_path}")

def save_feature_importance(model, config: DictConfig):
    """Genera y guarda un gráfico de importancia de características.

    Args:
        model: Modelo entrenado.
        config: Configuración con ruta para gráficos.
    """
    model_name = config.model_config._name
    graphics_dir = Path(get_original_cwd()) / config.graphics.dir / model_name
    graphics_dir.mkdir(parents=True, exist_ok=True)
    importance_path = graphics_dir / f"feature_importance_{model_name}.png"
    
    importances = model.named_steps['model'].feature_importances_
    feature_names = config.process.features
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances, y=feature_names, palette="Blues")
    plt.xlabel("Importancia")
    plt.ylabel("Característica")
    plt.title(f"Importancia de Características - {model_name}")
    plt.savefig(importance_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Gráfico de importancia de características guardado en {importance_path}")

def save_class_report(class_report: str, config: DictConfig):
    """Guarda el reporte de clasificación como archivo de texto.

    Args:
        class_report: Texto del reporte de clasificación.
        config: Configuración con ruta para el reporte.
    """
    metrics_dir = Path(get_original_cwd()) / config.metrics.dir
    metrics_dir.mkdir(parents=True, exist_ok=True)
    report_path = metrics_dir / f"class_report_{config.model_config._name}.txt"
    with open(report_path, "w") as f:
        f.write(class_report)
    logger.info(f"Reporte de clasificación guardado en {report_path}")

def evaluate(config: DictConfig):
    """Evalúa el modelo con métricas detalladas y registra resultados localmente.

    Args:
        config: Configuración con rutas y parámetros de evaluación.

    Raises:
        FileNotFoundError: Si los archivos de datos o modelo no existen.
    """
    logger.info("Iniciando evaluación")
    X_test, y_test = load_data(config)
    model = load_model(config)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    metrics = compute_metrics(y_test, y_pred, config)
    for metric_name, score in metrics.items():
        logger.info(f"{metric_name}: {score:.3f}")
    
    save_metrics(metrics, config)
    save_confusion_matrix(y_test, y_pred, config)
    save_class_distribution(y_test, config)
    save_metrics_bar(metrics, config)
    save_roc_curve(y_test, y_pred_proba, config)
    save_feature_scatter(X_test, y_test, config)
    save_feature_importance(model, config)
    
    class_report = classification_report(
        y_test, y_pred, target_names=config.process.target_classes, labels=config.process.target_classes
    )
    logger.info("\nReporte de clasificación:\n" + class_report)
    save_class_report(class_report, config)

if __name__ == "__main__":
    """Punto de entrada para ejecutar la evaluación directamente."""
    import hydra

    @hydra.main(version_base=None, config_path="../config", config_name="main")
    def main(config: DictConfig):
        evaluate(config)

    main()