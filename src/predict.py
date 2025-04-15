"""Realiza predicciones con el modelo entrenado en nuevos datos.

Carga datos nuevos, aplica preprocesamiento, carga el modelo, realiza predicciones,
imprime resultados en consola y guarda resultados en un archivo CSV.

Dependencias:
    - pandas: Para manipulación de datos.
    - sklearn: Para preprocesamiento.
    - joblib: Para cargar modelos.
    - hydra: Para configuraciones.
    - pathlib: Para rutas.
    - logging: Para registro de eventos.
"""

import pandas as pd
import joblib
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from pathlib import Path
import logging

# Configura el logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_new_data(config: DictConfig) -> tuple:
    """Carga datos nuevos desde un archivo especificado.

    Args:
        config: Configuración con ruta a datos nuevos.

    Returns:
        Tupla con datos (DataFrame) y etiquetas (None si no están disponibles).
    """
    data_path = Path(get_original_cwd()) / config.predict.input_path
    logger.info(f"Cargando datos nuevos desde {data_path}")
    try:
        data = pd.read_csv(data_path)
        labels = None  # Nuevos datos no tienen etiquetas
        return data, labels
    except FileNotFoundError as e:
        logger.error(f"No se encontró el archivo de datos: {e}")
        raise

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

def save_predictions(predictions: pd.DataFrame, config: DictConfig):
    """Guarda las predicciones en un archivo CSV.

    Args:
        predictions: DataFrame con las predicciones.
        config: Configuración con ruta para guardar resultados.
    """
    output_dir = Path(get_original_cwd()) / config.predict.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / config.predict.output_name
    predictions.to_csv(output_path, index=False)
    logger.info(f"Predicciones guardadas en {output_path}")

def predict(config: DictConfig):
    """Realiza predicciones con el modelo entrenado en nuevos datos.

    Args:
        config: Configuración con rutas y parámetros.

    Raises:
        FileNotFoundError: Si los archivos de datos o modelo no existen.
    """
    logger.info("Iniciando predicciones")
    data, _ = load_new_data(config)
    model = load_model(config)
    
    # Validar columnas
    expected_cols = config.process.features
    if not all(col in data.columns for col in expected_cols):
        missing_cols = [col for col in expected_cols if col not in data.columns]
        logger.error(f"Faltan columnas en los datos nuevos: {missing_cols}")
        raise ValueError(f"Faltan columnas en los datos nuevos: {missing_cols}")
    
    predictions = model.predict(data)
    pred_df = pd.DataFrame(predictions, columns=["Puntaje_Credito"])
    
    # Imprimir predicciones en consola
    logger.info("Predicciones:")
    for idx, pred in enumerate(pred_df["Puntaje_Credito"]):
        logger.info(f"Instancia {idx + 1}: {pred}")
    
    save_predictions(pred_df, config)

if __name__ == "__main__":
    """Punto de entrada para ejecutar las predicciones directamente."""
    import hydra

    @hydra.main(version_base=None, config_path="../config", config_name="main")
    def main(config: DictConfig):
        predict(config)

    main()