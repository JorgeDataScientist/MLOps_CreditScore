"""Ejecuta todas las etapas del pipeline de Credit Scoring en secuencia.

Corre preprocesamiento, entrenamiento, evaluación y predicción con un solo comando.

Dependencias:
    - subprocess: Para ejecutar scripts.
    - logging: Para registro de eventos.
    - pathlib: Para manejo de rutas.
    - os: Para manejo del entorno.
"""

import subprocess
import logging
from pathlib import Path
import os

# Configura el logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_stage(script: str):
    """Ejecuta un script de etapa específica.

    Args:
        script: Nombre del script a ejecutar (ej. 'preprocess.py').

    Raises:
        subprocess.CalledProcessError: Si el script falla.
    """
    script_path = Path.cwd() / "src" / script
    # Usar el python del entorno virtual
    python_path = os.path.join(os.environ.get("VIRTUAL_ENV", ""), "Scripts", "python.exe")
    logger.info(f"Ejecutando {script} con {python_path}...")
    try:
        result = subprocess.run([python_path, str(script_path)], check=True, capture_output=True, text=True)
        logger.info(f"{script} completado exitosamente")
        logger.debug(result.stdout)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error en {script}: {e.stderr}")
        raise

def run_pipeline():
    """Ejecuta todas las etapas del pipeline en orden."""
    stages = ["preprocess.py", "train.py", "evaluate.py", "predict.py"]
    for stage in stages:
        run_stage(stage)

if __name__ == "__main__":
    run_pipeline()