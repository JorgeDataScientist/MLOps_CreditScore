"""Script para ejecutar todos los tests unitarios en el proyecto.

Ejecuta pytest en el directorio tests/ con cobertura y reportes.

Dependencias:
    - pytest: Para ejecutar pruebas.
    - pytest-cov: Para reportes de cobertura.
    - subprocess: Para ejecutar comandos.
    - logging: Para registro de eventos.
"""

import subprocess
import logging
import os
from pathlib import Path
import glob

# Configura el logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_all_tests():
    """Ejecuta todos los tests en el directorio tests/ con cobertura."""
    # El script está en tests/, así que el directorio tests/ es el padre
    tests_dir = Path(__file__).parent
    
    if not tests_dir.is_dir():
        logger.error(f"Directorio {tests_dir} no encontrado")
        return
    
    # Verifica si hay archivos de prueba
    test_files = glob.glob(str(tests_dir / "test_*.py"))
    if not test_files:
        logger.warning(f"No se encontraron archivos de prueba en {tests_dir}")
        return
    
    # Usa el python del entorno virtual
    python_path = os.path.join(os.environ.get("VIRTUAL_ENV", ""), "Scripts", "python.exe")
    
    logger.info(f"Ejecutando todos los tests en {tests_dir}...")
    try:
        result = subprocess.run(
            [
                python_path, "-m", "pytest",
                str(tests_dir),
                "-v",
                "--cov=src",
                "--cov-report=html:coverage_report",
                "--cov-report=term",
                "--disable-warnings"
            ],
            check=True,
            text=True,
            encoding="utf-8"
        )
        logger.info("Tests completados exitosamente")
        logger.debug(result.stdout)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error al ejecutar tests: {e.stderr}")
        raise
    except UnicodeDecodeError as e:
        logger.error(f"Error de codificación: {e}")
        raise

if __name__ == "__main__":
    run_all_tests()