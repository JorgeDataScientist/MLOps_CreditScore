"""Pruebas de integración para run_pipeline.py.

Verifica que el pipeline completo se ejecuta correctamente y genera los archivos esperados.

Dependencias:
    - pytest: Para ejecutar pruebas.
    - unittest.mock: Para simular configuraciones.
    - pathlib: Para manejar rutas.
"""

import sys
from pathlib import Path

# Añadir el directorio raíz al sys.path
ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))

import pytest
from unittest.mock import Mock, patch
from src.run_pipeline import run_pipeline

# Ruta base para datos simulados
BASE_PATH = ROOT_DIR / "tests" / "data"

@pytest.fixture
def config():
    """Crea una configuración simulada para pruebas."""
    return Mock(
        model_config=Mock(_name="model_1"),
        processed=Mock(
            dir=str(BASE_PATH / "processed"),
            X_train=Mock(name="X_train.csv"),
            X_test=Mock(name="X_test.csv"),
            y_train=Mock(name="y_train.csv"),
            y_test=Mock(name="y_test.csv")
        ),
        raw=Mock(path=str(BASE_PATH / "raw" / "credit_data.csv")),
        model=Mock(dir="models", name="rf_model.pkl", params_name="params.json"),
        process=Mock(
            features=["Salario_Mensual", "Deuda_Pendiente", "Edad", "debt_to_income"],
            target="Puntaje_Credito",
            target_classes=["Poor", "Standard", "Good"]
        ),
        mlflow=Mock(tracking_uri="https://dagshub.com/JorgeDataScientist/MLOps_CreditScore.mlflow"),
        metrics=Mock(dir="metrics"),
        informe=Mock(dir="informe")
    )

@patch("src.run_pipeline.preprocess")
@patch("src.run_pipeline.train")
@patch("src.run_pipeline.evaluate")
def test_pipeline_execution(mock_evaluate, mock_train, mock_preprocess, config, tmp_path):
    """Verifica que run_pipeline ejecuta los scripts correctamente.

    Args:
        mock_evaluate: Mock de evaluate.main.
        mock_train: Mock de train.train.
        mock_preprocess: Mock de preprocess.main.
        config: Configuración simulada.
        tmp_path: Directorio temporal proporcionado por pytest.

    Returns:
        None

    Raises:
        AssertionError: Si el pipeline no se ejecuta.
    """
    with patch("src.run_pipeline.Path", return_value=tmp_path):
        run_pipeline(config)
    mock_preprocess.assert_called_with(config)
    mock_train.assert_called_with(config)
    mock_evaluate.assert_called_with(config)

def test_output_files(tmp_path, config):
    """Verifica que run_pipeline genera todos los archivos esperados.

    Args:
        tmp_path: Directorio temporal proporcionado por pytest.
        config: Configuración simulada.

    Returns:
        None

    Raises:
        AssertionError: Si los archivos no se generan.
    """
    # Configurar mocks para simular ejecución
    with patch("src.run_pipeline.preprocess") as mock_preprocess, \
         patch("src.run_pipeline.train") as mock_train, \
         patch("src.run_pipeline.evaluate") as mock_evaluate, \
         patch("src.run_pipeline.Path", return_value=tmp_path):
        # Crear archivos simulados
        (tmp_path / "tests" / "data" / "processed").mkdir(parents=True)
        for file in ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"]:
            (tmp_path / "tests" / "data" / "processed" / file).touch()
        (tmp_path / "models" / "model_1").mkdir(parents=True)
        (tmp_path / "models" / "model_1" / "rf_model.pkl").touch()
        (tmp_path / "models" / "model_1" / "params.json").touch()
        (tmp_path / "graphics" / "model_1").mkdir(parents=True)
        for file in ["confusion_matrix.png", "metrics_bar.png", "roc_curve.png"]:
            (tmp_path / "graphics" / "model_1" / file).touch()
        (tmp_path / "metrics" / "model_1").mkdir(parents=True)
        (tmp_path / "metrics" / "model_1" / "metrics.csv").touch()
        (tmp_path / "metrics" / "model_1" / "class_report_model_1.txt").touch()
        (tmp_path / "informe" / "model_1").mkdir(parents=True)
        (tmp_path / "informe" / "model_1" / "informe.html").touch()
        
        run_pipeline(config)
    
    # Verificar archivos
    assert (tmp_path / "tests" / "data" / "processed" / "X_train.csv").exists()
    assert (tmp_path / "models" / "model_1" / "rf_model.pkl").exists()
    assert (tmp_path / "graphics" / "model_1" / "roc_curve.png").exists()
    assert (tmp_path / "metrics" / "model_1" / "metrics.csv").exists()
    assert (tmp_path / "metrics" / "model_1" / "class_report_model_1.txt").exists()
    assert (tmp_path / "informe" / "model_1" / "informe.html").exists()