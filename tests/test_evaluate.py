"""Pruebas unitarias para evaluate.py.

Verifica funciones de carga de datos y modelo, evaluación, guardado de métricas,
reporte de clasificación y generación de informe EDA.

Dependencias:
    - pytest: Para ejecutar pruebas.
    - pandas: Para manipulación de datos.
    - sklearn: Para métricas y modelos.
    - unittest.mock: Para simular configuraciones.
    - pathlib: Para manejar rutas.
"""

import sys
from pathlib import Path

# Añadir el directorio raíz al sys.path
ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))

import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from unittest.mock import Mock, patch
from src.evaluate import load_data_and_model, evaluate_model, save_metrics, save_class_report, generate_eda_report

# Ruta base para datos simulados
BASE_PATH = ROOT_DIR / "tests" / "data"

@pytest.fixture
def config():
    """Crea una configuración simulada para pruebas."""
    return Mock(
        model_config=Mock(_name="model_1"),
        process=Mock(target_classes=["Poor", "Standard", "Good"]),
        model=Mock(name="rf_model.pkl"),
        processed=Mock(
            X_test=Mock(path=str(BASE_PATH / "processed" / "X_test.csv")),
            y_test=Mock(path=str(BASE_PATH / "processed" / "y_test.csv"))
        ),
        metrics=Mock(dir="metrics"),
        informe=Mock(dir="informe")
    )

@pytest.fixture
def processed_data():
    """Carga datos procesados simulados."""
    X_test = pd.read_csv(BASE_PATH / "processed" / "X_test.csv")
    y_test = pd.read_csv(BASE_PATH / "processed" / "y_test.csv")
    return X_test, y_test["Puntaje_Credito"].values

@pytest.fixture
def trained_model(processed_data):
    """Crea un modelo entrenado simulado."""
    X_train = pd.read_csv(BASE_PATH / "processed" / "X_train.csv")
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, pd.read_csv(BASE_PATH / "processed" / "y_train.csv")["Puntaje_Credito"].values)
    return model

def test_load_data_and_model(config, tmp_path):
    """Verifica que load_data_and_model carga datos y modelo correctamente.

    Args:
        config: Configuración simulada.
        tmp_path: Directorio temporal proporcionado por pytest.

    Returns:
        None

    Raises:
        AssertionError: Si los datos o modelo no se cargan.
    """
    # Crear un modelo simulado
    model = RandomForestClassifier()
    model_path = tmp_path / "models" / "model_1" / "rf_model.pkl"
    model_path.parent.mkdir(parents=True)
    import joblib
    joblib.dump(model, model_path)
    
    with patch("src.evaluate.Path", return_value=tmp_path):
        X_test, y_test, loaded_model = load_data_and_model(config)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_test, np.ndarray)
    assert isinstance(loaded_model, RandomForestClassifier)
    assert all(col in X_test.columns for col in ["Salario_Mensual", "Deuda_Pendiente", "Edad"])

def test_evaluate_model(processed_data, trained_model, config):
    """Verifica que evaluate_model calcula métricas correctamente.

    Args:
        processed_data: Datos simulados procesados.
        trained_model: Modelo entrenado simulado.
        config: Configuración simulada.

    Returns:
        None

    Raises:
        AssertionError: Si las métricas no se calculan.
    """
    X_test, y_test = processed_data
    metrics, y_pred = evaluate_model(X_test, y_test, trained_model, config)
    assert isinstance(metrics, dict)
    assert "accuracy" in metrics
    assert "f1_macro" in metrics
    assert "roc_auc" in metrics
    assert all(0 <= value <= 1 for value in metrics.values())
    assert len(y_pred) == len(y_test)

def test_save_metrics(tmp_path, processed_data, trained_model, config):
    """Verifica que save_metrics guarda métricas correctamente.

    Args:
        tmp_path: Directorio temporal proporcionado por pytest.
        processed_data: Datos simulados procesados.
        trained_model: Modelo entrenado simulado.
        config: Configuración simulada.

    Returns:
        None

    Raises:
        AssertionError: Si el archivo no se guarda.
    """
    X_test, y_test = processed_data
    metrics, _ = evaluate_model(X_test, y_test, trained_model, config)
    with patch("src.evaluate.Path", return_value=tmp_path):
        save_metrics(metrics, config)
    assert (tmp_path / "metrics" / "model_1" / "metrics.csv").exists()

def test_save_class_report(tmp_path, processed_data, trained_model, config):
    """Verifica que save_class_report guarda el reporte correctamente.

    Args:
        tmp_path: Directorio temporal proporcionado por pytest.
        processed_data: Datos simulados procesados.
        trained_model: Modelo entrenado simulado.
        config: Configuración simulada.

    Returns:
        None

    Raises:
        AssertionError: Si el archivo no se guarda.
    """
    X_test, y_test = processed_data
    _, y_pred = evaluate_model(X_test, y_test, trained_model, config)
    with patch("src.evaluate.Path", return_value=tmp_path):
        save_class_report(y_test, y_pred, config)
    assert (tmp_path / "metrics" / "model_1" / "class_report_model_1.txt").exists()

@patch("src.evaluate.ProfileReport")
def test_generate_eda_report(mock_profile, tmp_path, processed_data, config):
    """Verifica que generate_eda_report genera el informe EDA.

    Args:
        mock_profile: Mock de ProfileReport.
        tmp_path: Directorio temporal proporcionado por pytest.
        processed_data: Datos simulados procesados.
        config: Configuración simulada.

    Returns:
        None

    Raises:
        AssertionError: Si el informe no se guarda.
    """
    X_train = pd.read_csv(BASE_PATH / "processed" / "X_train.csv")
    y_train = pd.read_csv(BASE_PATH / "processed" / "y_train.csv")["Puntaje_Credito"].values
    with patch("src.evaluate.Path", return_value=tmp_path):
        generate_eda_report(X_train, y_train, config)
    assert (tmp_path / "informe" / "model_1" / "informe.html").exists()
    mock_profile.assert_called()