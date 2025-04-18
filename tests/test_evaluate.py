"""Pruebas unitarias para evaluate.py.

Verifica funciones de evaluación de modelos, cálculo de métricas y guardado de resultados.

Dependencias:
    - pytest: Para ejecutar pruebas.
    - pandas: Para manipulación de datos.
    - sklearn: Para métricas y reportes.
    - joblib: Para cargar modelos simulados.
    - unittest.mock: Para simular configuraciones y rutas.
    - pathlib: Para manejo de rutas.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
from src.evaluate import (
    load_data_and_model,
    save_metrics,
    save_class_report
)
from omegaconf import DictConfig

@pytest.fixture
def config():
    """Crea una configuración simulada para pruebas."""
    return Mock(
        model_config=Mock(_name="rf_base"),
        process=Mock(
            target_classes=["Good", "Standard", "Poor"],
            target="Puntaje_Credito"
        )
    )

@pytest.fixture
def sample_data():
    """Crea datos de prueba simulados."""
    X_test = pd.DataFrame({
        "Edad": [25, 30],
        "Salario_Mensual": [5000.0, 6000.0]
    })
    y_test = np.array(["Good", "Standard"])
    X_train = pd.DataFrame({
        "Edad": [20, 28],
        "Salario_Mensual": [4500.0, 5500.0]
    })
    y_train = np.array(["Good", "Poor"])
    return X_test, y_test, X_train, y_train

def test_load_data_and_model(tmp_path, config):
    """Verifica que load_data_and_model carga datos y modelo correctamente."""
    base_path = tmp_path
    (base_path / "data/processed").mkdir(parents=True)
    (base_path / "models/rf_base").mkdir(parents=True)
    
    X_test = pd.DataFrame({"Edad": [25, 30]})
    y_test = pd.DataFrame({"Puntaje_Credito": ["Good", "Standard"]})
    X_test.to_csv(base_path / "data/processed/X_test.csv", index=False)
    y_test.to_csv(base_path / "data/processed/y_test.csv", index=False)
    
    mock_model = Mock()
    with patch("joblib.load", return_value=mock_model), \
         patch("hydra.utils.get_original_cwd", return_value=str(base_path)):
        X_test_loaded, y_test_loaded, model_loaded = load_data_and_model(config)
    
    assert isinstance(X_test_loaded, pd.DataFrame)
    assert isinstance(y_test_loaded, np.ndarray)
    assert X_test_loaded.shape == (2, 1)
    assert y_test_loaded.tolist() == ["Good", "Standard"]
    assert model_loaded == mock_model

def test_save_metrics(tmp_path, config):
    """Verifica que save_metrics guarda métricas en un archivo CSV."""
    metrics = {"accuracy": 0.85, "f1_macro": 0.80, "roc_auc": 0.90}
    base_path = tmp_path
    
    with patch("hydra.utils.get_original_cwd", return_value=str(base_path)):
        save_metrics(metrics, config)
    
    metrics_path = base_path / "metrics/rf_base/metrics.csv"
    assert metrics_path.exists()
    metrics_df = pd.read_csv(metrics_path)
    assert metrics_df.shape == (1, 3)
    assert metrics_df["accuracy"].iloc[0] == 0.85
    assert metrics_df["f1_macro"].iloc[0] == 0.80
    assert metrics_df["roc_auc"].iloc[0] == 0.90

def test_save_class_report(tmp_path, sample_data, config):
    """Verifica que save_class_report guarda el reporte de clasificación."""
    _, y_test, _, _ = sample_data
    y_pred = np.array(["Good", "Standard"])
    base_path = tmp_path
    
    with patch("hydra.utils.get_original_cwd", return_value=str(base_path)), \
         patch("builtins.open", mock_open()) as mocked_file:
        save_class_report(y_test, y_pred, config)
    
    report_path = base_path / "metrics/rf_base/class_report_rf_base.txt"
    assert report_path.parent.exists()
    mocked_file.assert_called_once_with(report_path, "w")
    assert config.process.target_classes == ["Good", "Standard", "Poor"]