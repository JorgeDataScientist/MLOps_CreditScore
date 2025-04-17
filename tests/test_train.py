"""Pruebas unitarias para train.py.

Verifica funciones de carga de datos, construcción del pipeline, cálculo de métricas,
generación de gráficas y guardado del modelo.

Dependencias:
    - pytest: Para ejecutar pruebas.
    - pandas: Para manipulación de datos.
    - sklearn: Para métricas y modelos.
    - unittest.mock: Para simular MLflow y configuraciones.
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from unittest.mock import Mock, patch
from src.train import load_data, build_pipeline, save_confusion_matrix, save_metrics_bar, save_roc_curve, train
from src.utils import compute_metrics, BaseLogger

# Ruta base para datos simulados
BASE_PATH = ROOT_DIR / "tests" / "data"

@pytest.fixture
def config():
    """Crea una configuración simulada para pruebas."""
    return Mock(
        processed=Mock(
            X_train=Mock(path=str(BASE_PATH / "processed" / "X_train.csv")),
            X_test=Mock(path=str(BASE_PATH / "processed" / "X_test.csv")),
            y_train=Mock(path=str(BASE_PATH / "processed" / "y_train.csv")),
            y_test=Mock(path=str(BASE_PATH / "processed" / "y_test.csv"))
        ),
        model_config=Mock(
            _name="model_1",
            params=Mock(random_state=42),
            search_space={},
            cv=Mock(folds=5, scoring="f1_macro"),
            optimization=Mock(n_iter=10),
            metrics=["accuracy", "f1_macro", "f1_per_class"]
        ),
        model=Mock(dir="models", name="rf_model.pkl", params_name="params.json"),
        process=Mock(
            features=["Salario_Mensual", "Deuda_Pendiente", "Edad", "debt_to_income"],
            target="Puntaje_Credito",
            target_classes=["Poor", "Standard", "Good"]
        ),
        mlflow=Mock(tracking_uri="https://dagshub.com/JorgeDataScientist/MLOps_CreditScore.mlflow")
    )

@pytest.fixture
def processed_data():
    """Carga datos procesados simulados."""
    X_train = pd.read_csv(BASE_PATH / "processed" / "X_train.csv")
    X_test = pd.read_csv(BASE_PATH / "processed" / "X_test.csv")
    y_train = pd.read_csv(BASE_PATH / "processed" / "y_train.csv")
    y_test = pd.read_csv(BASE_PATH / "processed" / "y_test.csv")
    return X_train, X_test, y_train["Puntaje_Credito"].values, y_test["Puntaje_Credito"].values

def test_load_data(config):
    """Verifica que load_data carga datos correctamente."""
    X_train, X_test, y_train, y_test = load_data(config)
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_train, np.ndarray)
    assert isinstance(y_test, np.ndarray)
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)

def test_build_pipeline(config):
    """Verifica que build_pipeline crea un Pipeline correcto."""
    pipeline = build_pipeline(config)
    assert isinstance(pipeline, Pipeline)
    assert isinstance(pipeline.named_steps["scaler"], StandardScaler)
    assert isinstance(pipeline.named_steps["model"], RandomForestClassifier)

def test_compute_metrics(processed_data, config):
    """Verifica que compute_metrics calcula métricas correctamente."""
    X_train, X_test, y_train, y_test = processed_data
    model = RandomForestClassifier(random_state=42).fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = compute_metrics(y_test, y_pred, config)
    assert isinstance(metrics, dict)
    assert "test_accuracy" in metrics
    assert "test_f1_macro" in metrics
    assert all(key in metrics for key in ["test_f1_poor", "test_f1_standard", "test_f1_good"])
    assert all(0 <= value <= 1 for value in metrics.values())

def test_save_confusion_matrix(tmp_path, processed_data, config):
    """Verifica que save_confusion_matrix guarda la gráfica correctamente."""
    X_train, X_test, y_train, y_test = processed_data
    model = RandomForestClassifier(random_state=42).fit(X_train, y_train)
    y_pred = model.predict(X_test)
    config.model_config._name = "model_1"
    with patch("src.train.Path", return_value=tmp_path):
        save_confusion_matrix(y_test, y_pred, config)
    assert (tmp_path / "graphics" / "model_1" / "confusion_matrix.png").exists()

def test_save_metrics_bar(tmp_path, processed_data, config):
    """Verifica que save_metrics_bar guarda la gráfica correctamente."""
    X_train, X_test, y_train, y_test = processed_data
    model = RandomForestClassifier(random_state=42).fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = compute_metrics(y_test, y_pred, config)
    config.model_config._name = "model_1"
    with patch("src.train.Path", return_value=tmp_path):
        save_metrics_bar(metrics, config)
    assert (tmp_path / "graphics" / "model_1" / "metrics_bar.png").exists()

def test_save_roc_curve(tmp_path, processed_data, config):
    """Verifica que save_roc_curve guarda la gráfica correctamente."""
    X_train, X_test, y_train, y_test = processed_data
    model = RandomForestClassifier(random_state=42).fit(X_train, y_train)
    y_pred_proba = model.predict_proba(X_test)
    config.model_config._name = "model_1"
    with patch("src.train.Path", return_value=tmp_path):
        save_roc_curve(y_test, y_pred_proba, config)
    assert (tmp_path / "graphics" / "model_1" / "roc_curve.png").exists()

def test_model_saving(tmp_path, processed_data, config):
    """Verifica que train guarda el modelo correctamente."""
    config.model.dir = str(tmp_path / "models")
    config.model.name = "rf_model.pkl"
    config.model.params_name = "params.json"
    config.model_config._name = "model_1"
    with patch("src.train.RandomizedSearchCV.fit", return_value=Mock(best_estimator_=RandomForestClassifier())):
        train(config)
    assert (tmp_path / "models" / "model_1" / "rf_model.pkl").exists()

def test_params_saving(tmp_path, processed_data, config):
    """Verifica que train guarda los hiperparámetros correctamente."""
    config.model.dir = str(tmp_path / "models")
    config.model.name = "rf_model.pkl"
    config.model.params_name = "params.json"
    config.model_config._name = "model_1"
    with patch("src.train.RandomizedSearchCV.fit", return_value=Mock(best_estimator_=RandomForestClassifier(), best_params_={"n_estimators": 100})):
        train(config)
    assert (tmp_path / "models" / "model_1" / "params.json").exists()

@patch("src.utils.BaseLogger")
def test_mlflow_logging(mock_logger, processed_data, config):
    """Verifica que train registra métricas y modelo en MLflow."""
    config.model.dir = "models"
    config.model_config._name = "model_1"
    with patch("src.train.RandomizedSearchCV.fit", return_value=Mock(best_estimator_=RandomForestClassifier(), best_params_={"n_estimators": 100})):
        train(config)
    mock_logger.return_value.log_params.assert_called()
    mock_logger.return_value.log_metrics.assert_called()
    mock_logger.return_value.log_model.assert_called()