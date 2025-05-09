"""Pruebas unitarias para train.py.

Verifica funciones de carga de datos, construcción del pipeline, cálculo de métricas,
guardado de la matriz de confusión y gráfica de métricas.

Dependencias:
    - pytest: Para ejecutar pruebas.
    - pandas: Para manipulación de datos.
    - sklearn: Para pipeline, modelos y métricas.
    - unittest.mock: Para simular configuraciones.
    - pathlib: Para manejo de rutas.
"""

import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Añadir el directorio raíz al sys.path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

import pytest
import pandas as pd
import numpy as np
from omegaconf import OmegaConf
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from src.train import load_data, build_pipeline, save_confusion_matrix, save_metrics_bar
from src.utils import compute_metrics
import matplotlib
matplotlib.use('Agg')  # Usar backend no interactivo para evitar errores de Tkinter

# Ruta base para datos simulados
BASE_PATH = ROOT_DIR / "tests" / "data"

@pytest.fixture
def config(tmp_path):
    """Crea una configuración simulada para pruebas."""
    search_space = OmegaConf.create({"n_estimators": [100, 200], "max_depth": [None, 10, 20]})
    cv_config = Mock(folds=5, scoring="f1_macro")
    cv_config.__str__ = lambda self: "f1_macro"
    params_mock = Mock(random_state=42)  # Mock para params con atributo random_state
    params_mock.__getitem__ = lambda self, key: {"random_state": 42}[key]  # Simula dict para **params
    params_mock.keys = lambda: ["random_state"]  # Necesario para **params
    return Mock(
        processed=Mock(
            X_train=Mock(path=str(BASE_PATH / "processed" / "X_train.csv")),
            X_test=Mock(path=str(BASE_PATH / "processed" / "X_test.csv")),
            y_train=Mock(path=str(BASE_PATH / "processed" / "y_train.csv")),
            y_test=Mock(path=str(BASE_PATH / "processed" / "y_test.csv"))
        ),
        model_config=Mock(
            _name="model_1",
            params=params_mock,
            search_space=search_space,
            cv=cv_config,
            optimization=Mock(n_iter=10),
            metrics=["accuracy", "f1_macro", "f1_per_class"]
        ),
        model=Mock(dir=str(tmp_path / "models"), name="rf_model.pkl", params_name="params.json"),
        process=Mock(
            features=["Salario_Mensual", "Deuda_Pendiente", "Edad", "debt_to_income"],
            target="Puntaje_Credito",
            target_classes=["Good", "Standard"]
        ),
        mlflow=Mock(tracking_uri="https://dagshub.com/JorgeDataScientist/MLOps_CreditScore.mlflow")
    )

@pytest.fixture
def processed_data():
    """Carga datos procesados simulados y filtra clases."""
    X_train = pd.read_csv(BASE_PATH / "processed" / "X_train.csv")
    X_test = pd.read_csv(BASE_PATH / "processed" / "X_test.csv")
    y_train = pd.read_csv(BASE_PATH / "processed" / "y_train.csv")
    y_test = pd.read_csv(BASE_PATH / "processed" / "y_test.csv")
    valid_classes = ["Good", "Standard"]
    y_train = y_train[y_train["Puntaje_Credito"].isin(valid_classes)]["Puntaje_Credito"].values
    y_test = y_test[y_test["Puntaje_Credito"].isin(valid_classes)]["Puntaje_Credito"].values
    train_mask = pd.read_csv(BASE_PATH / "processed" / "y_train.csv")["Puntaje_Credito"].isin(valid_classes)
    test_mask = pd.read_csv(BASE_PATH / "processed" / "y_test.csv")["Puntaje_Credito"].isin(valid_classes)
    X_train = X_train[train_mask]
    X_test = X_test[test_mask]
    return X_train, X_test, y_train, y_test

def test_load_data(config, tmp_path):
    """Verifica que load_data carga datos correctamente."""
    with patch("src.train.get_original_cwd", return_value=str(tmp_path)):
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
    assert all(key in metrics for key in ["test_f1_good", "test_f1_standard"])

def test_save_confusion_matrix(tmp_path, processed_data, config):
    """Verifica que save_confusion_matrix guarda la gráfica correctamente."""
    X_train, X_test, y_train, y_test = processed_data
    model = RandomForestClassifier(random_state=42).fit(X_train, y_train)
    y_pred = model.predict(X_test)
    config.model_config._name = "model_1"
    with patch("src.train.Path", return_value=tmp_path):
        with patch("src.train.get_original_cwd", return_value=str(tmp_path)):
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
        with patch("src.train.get_original_cwd", return_value=str(tmp_path)):
            save_metrics_bar(metrics, config)
    assert (tmp_path / "graphics" / "model_1" / "metrics_bar.png").exists()