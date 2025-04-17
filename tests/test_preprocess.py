"""Pruebas unitarias para preprocess.py.

Verifica funciones de carga, limpieza, transformación, creación de features
y procesamiento de datos.

Dependencias:
    - pytest: Para ejecutar pruebas.
    - pandas: Para manipulación de datos.
    - pathlib: Para manejar rutas.
    - unittest.mock: Para simular configuraciones.
"""

import sys
from pathlib import Path

# Añadir el directorio raíz al sys.path
ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))

import pytest
import pandas as pd
from unittest.mock import Mock
from src.preprocess import get_data, clean_data, transform_data, create_new_features, process_data, rename_columns, handle_missing_values, encode_categorical

# Ruta base para datos simulados
BASE_PATH = ROOT_DIR / "tests" / "data"

@pytest.fixture
def config():
    """Crea una configuración simulada para pruebas."""
    return Mock(
        raw=Mock(path=str(BASE_PATH / "raw" / "credit_data.csv")),
        processed=Mock(
            dir=str(BASE_PATH / "processed"),
            X_train=Mock(name="X_train.csv"),
            X_test=Mock(name="X_test.csv"),
            y_train=Mock(name="y_train.csv"),
            y_test=Mock(name="y_test.csv")
        ),
        process=Mock(
            target="Puntaje_Credito",
            test_size=0.2,
            random_state=42,
            translations={},
            cleaning=Mock(min_age=18, max_age_credit_ratio=10.0, drop_columns=[]),
            encoding={},
            new_features=[
                Mock(name="debt_to_income", formula={}),
                Mock(name="payment_to_income", formula={}),
                Mock(name="credit_history_ratio", formula={})
            ]
        )
    )

@pytest.fixture
def raw_data():
    """Carga datos simulados."""
    return pd.read_csv(BASE_PATH / "raw" / "credit_data.csv")

def test_get_data(config, tmp_path):
    """Verifica que get_data carga datos correctamente.

    Args:
        config: Configuración simulada.
        tmp_path: Directorio temporal proporcionado por pytest.

    Returns:
        None

    Raises:
        AssertionError: Si los datos no se cargan o no tienen las columnas esperadas.
    """
    with pytest.MonkeyPatch.context() as m:
        m.setattr("src.preprocess.get_original_cwd", lambda: str(tmp_path))
        (tmp_path / "tests" / "data" / "raw").mkdir(parents=True)
        pd.DataFrame({"Salario_Mensual": [5000], "Puntaje_Credito": ["Good"]}).to_csv(
            tmp_path / "tests" / "data" / "raw" / "credit_data.csv", index=False
        )
        df = get_data(config.raw.path)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert "Salario_Mensual" in df.columns
    assert "Puntaje_Credito" in df.columns

def test_clean_data(raw_data, config):
    """Verifica que clean_data elimina valores nulos y aplica filtros.

    Args:
        raw_data: DataFrame simulado.
        config: Configuración simulada.

    Returns:
        None

    Raises:
        AssertionError: Si los datos no se limpian correctamente.
    """
    cleaned_df = clean_data(raw_data, config)
    assert cleaned_df.isna().sum().sum() == 0
    assert cleaned_df["Edad"].min() >= config.process.cleaning.min_age
    assert "age_credit_ratio" not in cleaned_df.columns

def test_transform_data(raw_data, config):
    """Verifica que transform_data codifica y selecciona columnas.

    Args:
        raw_data: DataFrame simulado.
        config: Configuración simulada.

    Returns:
        None

    Raises:
        AssertionError: Si la transformación no es correcta.
    """
    cleaned_df = clean_data(raw_data, config)
    transformed_df = transform_data(cleaned_df, config)
    assert "Puntaje_Credito" in transformed_df.columns
    assert all(col in transformed_df.columns for col in ["Salario_Mensual", "Deuda_Pendiente", "Edad"])

def test_create_new_features(raw_data, config):
    """Verifica que create_new_features genera nuevas columnas.

    Args:
        raw_data: DataFrame simulado.
        config: Configuración simulada.

    Returns:
        None

    Raises:
        AssertionError: Si las nuevas features no se crean.
    """
    cleaned_df = clean_data(raw_data, config)
    transformed_df = transform_data(cleaned_df, config)
    featured_df = create_new_features(transformed_df, config.process.new_features)
    assert "debt_to_income" in featured_df.columns
    assert "payment_to_income" in featured_df.columns
    assert "credit_history_ratio" in featured_df.columns

def test_process_data(config, tmp_path):
    """Verifica que process_data genera archivos procesados.

    Args:
        config: Configuración simulada.
        tmp_path: Directorio temporal proporcionado por pytest.

    Returns:
        None

    Raises:
        AssertionError: Si los archivos no se guardan.
    """
    with pytest.MonkeyPatch.context() as m:
        m.setattr("src.preprocess.get_original_cwd", lambda: str(tmp_path))
        (tmp_path / "tests" / "data" / "raw").mkdir(parents=True)
        pd.DataFrame({
            "Salario_Mensual": [5000, 6000],
            "Deuda_Pendiente": [2000, 2500],
            "Edad": [30, 40],
            "Num_Cuentas_Bancarias": [2, 3],
            "Num_Prestamos": [1, 2],
            "Edad_Historial_Credito": [24, 36],
            "Total_Cuota_Mensual": [500, 600],
            "Puntaje_Credito": ["Good", "Standard"]
        }).to_csv(tmp_path / "tests" / "data" / "raw" / "credit_data.csv", index=False)
        config.processed.dir = str(tmp_path / "tests" / "data" / "processed")
        process_data(config)
    assert (tmp_path / "tests" / "data" / "processed" / "X_train.csv").exists()
    assert (tmp_path / "tests" / "data" / "processed" / "X_test.csv").exists()
    assert (tmp_path / "tests" / "data" / "processed" / "y_train.csv").exists()
    assert (tmp_path / "tests" / "data" / "processed" / "y_test.csv").exists()

def test_rename_columns(raw_data):
    """Verifica que rename_columns renombra columnas correctamente.

    Args:
        raw_data: DataFrame simulado.

    Returns:
        None

    Raises:
        AssertionError: Si las columnas no se renombran.
    """
    translations = {"Salario_Mensual": "Ingreso_Mensual"}
    renamed_df = rename_columns(raw_data, translations)
    assert "Ingreso_Mensual" in renamed_df.columns
    assert "Salario_Mensual" not in renamed_df.columns

def test_handle_missing_values(raw_data):
    """Verifica que handle_missing_values imputa valores nulos.

    Args:
        raw_data: DataFrame simulado.

    Returns:
        None

    Raises:
        AssertionError: Si los valores nulos no se manejan.
    """
    cleaned_df = handle_missing_values(raw_data)
    assert cleaned_df.isna().sum().sum() == 0

def test_encode_categorical(raw_data):
    """Verifica que encode_categorical codifica columnas categóricas.

    Args:
        raw_data: DataFrame simulado.

    Returns:
        None

    Raises:
        AssertionError: Si la codificación no es correcta.
    """
    encoded_df = encode_categorical(raw_data, "Puntaje_Credito", drop="Poor")
    assert "Puntaje_Credito_Standard" in encoded_df.columns
    assert "Puntaje_Credito_Good" in encoded_df.columns
    assert "Puntaje_Credito_Poor" not in encoded_df.columns
    assert "Puntaje_Credito" not in encoded_df.columns