"""Pruebas unitarias para preprocess.py.

Verifica funciones de carga, limpieza y transformación de datos.

Dependencias:
    - pytest: Para ejecutar pruebas.
    - pandas: Para manipulación de datos.
    - unittest.mock: Para simular configuraciones y rutas.
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
from src.preprocess import (
    get_data,
    rename_columns,
    strip_strings,
    handle_missing_values,
    filter_minimum_values,
    filter_by_age,
    filter_by_age_credit_ratio,
    drop_columns,
    encode_categorical,
    select_final_columns
)

@pytest.fixture
def config():
    """Crea una configuración simulada para pruebas."""
    return Mock(
        raw=Mock(path="data/raw/credit_score_raw.csv"),
        process=Mock(
            translations={
                "Edad": "Age",
                "Salario_Mensual": "Monthly_Salary",
                "Puntaje_Credito": "Credit_Score"
            },
            cleaning=Mock(
                min_age=18,
                max_age_credit_ratio=2.0,
                drop_columns=["ID", "Nombre"]
            )
        )
    )

def test_get_data(tmp_path, config):
    """Verifica que get_data carga un archivo CSV correctamente."""
    csv_path = tmp_path / "data" / "raw" / "credit_score_raw.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_expected = pd.DataFrame({
        "Edad": [25, 30],
        "Salario_Mensual": [5000.0, 6000.0],
        "Puntaje_Credito": ["Good", "Standard"]
    })
    df_expected.to_csv(csv_path, index=False)
    
    with patch("src.preprocess.get_original_cwd", return_value=str(tmp_path)):
        df = get_data(config.raw.path)
    
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 3)
    assert list(df.columns) == ["Edad", "Salario_Mensual", "Puntaje_Credito"]
    pd.testing.assert_frame_equal(df, df_expected)

def test_rename_columns(config):
    """Verifica que rename_columns renombra columnas según el diccionario de traducciones."""
    df_input = pd.DataFrame({
        "Edad": [25, 30],
        "Salario_Mensual": [5000.0, 6000.0],
        "Puntaje_Credito": ["Good", "Standard"],
        "Extra_Col": [1, 2]
    })
    translations = config.process.translations
    
    df_result = rename_columns(df_input, translations)
    
    assert isinstance(df_result, pd.DataFrame)
    assert df_result.shape == (2, 4)
    assert list(df_result.columns) == ["Age", "Monthly_Salary", "Credit_Score", "Extra_Col"]
    assert df_result["Age"].equals(df_input["Edad"])
    assert df_result["Monthly_Salary"].equals(df_input["Salario_Mensual"])
    assert df_result["Credit_Score"].equals(df_input["Puntaje_Credito"])
    assert df_result["Extra_Col"].equals(df_input["Extra_Col"])

def test_strip_strings():
    """Verifica que strip_strings elimina espacios en blanco de columnas de tipo string."""
    df_input = pd.DataFrame({
        "Puntaje_Credito": [" Good ", "Standard "],
        "Edad": [25, 30],
        "Nombre": ["  Juan  ", " Maria "]
    })
    df_expected = pd.DataFrame({
        "Puntaje_Credito": ["Good", "Standard"],
        "Edad": [25, 30],
        "Nombre": ["Juan", "Maria"]
    })
    
    df_result = strip_strings(df_input)
    
    assert isinstance(df_result, pd.DataFrame)
    assert df_result.shape == (2, 3)
    assert list(df_result.columns) == ["Puntaje_Credito", "Edad", "Nombre"]
    pd.testing.assert_frame_equal(df_result, df_expected)

def test_handle_missing_values():
    """Verifica que handle_missing_values imputa valores nulos correctamente."""
    df_input = pd.DataFrame({
        "Edad": [25, np.nan, 30],
        "Salario_Mensual": [5000.0, np.nan, 6000.0],
        "Puntaje_Credito": ["Good", np.nan, "Standard"]
    })
    df_expected = pd.DataFrame({
        "Edad": [25, 27.5, 30],
        "Salario_Mensual": [5000.0, 5500.0, 6000.0],
        "Puntaje_Credito": ["Good", "unknown", "Standard"]
    })
    
    df_result = handle_missing_values(df_input)
    
    assert isinstance(df_result, pd.DataFrame)
    assert df_result.shape == (3, 3)
    assert list(df_result.columns) == ["Edad", "Salario_Mensual", "Puntaje_Credito"]
    assert not df_result.isna().any().any()
    pd.testing.assert_frame_equal(df_result, df_expected)

def test_filter_minimum_values():
    """Verifica que filter_minimum_values elimina filas con valores no positivos."""
    df_input = pd.DataFrame({
        "Num_Cuentas_Bancarias": [1, 0, 2],
        "Num_Prestamos": [2, 1, 0],
        "Edad": [25, 30, 35],
        "Puntaje_Credito": ["Good", "Standard", "Good"]
    })
    df_expected = pd.DataFrame({
        "Num_Cuentas_Bancarias": [1],
        "Num_Prestamos": [2],
        "Edad": [25],
        "Puntaje_Credito": ["Good"]
    })
    
    df_result = filter_minimum_values(df_input)
    
    assert isinstance(df_result, pd.DataFrame)
    assert df_result.shape == (1, 4)
    assert list(df_result.columns) == ["Num_Cuentas_Bancarias", "Num_Prestamos", "Edad", "Puntaje_Credito"]
    pd.testing.assert_frame_equal(df_result.reset_index(drop=True), df_expected.reset_index(drop=True))

def test_filter_by_age(config):
    """Verifica que filter_by_age filtra filas con edad menor a la mínima."""
    df_input = pd.DataFrame({
        "Edad": [16, 18, 25],
        "Salario_Mensual": [3000.0, 5000.0, 6000.0],
        "Puntaje_Credito": ["Standard", "Good", "Good"]
    })
    df_expected = pd.DataFrame({
        "Edad": [18, 25],
        "Salario_Mensual": [5000.0, 6000.0],
        "Puntaje_Credito": ["Good", "Good"]
    })
    
    df_result = filter_by_age(df_input, config.process.cleaning.min_age)
    
    assert isinstance(df_result, pd.DataFrame)
    assert df_result.shape == (2, 3)
    assert list(df_result.columns) == ["Edad", "Salario_Mensual", "Puntaje_Credito"]
    pd.testing.assert_frame_equal(df_result.reset_index(drop=True), df_expected.reset_index(drop=True))

def test_filter_by_age_credit_ratio(config):
    """Verifica que filter_by_age_credit_ratio filtra filas según la relación edad/historial crediticio."""
    df_input = pd.DataFrame({
        "Edad": [25, 30, 20],
        "Edad_Historial_Credito": [60, 12, 120],
        "Puntaje_Credito": ["Good", "Standard", "Good"]
    })
    df_expected = pd.DataFrame({
        "Edad": [20],
        "Edad_Historial_Credito": [120],
        "Puntaje_Credito": ["Good"]
    })
    
    df_result = filter_by_age_credit_ratio(df_input, config.process.cleaning.max_age_credit_ratio)
    
    assert isinstance(df_result, pd.DataFrame)
    assert df_result.shape == (1, 3)
    assert list(df_result.columns) == ["Edad", "Edad_Historial_Credito", "Puntaje_Credito"]
    assert "age_credit_ratio" not in df_result.columns
    pd.testing.assert_frame_equal(df_result.reset_index(drop=True), df_expected.reset_index(drop=True))

def test_drop_columns(config):
    """Verifica que drop_columns elimina las columnas especificadas."""
    df_input = pd.DataFrame({
        "Edad": [25, 30],
        "ID": [1, 2],
        "Nombre": ["Juan", "Maria"],
        "Puntaje_Credito": ["Good", "Standard"]
    })
    df_expected = pd.DataFrame({
        "Edad": [25, 30],
        "Puntaje_Credito": ["Good", "Standard"]
    })
    
    df_result = drop_columns(df_input, config.process.cleaning.drop_columns)
    
    assert isinstance(df_result, pd.DataFrame)
    assert df_result.shape == (2, 2)
    assert list(df_result.columns) == ["Edad", "Puntaje_Credito"]
    pd.testing.assert_frame_equal(df_result.reset_index(drop=True), df_expected.reset_index(drop=True))

def test_encode_categorical():
    """Verifica que encode_categorical codifica una columna categórica con OneHotEncoder."""
    df_input = pd.DataFrame({
        "Ocupacion": ["Ingeniero", "Profesor"],
        "Edad": [25, 30]
    })
    df_expected = pd.DataFrame({
        "Edad": [25, 30],
        "Ocupacion_Profesor": [0.0, 1.0]
    })
    
    df_result = encode_categorical(df_input, "Ocupacion", drop="first")
    
    assert isinstance(df_result, pd.DataFrame)
    assert df_result.shape == (2, 2)
    assert list(df_result.columns) == ["Edad", "Ocupacion_Profesor"]
    pd.testing.assert_frame_equal(df_result.reset_index(drop=True), df_expected.reset_index(drop=True))

def test_select_final_columns():
    """Verifica que select_final_columns selecciona columnas numéricas, codificadas y el target."""
    df_input = pd.DataFrame({
        "Edad": [25, 30],
        "Salario_Mensual": [5000.0, 6000.0],
        "Puntaje_Credito": ["Good", "Standard"],
        "Ocupacion_Ingeniero": [1.0, 0.0],
        "Ocupacion_Profesor": [0.0, 1.0],
        "Nombre": ["Juan", "Maria"]  # Columna no numérica ni codificada
    })
    df_expected = pd.DataFrame({
        "Edad": [25, 30],
        "Salario_Mensual": [5000.0, 6000.0],
        "Ocupacion_Ingeniero": [1.0, 0.0],
        "Ocupacion_Profesor": [0.0, 1.0],
        "Puntaje_Credito": ["Good", "Standard"]
    })
    
    df_result = select_final_columns(df_input, "Puntaje_Credito")
    
    assert isinstance(df_result, pd.DataFrame)
    assert df_result.shape == (2, 5)
    assert list(df_result.columns) == [
        "Edad",
        "Salario_Mensual",
        "Ocupacion_Ingeniero",
        "Ocupacion_Profesor",
        "Puntaje_Credito"
    ]
    pd.testing.assert_frame_equal(df_result.reset_index(drop=True), df_expected.reset_index(drop=True))