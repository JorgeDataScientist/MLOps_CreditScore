"""Pruebas unitarias para run_pipeline.py.

Verifica funciones que ejecutan el pipeline completo y sus etapas.

Dependencias:
    - pytest: Para ejecutar pruebas.
    - unittest.mock: Para simular subprocess y sistema de archivos.
    - pathlib: Para manejo de rutas.
"""

import pytest
from unittest.mock import patch, Mock
from pathlib import Path
import os
import subprocess
from src.run_pipeline import run_stage, run_pipeline

@pytest.fixture
def mock_subprocess_run():
    """Simula subprocess.run."""
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = Mock(stdout="Success", stderr="")
        yield mock_run

@pytest.fixture
def mock_path():
    """Simula Path.cwd."""
    with patch("pathlib.Path.cwd", return_value=Path("/mock/path")):
        yield

@pytest.fixture
def mock_os_environ():
    """Simula os.environ para el entorno virtual."""
    with patch("os.environ", {"VIRTUAL_ENV": "/mock/path/env"}):
        yield

def test_run_stage_success(mock_subprocess_run, mock_path, mock_os_environ):
    """Verifica que run_stage ejecuta un script correctamente."""
    run_stage("preprocess.py")
    expected_python_path = os.path.join("/mock/path/env", "Scripts", "python.exe")
    expected_script_path = str(Path("/mock/path/src/preprocess.py"))
    mock_subprocess_run.assert_called_once_with(
        [expected_python_path, expected_script_path],
        check=True, capture_output=True, text=True, encoding="utf-8"
    )

def test_run_stage_failure(mock_subprocess_run, mock_path, mock_os_environ):
    """Verifica que run_stage maneja errores de subprocess."""
    mock_subprocess_run.side_effect = subprocess.CalledProcessError(
        1, cmd=["python", "preprocess.py"], stderr="Error"
    )
    with pytest.raises(subprocess.CalledProcessError):
        run_stage("preprocess.py")

def test_run_stage_encoding_error(mock_subprocess_run, mock_path, mock_os_environ):
    """Verifica que run_stage maneja errores de codificación."""
    mock_subprocess_run.side_effect = UnicodeDecodeError("utf-8", b"", 0, 1, "invalid")
    with pytest.raises(UnicodeDecodeError):
        run_stage("preprocess.py")

def test_run_pipeline(mock_subprocess_run, mock_path, mock_os_environ):
    """Verifica que run_pipeline ejecuta todas las etapas."""
    run_pipeline()
    assert mock_subprocess_run.call_count == 4
    expected_calls = [
        ([os.path.join("/mock/path/env", "Scripts", "python.exe"), str(Path("/mock/path/src/preprocess.py"))],),
        ([os.path.join("/mock/path/env", "Scripts", "python.exe"), str(Path("/mock/path/src/train.py"))],),
        ([os.path.join("/mock/path/env", "Scripts", "python.exe"), str(Path("/mock/path/src/evaluate.py"))],),
        ([os.path.join("/mock/path/env", "Scripts", "python.exe"), str(Path("/mock/path/src/predict.py"))],),
    ]
    for i, call in enumerate(mock_subprocess_run.call_args_list):
        assert call.args[0] == expected_calls[i][0]
        assert call.kwargs == {"check": True, "capture_output": True, "text": True, "encoding": "utf-8"}