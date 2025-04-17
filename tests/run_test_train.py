"""Script para ejecutar pruebas de test_train.py."""

import pytest

def main():
    """Ejecuta las pruebas de test_train.py."""
    pytest.main(["tests/test_train.py", "-v", "-W", "ignore::DeprecationWarning"])

if __name__ == "__main__":
    main()