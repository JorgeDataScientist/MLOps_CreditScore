import subprocess
import yaml
from pathlib import Path
import os

# Leer versioning_dagshub.yaml desde el directorio config en la raíz
config_path = Path(__file__).parent.parent / "config" / "versioning_dagshub.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Extraer directorios a versionar
dvc_dirs = config["dvc"]["directories"]
root_dir = Path(__file__).parent.parent

# Cambiar al directorio raíz
os.chdir(root_dir)

# Inicializar DVC si no está inicializado
if not (root_dir / ".dvc").exists():
    subprocess.run("dvc init", shell=True, check=True)
    print("DVC inicializado.")
else:
    print("DVC ya estaba inicializado.")

# Añadir a Git y commitear solo si hay cambios
subprocess.run("git add .dvc", shell=True, check=True)
result = subprocess.run("git status --porcelain .dvc", shell=True, capture_output=True, text=True)
if result.stdout.strip():
    subprocess.run('git commit -m "dvc init"', shell=True, check=True)
    print("DVC commiteado.")
else:
    print("No hay cambios en .dvc para commitear.")

# Versionar directorios con DVC si existen
for directory in dvc_dirs:
    if (root_dir / directory).exists():
        subprocess.run(f"dvc add {directory}", shell=True, check=True)
        print(f"{directory} añadido a DVC.")
    else:
        print(f"{directory} no existe, se omite.")

# Añadir archivos .dvc y .gitignore a Git
dvc_files = " ".join(f"{d}.dvc" for d in dvc_dirs if (root_dir / d).exists())
if dvc_files:
    subprocess.run(f"git add {dvc_files} .gitignore", shell=True, check=True)
    subprocess.run('git commit -m "Añadir archivos rastreados con DVC" --allow-empty', shell=True, check=True)
    print("Archivos .dvc commiteados.")
else:
    print("No hay archivos .dvc para commitear.")