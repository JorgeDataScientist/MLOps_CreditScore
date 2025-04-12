import subprocess
import yaml
from pathlib import Path
import os

# Leer versioning_github.yaml desde el directorio config en la raíz
config_path = Path(__file__).parent.parent / "config" / "versioning_github.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Extraer directorios a versionar
dvc_dirs = config["dvc"]["directories"]
root_dir = Path(__file__).parent.parent

# Asegurarse de estar en la rama main
subprocess.run("git checkout main", shell=True, check=True)
print("En rama main.")

# Cambiar al directorio raíz
os.chdir(root_dir)

# Inicializar DVC si no está inicializado
if not (root_dir / ".dvc").exists():
    subprocess.run("dvc init", shell=True, check=True)
    print("DVC inicializado.")
else:
    print("DVC ya está inicializado.")

# Añadir .dvc a Git
subprocess.run("git add .dvc", shell=True, check=True)
result = subprocess.run("git status --porcelain .dvc", shell=True, capture_output=True, text=True)
if result.stdout.strip():
    subprocess.run('git commit -m "dvc init para GitHub"', shell=True, check=True)
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

# Añadir archivos .dvc, .gitignore y el propio script a Git
dvc_files = " ".join(f"{d}.dvc" for d in dvc_dirs if (root_dir / d).exists())
if dvc_files:
    subprocess.run(f"git add {dvc_files} .gitignore versioning_github/init_dvc_github.py", shell=True, check=True)
    subprocess.run('git commit -m "Añadir archivos rastreados con DVC para GitHub" --allow-empty', shell=True, check=True)
    print("Archivos .dvc y script commiteados.")
else:
    print("No hay archivos .dvc para commitear.")