import subprocess
import yaml
from pathlib import Path

# Leer versioning_github.yaml desde el directorio config en la raíz
config_path = Path(__file__).parent.parent / "config" / "versioning_github.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Extraer datos
username = config["github"]["username"]
email = config["git"]["email"]
git_url = config["github"]["git_url"]

# Verificar si Git está inicializado
root_dir = Path(__file__).parent.parent
if not (root_dir / ".git").exists():
    print("Error: No se encontró un repositorio Git. Inicializa Git primero.")
    exit(1)

print("Git ya está inicializado.")

# Configurar usuario
subprocess.run(f'git config user.email "{email}"', shell=True, check=True)
subprocess.run(f'git config user.name "{username}"', shell=True, check=True)
print("Usuario configurado.")

# Añadir remoto GitHub
result = subprocess.run("git remote", shell=True, capture_output=True, text=True)
if "github" not in result.stdout:
    subprocess.run(f"git remote add github {git_url}", shell=True, check=True)
    print("Remoto GitHub configurado.")
else:
    print("Remoto GitHub ya estaba configurado.")

# Crear rama main si no existe
result = subprocess.run("git branch --list main", shell=True, capture_output=True, text=True)
if not result.stdout.strip():
    subprocess.run("git checkout -b main", shell=True, check=True)
    print("Rama main creada.")
else:
    subprocess.run("git checkout main", shell=True, check=True)
    print("Cambiado a rama main.")

# Añadir y commitear (usando el .gitignore existente)
result = subprocess.run("git status --porcelain", shell=True, capture_output=True, text=True)
if result.stdout.strip():
    subprocess.run("git add .", shell=True, check=True)
    subprocess.run('git commit -m "Initial commit para GitHub"', shell=True, check=True)
    print("Commit inicial realizado en rama main.")
else:
    print("No hay cambios para commitear en rama main.")