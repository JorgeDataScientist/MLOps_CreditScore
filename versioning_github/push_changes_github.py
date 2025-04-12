import subprocess
import yaml
from pathlib import Path
from datetime import datetime

# Leer versioning_github.yaml desde el directorio config en la raíz
config_path = Path(__file__).parent.parent / "config" / "versioning_github.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Extraer datos
dvc_dirs = config["dvc"]["directories"]
username = config["github"]["username"]
email = config["git"]["email"]
creator_name = "Jorge Luis Garcia"

# Asegurarse de estar en la rama main
result = subprocess.run("git status --porcelain", shell=True, capture_output=True, text=True)
if result.stdout.strip():
    print("Hay cambios no commiteados. Por favor, commitea o guarda los cambios antes de cambiar a main.")
    exit(1)

try:
    subprocess.run("git checkout main", shell=True, check=True)
    print("En rama main.")
except subprocess.CalledProcessError:
    print("Error: No se pudo cambiar a la rama main. Asegúrate de que exista.")
    exit(1)

# Solicitar mensaje del commit
commit_message = input("Ingresa el mensaje del commit: ")

# Verificar cambios en Git
result = subprocess.run("git status --porcelain", shell=True, capture_output=True, text=True)
changes_to_commit = result.stdout.strip()

# DVC: Actualizar directorios
for directory in dvc_dirs:
    if (Path(__file__).parent.parent / directory).exists():
        subprocess.run(f"dvc add {directory}", shell=True, check=True)
        print(f"{directory} actualizado en DVC.")
    else:
        print(f"{directory} no existe, se omite.")

# Git: Añadir archivos .dvc
dvc_files = " ".join(f"{d}.dvc" for d in dvc_dirs if (Path(__file__).parent.parent / d).exists())
if dvc_files:
    subprocess.run(f"git add {dvc_files}", shell=True, check=True)

# Escribir en history.txt antes del commit
history_path = Path(__file__).parent / "history.txt"
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
log_entry = (
    f"{timestamp} - Commit: \"{commit_message}\" - Hash: [pending] - "
    f"Creador: {creator_name} - Email: {email} - Username GitHub: {username}\n"
)
with open(history_path, "a") as f:
    f.write(log_entry)
print("Registro añadido a history.txt.")

# Añadir history.txt y otros cambios a Git
subprocess.run("git add .", shell=True, check=True)

# Commitear todos los cambios
if changes_to_commit or dvc_files:
    subprocess.run(f'git commit -m "{commit_message}"', shell=True, check=True)
    print("Cambios commiteados en Git.")
else:
    subprocess.run(f'git commit -m "{commit_message}" --allow-empty', shell=True, check=True)
    print("Cambios commiteados en Git (commit vacío).")

# Obtener hash del commit
commit_hash = subprocess.check_output("git log -1 --pretty=format:%h", shell=True).decode().strip()

# Actualizar history.txt con el hash correcto
with open(history_path, "r") as f:
    lines = f.readlines()
lines[-1] = (
    f"{timestamp} - Commit: \"{commit_message}\" - Hash: {commit_hash} - "
    f"Creador: {creator_name} - Email: {email} - Username GitHub: {username}\n"
)
with open(history_path, "w") as f:
    f.writelines(lines)

# Subir a GitHub
subprocess.run("git push github main", shell=True, check=True)
subprocess.run("dvc push", shell=True, check=True)
print("Cambios subidos a GitHub.")