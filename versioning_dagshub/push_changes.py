import subprocess
import yaml
from pathlib import Path
from datetime import datetime

# Leer versioning_dagshub.yaml desde el directorio config
config_path = Path(__file__).parent.parent / "config" / "versioning_dagshub.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Extraer datos
dvc_dirs = config["dvc"]["directories"]
username = config["dagshub"]["username"]
email = config["git"]["email"]
creator_name = "Jorge Luis Garcia"

# Solicitar mensaje del commit
commit_message = input("Ingresa el mensaje del commit: ")

# DVC: Actualizar directorios
for directory in dvc_dirs:
    dir_path = Path(__file__).parent.parent / directory
    if dir_path.exists():
        subprocess.run(f"dvc add {directory}", shell=True, check=True)
        print(f"{directory} actualizado en DVC.")
    else:
        print(f"{directory} no existe, se omite.")

# Git: Añadir todos los cambios
subprocess.run("git add .", shell=True, check=True)

# Escribir en history.txt antes del commit
history_path = Path(__file__).parent / "history.txt"
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
log_entry = (
    f"{timestamp} - Commit: \"{commit_message}\" - Hash: [TBD] - "
    f"Creador: {creator_name} - Email: {email} - Username DAGsHub: {username}\n"
)
with open(history_path, "a") as f:
    f.write(log_entry)
subprocess.run("git add versioning_dagshub/history.txt", shell=True, check=True)

# Commitear todos los cambios
result = subprocess.run("git status --porcelain", shell=True, capture_output=True, text=True)
if result.stdout.strip():
    subprocess.run(f'git commit -m "{commit_message}"', shell=True, check=True)
    print("Cambios commiteados en Git.")
else:
    print("No hay cambios para commitear.")
    exit(0)

# Actualizar hash en history.txt
commit_hash = subprocess.check_output("git log -1 --pretty=format:%h", shell=True).decode().strip()
with open(history_path, "r") as f:
    lines = f.readlines()
lines[-1] = (
    f"{timestamp} - Commit: \"{commit_message}\" - Hash: {commit_hash} - "
    f"Creador: {creator_name} - Email: {email} - Username DAGsHub: {username}\n"
)
with open(history_path, "w") as f:
    f.writelines(lines)
subprocess.run("git add versioning_dagshub/history.txt", shell=True, check=True)
subprocess.run(f'git commit -m "Actualizar hash en history.txt: {commit_message}"', shell=True, check=True)
print("Registro actualizado en history.txt.")

# Subir a DAGsHub
subprocess.run("git push origin master", shell=True, check=True)
subprocess.run("dvc push", shell=True, check=True)
print("Cambios subidos a DAGsHub.")