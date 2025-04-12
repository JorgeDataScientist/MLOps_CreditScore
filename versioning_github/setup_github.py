import subprocess
import yaml
from pathlib import Path

# Leer versioning_github.yaml desde el directorio config en la raíz
config_path = Path(__file__).parent.parent / "config" / "versioning_github.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Extraer datos
git_url = config["github"]["git_url"]

# Asegurarse de estar en la rama main
subprocess.run("git checkout main", shell=True, check=True)
print("En rama main.")

# Verificar remoto GitHub
result = subprocess.run("git remote", shell=True, capture_output=True, text=True)
if "github" not in result.stdout:
    subprocess.run(f"git remote add github {git_url}", shell=True, check=True)
    print("Remoto GitHub configurado.")
else:
    print("Remoto GitHub ya está configurado.")

# Configurar remoto DVC local
dvc_remote_path = Path(__file__).parent.parent / "dvc_storage_github"
subprocess.run(f'dvc remote add github-local "{dvc_remote_path}" --force', shell=True, check=True)
subprocess.run("dvc remote default github-local", shell=True, check=True)
print("Remoto DVC local configurado para GitHub.")

# Commitear cambios de DVC
result = subprocess.run("git status --porcelain .dvc/config", shell=True, capture_output=True, text=True)
if result.stdout.strip():
    subprocess.run("git add .dvc/config", shell=True, check=True)
    subprocess.run('git commit -m "Configurar remoto DVC local para GitHub"', shell=True, check=True)
    print("Configuración DVC commiteada.")
else:
    print("No hay cambios en .dvc/config para commitear.")

# Subir a GitHub
subprocess.run("git push github main", shell=True, check=True)
subprocess.run("dvc push", shell=True, check=True)
print("Proyecto subido a GitHub.")