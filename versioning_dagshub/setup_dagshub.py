import subprocess
import yaml
from pathlib import Path

# Leer versioning_dagshub.yaml desde el directorio config en la raíz
config_path = Path(__file__).parent.parent / "config" / "versioning_dagshub.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Extraer datos
username = config["dagshub"]["username"]
token = config["dagshub"]["token"]
git_url = config["dagshub"]["git_url"]
dvc_s3_url = config["dagshub"]["dvc_s3_url"]
dvc_endpoint_url = config["dagshub"]["dvc_endpoint_url"]

# Configurar remoto Git si no existe
result = subprocess.run("git remote", shell=True, capture_output=True, text=True)
if "origin" not in result.stdout:
    subprocess.run(f"git remote add origin {git_url}", shell=True, check=True)
    print("Remoto Git configurado.")
else:
    print("Remoto Git ya estaba configurado.")

# Configurar remoto DVC con S3
subprocess.run(f"dvc remote add origin {dvc_s3_url} --force", shell=True, check=True)
subprocess.run(f"dvc remote modify origin endpointurl {dvc_endpoint_url}", shell=True, check=True)
subprocess.run(f"dvc remote modify origin --local access_key_id {token}", shell=True, check=True)
subprocess.run(f"dvc remote modify origin --local secret_access_key {token}", shell=True, check=True)
subprocess.run("dvc remote default origin", shell=True, check=True)
print("Remoto DVC S3 configurado.")

# Commitear cambios de DVC solo si hay cambios
result = subprocess.run("git status --porcelain .dvc/config", shell=True, capture_output=True, text=True)
if result.stdout.strip():
    subprocess.run("git add .dvc/config", shell=True, check=True)
    subprocess.run('git commit -m "Configurar remoto S3 en DVC"', shell=True, check=True)
    print("Configuración DVC commiteada.")
else:
    print("No hay cambios en .dvc/config para commitear.")

# Subir a DAGsHub
subprocess.run("git push -u origin master", shell=True, check=True)
subprocess.run("dvc push", shell=True, check=True)
print("Proyecto subido a DAGsHub.")