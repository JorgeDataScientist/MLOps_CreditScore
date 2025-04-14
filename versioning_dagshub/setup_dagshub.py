import subprocess
import yaml
from pathlib import Path

# Leer versioning_dagshub.yaml desde el directorio config
config_path = Path(__file__).parent.parent / "config" / "versioning_dagshub.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Extraer datos
username = config["dagshub"]["username"]
token = config["dagshub"]["token"]
git_url = config["dagshub"]["git_url"]
dvc_s3_url = config["dagshub"]["dvc_s3_url"]
dvc_endpoint_url = config["dagshub"]["dvc_endpoint_url"]

# Configurar remoto Git
subprocess.run(f"git remote add origin {git_url}", shell=True, check=True)
print("Remoto Git configurado.")

# Configurar remoto DVC con S3
subprocess.run(f"dvc remote add origin {dvc_s3_url}", shell=True, check=True)
subprocess.run(f"dvc remote modify origin endpointurl {dvc_endpoint_url}", shell=True, check=True)
subprocess.run(f"dvc remote modify origin --local access_key_id {token}", shell=True, check=True)
subprocess.run(f"dvc remote modify origin --local secret_access_key {token}", shell=True, check=True)
subprocess.run("dvc remote default origin", shell=True, check=True)
subprocess.run("dvc config core.autostage true", shell=True, check=True)
print("Remoto DVC S3 configurado con autostage.")

# Commitear cambios de DVC
subprocess.run("git add .dvc/config", shell=True, check=True)
subprocess.run('git commit -m "Configurar remoto S3 en DVC"', shell=True, check=True)
print("Configuración DVC commiteada.")

# Subir a DAGsHub
subprocess.run("git push -u origin master", shell=True, check=True)
subprocess.run("dvc push", shell=True, check=True)
print("Proyecto subido a DAGsHub.")