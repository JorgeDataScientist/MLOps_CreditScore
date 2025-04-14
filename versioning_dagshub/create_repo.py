import requests
import yaml
from pathlib import Path

# Leer versioning_dagshub.yaml desde el directorio config en la raíz
config_path = Path(__file__).parent.parent / "config" / "versioning_dagshub.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Extraer datos
username = config["dagshub"]["username"]
token = config["dagshub"]["token"]
repo_name = config["dagshub"]["repo_name"]
description = config["dagshub"]["description"]

# API de DAGsHub para crear repositorio
url = "https://dagshub.com/api/v1/user/repos"
headers = {"Authorization": f"Bearer {token}"}
data = {
    "name": repo_name,
    "description": description,
    "private": False
}

# Hacer la solicitud
response = requests.post(url, headers=headers, json=data)

# Verificar resultado
if response.status_code == 201:
    print(f"Repositorio {username}/{repo_name} creado exitosamente!")
else:
    print(f"Error al crear el repositorio: {response.status_code} - {response.text}")