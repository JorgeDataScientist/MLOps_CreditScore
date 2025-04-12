import requests
import yaml
from pathlib import Path

# Leer versioning_github.yaml desde el directorio config en la raíz
config_path = Path(__file__).parent.parent / "config" / "versioning_github.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Extraer datos
username = config["github"]["username"]
token = config["github"]["token"]
repo_name = config["github"]["repo_name"]
description = config["github"]["description"]

# API de GitHub para crear repositorio
url = "https://api.github.com/user/repos"
headers = {"Authorization": f"token {token}"}
data = {
    "name": repo_name,
    "description": description,
    "private": False  # Cambia a True si quieres privado
}

# Hacer la solicitud
response = requests.post(url, headers=headers, json=data)

# Verificar resultado
if response.status_code == 201:
    print(f"Repositorio {username}/{repo_name} creado exitosamente en GitHub!")
else:
    print(f"Error al crear el repositorio: {response.status_code} - {response.text}")