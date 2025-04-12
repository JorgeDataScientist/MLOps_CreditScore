import subprocess
import yaml
from pathlib import Path

# Leer versioning_dagshub.yaml desde el directorio config en la raíz
config_path = Path(__file__).parent.parent / "config" / "versioning_dagshub.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Extraer datos
username = config["dagshub"]["username"]
email = config["git"]["email"]
gitignore_items = config["gitignore"]

# Verificar si Git ya está inicializado
if not (Path(__file__).parent.parent / ".git").exists():
    subprocess.run("git init", shell=True, check=True)
    print("Git inicializado.")
else:
    print("Git ya estaba inicializado.")

# Configurar usuario
subprocess.run(f'git config user.email "{email}"', shell=True, check=True)
subprocess.run(f'git config user.name "{username}"', shell=True, check=True)
print("Usuario configurado.")

# Crear .gitignore
with open(Path(__file__).parent.parent / ".gitignore", "w") as f:
    f.write("\n".join(gitignore_items))
print(".gitignore creado.")

# Añadir y commitear
subprocess.run("git add .", shell=True, check=True)
subprocess.run('git commit -m "Initial commit con proyecto"', shell=True, check=True)
print("Commit inicial realizado.")