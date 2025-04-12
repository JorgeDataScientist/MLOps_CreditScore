# 🌟 README - Control de Versionado con Git y DVC (GitHub)

La carpeta `versioning_github/` contiene scripts Python para gestionar el versionado del proyecto **MLOps_CreditScore** en GitHub con Git y DVC de forma modular y eficiente. ¡Estos scripts te permiten crear, configurar y actualizar tu repositorio en GitHub sin complicaciones! 🚀

## 🛠️ ¿Para qué sirven estos archivos?
Los scripts te ayudan a:
- Crear un repositorio en GitHub. 📦
- Configurar Git localmente en la rama `main`. 🗃️
- Inicializar DVC con un remoto local para datos y modelos. 💾
- Conectar el proyecto a GitHub y subir cambios. 🔗
- Actualizar el repositorio con nuevos commits y datos. ⬆️

Cada script se ejecuta desde la raíz del proyecto (`G:\MLOps Proyecto End_to_End\IntelligentCreditScoringPipeline`) y usa un archivo `versioning_github.yaml` para configuraciones. Un archivo `history.txt` registra las versiones para GitHub. 📜

---

## 📋 Pasos para usar los scripts

### 1. Crear el archivo `versioning_github.yaml` ✨
Crea `config/versioning_github.yaml` con las credenciales y configuraciones. **Es confidencial** porque incluye el token de GitHub, por lo que está excluido vía `.gitignore`.

**Ejemplo**:
```yaml
github:
  username: "JorgeDataScientist"
  token: "your_github_personal_access_token"
  repo_name: "MLOps_CreditScore"
  git_url: "https://github.com/JorgeDataScientist/MLOps_CreditScore.git"
  description: "Proyecto MLOps para analizar y predecir puntajes crediticios"
git:
  email: "jorgeluisdatascientist@gmail.com"
dvc:
  directories:
    - "data"
    - "models"
```

Copia este ejemplo, ajusta el token, y guárdalo en `config/versioning_github.yaml`.

---

### 2. Crear el repositorio en GitHub 🏗️
Usa `create_repo_github.py` para crear el repositorio.

- **Ejecutar**:
  ```
  python versioning_github\create_repo_github.py
  ```
- **Función**: Crea un repositorio público en GitHub usando el token y nombre de `versioning_github.yaml`.
- **Frecuencia**: Solo una vez.

---

### 3. Configurar Git para GitHub 🗄️
Configura Git con `init_git_github.py`.

- **Ejecutar**:
  ```
  python versioning_github\init_git_github.py
  ```
- **Función**: Añade el remoto `github`, crea la rama `main`, configura usuario y email, y hace un commit inicial usando el `.gitignore` existente.
- **Frecuencia**: Solo una vez (requiere un repositorio Git inicializado, como el de DAGsHub).

---

### 4. Inicializar DVC para GitHub 📂
Prepara DVC con `init_dvc_github.py`.

- **Ejecutar**:
  ```
  python versioning_github\init_dvc_github.py
  ```
- **Función**: Inicializa DVC (si no está hecho), añade `data/` y `models/`, y commitea archivos `.dvc` en la rama `main`.
- **Frecuencia**: Solo una vez.

---

### 5. Conectar a GitHub 🔌
Conecta el proyecto con `setup_github.py`.

- **Ejecutar**:
  ```
  python versioning_github\setup_github.py
  ```
- **Función**: Configura un remoto DVC local (`dvc_storage_github/`), commitea la configuración, y sube el proyecto a GitHub (`github main`).
- **Frecuencia**: Solo una vez.

---

### 6. Subir cambios futuros ⬆️
Usa `push_changes_github.py` para actualizar el repositorio.

- **Ejecutar**:
  ```
  python versioning_github\push_changes_github.py
  ```
- **Función**: Pide un mensaje de commit, sube cambios a Git (`github main`) y DVC (remoto local), y registra en `versioning_github/history.txt`.
- **Frecuencia**: Cada vez que hay cambios.

---

## 📜 Registro en `history.txt` 🕰️
Cada ejecución de `push_changes_github.py` añade una línea a `versioning_github/history.txt`.

**Ejemplo**:
```
2025-04-11 10:15:30 - Commit: "Actualizar modelo" - Hash: def456 - Creador: Jorge Luis Garcia - Email: jorgeluisdatascientist@gmail.com - Username GitHub: JorgeDataScientist
```

---

## 📝 Notas finales
- Ejecuta los scripts desde la raíz del proyecto. 🌳
- Instala dependencias:
  ```
  pip install requests pyyaml dvc
  ```
- Usa `git checkout master` para trabajar con DAGsHub y `git checkout main` para GitHub. 🔄
- El remoto DVC local (`dvc_storage_github/`) asegura que los datos no se mezclen con DAGsHub.
- Si hay errores, revisa los mensajes o el `versioning_github.yaml`. 🔍

**Jorge Luis Garcia**  
jorgeluisdatascientist@gmail.com  
GitHub: JorgeDataScientist 🌟