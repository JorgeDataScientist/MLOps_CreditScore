# 🌟 README - Control de Versionado con Git y DVC (DAGsHub)

La carpeta `versioning_dagshub/` contiene scripts Python para gestionar el versionado del proyecto **IntelligentCreditScoringPipeline** con Git, DVC y DAGsHub. ¡Estos scripts te permiten crear, configurar y actualizar tu repositorio como todo un experto! 🚀

## 🛠️ ¿Para qué sirven estos archivos?
Los scripts te ayudan a:
- Crear un repositorio en DAGsHub. 📦
- Inicializar Git y DVC localmente. 🗃️
- Conectar el proyecto a DAGsHub con S3. 🔗
- Subir cambios (código, datos, modelos) al repositorio remoto. ⬆️

Cada script se ejecuta desde la raíz del proyecto (`G:\MLOps Proyecto End_to_End\IntelligentCreditScoringPipeline`) y usa `config/versioning_dagshub.yaml` para configuraciones. Un archivo `history.txt` registra las versiones. 📜

---

## 📋 Pasos para usar los scripts

### 1. Crear el archivo `versioning_dagshub.yaml` ✨
Crea `config/versioning_dagshub.yaml` con tus credenciales. **Es confidencial** porque incluye el token de DAGsHub, así que está excluido vía `.gitignore`.

**Ejemplo**:
```yaml
dagshub:
  username: "JorgeDataScientist"
  token: "your_dagshub_token"
  repo_name: "MLOps_CreditScore"
  git_url: "https://dagshub.com/JorgeDataScientist/MLOps_CreditScore.git"
  dvc_s3_url: "s3://dvc"
  dvc_endpoint_url: "https://dagshub.com/JorgeDataScientist/MLOps_CreditScore.s3"
  description: "Proyecto MLOps para analizar y predecir puntajes crediticios"
git:
  email: "jorgeluisdatascientist@gmail.com"
gitignore:
  - "env_pipeline/"
  - "mlruns/"
  - ".vscode/"
  - ".pytest_cache"
  - "__pycache__"
  - "*.log"
  - "*.pyc"
  - ".ipynb_checkpoints/"
  - "config/versioning_dagshub.yaml"
  - "data/"
  - "models/"
dvc:
  directories:
    - "data"
    - "models"
```

Copia este ejemplo, ajusta los valores, y guárdalo en `config/versioning_dagshub.yaml`.

---

### 2. Crear el repositorio en DAGsHub 🏗️
Usa `create_repo.py` para crear el repositorio remoto.

- **Ejecutar**:
  ```
  python versioning_dagshub\create_repo.py
  ```
- **Función**: Crea un repositorio público en DAGsHub (`JorgeDataScientist/MLOps_CreditScore`).
- **Frecuencia**: Solo una vez.

---

### 3. Inicializar Git 🗄️
Configura Git localmente con `init_git.py`.

- **Ejecutar**:
  ```
  python versioning_dagshub\init_git.py
  ```
- **Función**: Inicializa Git, configura usuario y email, genera `.gitignore`, y hace un commit inicial.
- **Frecuencia**: Solo una vez.

---

### 4. Inicializar DVC 📂
Prepara DVC con `init_dvc.py`.

- **Ejecutar**:
  ```
  python versioning_dagshub\init_dvc.py
  ```
- **Función**: Inicializa DVC, añade `data/` y `models/`, y commitea archivos `.dvc` en Git.
- **Frecuencia**: Solo una vez.

---

### 5. Conectar a DAGsHub 🔌
Conecta el proyecto con `setup_dagshub.py`.

- **Ejecutar**:
  ```
  python versioning_dagshub\setup_dagshub.py
  ```
- **Función**: Configura remotos Git y DVC (S3), activa `core.autostage`, commitea la configuración, y sube a DAGsHub.
- **Frecuencia**: Solo una vez.

---

### 6. Subir cambios futuros ⬆️
Usa `push_changes.py` para actualizar el repositorio.

- **Ejecutar**:
  ```
  python versioning_dagshub\push_changes.py
  ```
- **Función**: Pide un mensaje de commit, sube cambios a Git y DVC, y registra en `history.txt`.
- **Frecuencia**: Cada vez que hay cambios.

---

## 📜 Registro en `history.txt` 🕰️
Cada ejecución de `push_changes.py` añade una línea a `versioning_dagshub/history.txt`.

**Ejemplo**:
```
2025-04-11 10:15:30 - Commit: "Hola Mundo!" - Hash: 5ce1226 - Creador: Jorge Luis Garcia - Email: jorgeluisdatascientist@gmail.com - Username DAGsHub: JorgeDataScientist
```

---

## 📝 Notas finales
- Ejecuta los scripts desde la raíz del proyecto en el entorno `(env_pipeline)`. 🌳
- Instala dependencias:
  ```
  pip install dvc[s3]
  ```
- Revisa `config/versioning_dagshub.yaml` si hay errores. 🔍
- Los scripts están optimizados; evita modificarlos para mantener el flujo.

**Jorge Luis Garcia**  
jorgeluisdatascientist@gmail.com  
DAGsHub: JorgeDataScientist 🌟