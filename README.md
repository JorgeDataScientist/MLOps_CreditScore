

---

# 🌟 Intelligent Credit Scoring Pipeline 🚀

¡Hola y bienvenido al proyecto **Intelligent Credit Scoring Pipeline**! 🎉  
Soy **Jorge**, un apasionado de la ciencia de datos, y este proyecto es mi aventura para construir un sistema automatizado que predice puntajes crediticios de forma eficiente y confiable.

---

## 🎯 Objetivo del Proyecto

Crear un **pipeline de MLOps** que clasifique a las personas en tres categorías de puntaje crediticio: `Poor`, `Standard` y `Good`. Este sistema busca ayudar a instituciones financieras a evaluar riesgos crediticios rápidamente usando indicadores como ingresos, deudas y más.

### 🧩 Metas del pipeline:

- ⚙️ **Automático**: Desde la carga de datos hasta la evaluación del modelo.
- 📈 **Escalable**: Fácil de adaptar a nuevos datos o modelos.
- 📊 **Explicativo**: Genera informes y visualizaciones comprensibles.
- 💪 **Robusto**: Manejo de errores y resultados confiables.

---

## 🛠️ Arquitectura del Proyecto

### 📁 Estructura de Carpetas

```bash
data/             # Datos crudos y procesados
models/           # Modelos entrenados
graphics/         # Curvas ROC, matrices de confusión, etc.
metrics/          # CSVs de métricas y reportes
informe/          # Reportes EDA en HTML
src/              # Código fuente
config/           # Configuraciones en YAML
```

---

## 🔍 Principales Scripts

| Script            | Función |
|-------------------|---------|
| `preprocess.py`   | Limpieza, codificación y split de los datos |
| `train.py`        | Entrenamiento del modelo con optimización |
| `evaluate.py`     | Evaluación con métricas, gráficas e informes |
| `run_pipeline.py` | Orquesta la ejecución completa del pipeline |

---

## 🌐 Integraciones

- **MLflow**: Seguimiento de métricas, parámetros y modelos.
- **DAGsHub**: Colaboración y visualización remota de experimentos.

> 📍 MLflow Tracking: [dagshub.com/JorgeDataScientist/MLOps_CreditScore.mlflow](https://dagshub.com/JorgeDataScientist/MLOps_CreditScore.mlflow)

---

## 🛠️ ¿Cómo lo construí?

### 📝 Planeación
- Objetivo: Predecir puntajes crediticios con un modelo interpretable.
- Modelo elegido: `RandomForestClassifier`.
- Enfoque: Automatización completa con MLOps.

### 🗃️ Preparación de Datos
- Variables: ingresos mensuales, deudas, puntaje objetivo.
- Script: `preprocess.py`.

### ⚙️ Desarrollo del Pipeline
- Entrenamiento con `RandomizedSearchCV`.
- Configuración dinámica con **Hydra**.
- Validación con métricas y **EDA** automático (ydata-profiling).

### 📊 Visualizaciones y Reportes
- Curvas ROC, matrices de confusión y barras de métricas.
- Reportes detallados (`classification_report`, EDA en HTML).

---

## 🎮 Automatización

- Script `run_pipeline.py` ejecuta todo el flujo secuencial.
- Manejo de errores y logs con `logging`.

---

## 😓 Retos Superados

- 🛣️ Rutas erróneas en evaluaciones.
- 📐 Formatos incompatibles entre `y_test` y métricas.
- 🔑 Configuración compleja de MLflow con DAGsHub.
- ⏳ Entrenamiento lento → uso de `n_jobs=-1` y `n_iter` reducido.

---

## 🏆 Logros

- 📈 **Precisión**: accuracy = `0.832`, f1_macro = `0.819`, ROC AUC = `0.935`
- 🤖 **Pipeline automático** con un solo comando.
- 🧾 **Informes detallados** para entender el rendimiento.
- ☁️ **Integración total con MLflow y DAGsHub**.

---

## 🚀 ¿Cómo usar este proyecto?

### 1. Clona el repositorio

```bash
git clone https://github.com/JorgeDataScientist/MLOps_CreditScore.git
cd MLOps_CreditScore
```

### 2. Instala dependencias

```bash
python -m venv env_pipeline
.\env_pipeline\Scripts\activate
pip install -r requirements.txt
```

### 3. Configura DAGsHub (opcional)

Edita `config/versioning_dagshub.yaml` con tu usuario y token.

### 4. Ejecuta el pipeline

```bash
python src/run_pipeline.py
```

### 5. Explora los resultados

- 📊 Gráficas: `graphics/model_1/`
- 📈 Métricas: `metrics/model_1/`
- 📄 EDA: `informe/model_1/informe.html`
- 🌐 MLflow: Enlace a tu DAGsHub

---

## 🎨 ¿Qué hace especial a este pipeline?

- 🔁 Automatización con MLOps
- 📊 Visualizaciones claras e informativas
- ☁️ Herramientas modernas: MLflow, DAGsHub, Hydra
- 🔧 Configuración flexible para múltiples modelos

---

## 🌱 Próximos pasos

- 🆚 Agregar modelos como XGBoost y LightGBM
- 📉 Nuevas métricas como Precision-Recall curves
- 🖥️ Dashboard interactivo con Streamlit
- 🐳 Despliegue con Docker o Kubernetes

---

## 🙌 Agradecimientos

Gracias a la comunidad de código abierto por herramientas como **scikit-learn**, **Hydra**, **MLflow** y **ydata-profiling**.  
¡Y gracias a ti por llegar hasta aquí!  
¿Tienes ideas o quieres colaborar?  
📬 ¡Contáctame en GitHub!

---





Nos falta:

Punto 9: Crear visualización con Streamlit (sin app.py para visualización).
Punto 11: Automatizar con GitHub Actions (probablemente falta, ligado a las pruebas).