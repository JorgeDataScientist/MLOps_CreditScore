
# 🌟 Intelligent Credit Scoring Pipeline 🚀

¡Hola y bienvenido al proyecto Intelligent Credit Scoring Pipeline! 🎉 Soy Jorge, un apasionado de la ciencia de datos, y este proyecto es mi aventura para construir un sistema automatizado que predice puntajes crediticios de forma eficiente y confiable. 

## 🎯 Objetivo del Proyecto

Crear un pipeline de MLOps que clasifique a las personas en tres categorías de puntaje crediticio: Poor, Standard y Good. Este sistema busca ayudar a instituciones financieras a evaluar riesgos crediticios rápidamente usando indicadores como ingresos, deudas y más.

## 🧩 Metas del pipeline

- ⚙️ **Automático**: Desde la carga de datos hasta la evaluación del modelo.
- 📈 **Escalable**: Fácil de adaptar a nuevos datos o modelos.
- 📊 **Explicativo**: Genera informes y visualizaciones comprensibles.
- 💪 **Robusto**: Manejo de errores y resultados confiables.

## 🛠️ Arquitectura del Proyecto

### 📁 Estructura de Carpetas
- `data/`  # Datos crudos y procesados
- `models/`  # Modelos entrenados
- `graphics/`  # Curvas ROC, matrices de confusión, etc.
- `metrics/`  # CSVs de métricas y reportes
- `informe/`  # Reportes EDA en HTML
- `src/`  # Código fuente
- `config/`  # Configuraciones en YAML
- `deployment/`  # Archivos para desplegar la API con BentoML y Docker

### 🔍 Principales Scripts

| Script | Función |
|--------|---------|
| `preprocess.py` | Limpieza, codificación y split de los datos |
| `train.py` | Entrenamiento del modelo con optimización |
| `evaluate.py` | Evaluación con métricas, gráficas e informes |
| `run_pipeline.py` | Orquesta la ejecución completa del pipeline |
| `deployment/save_model_to_bentoml.py` | Registra el modelo en BentoML para despliegue |
| `deployment/service.py` | Define la API para predicciones |

### 🌐 Integraciones

- **MLflow**: Seguimiento de métricas, parámetros y modelos.
- **DAGsHub**: Colaboración y visualización remota de experimentos.

**MLflow Tracking**: [dagshub.com/JorgeDataScientist/MLOps_CreditScore.mlflow](dagshub.com/JorgeDataScientist/MLOps_CreditScore.mlflow)

## 🛠️ ¿Cómo lo construí?

### 📝 Planeación
- **Objetivo**: Predecir puntajes crediticios con un modelo interpretable.
- **Modelo elegido**: RandomForestClassifier.
- **Enfoque**: Automatización completa con MLOps.

### 🗃️ Preparación de Datos
- **Variables**: ingresos mensuales, deudas, puntaje objetivo.
- **Script**: `preprocess.py`.

### ⚙️ Desarrollo del Pipeline
- Entrenamiento con RandomizedSearchCV.
- Configuración dinámica con Hydra.
- Validación con métricas y EDA automático (ydata-profiling).

### 📊 Visualizaciones y Reportes
- Curvas ROC, matrices de confusión y barras de métricas.
- Reportes detallados (classification_report, EDA en HTML).

### 🎮 Automatización
- Script `run_pipeline.py` ejecuta todo el flujo secuencial.
- Manejo de errores y logs con logging.

## 😓 Retos Superados

- 🛣️ Rutas erróneas en evaluaciones.
- 📐 Formatos incompatibles entre `y_test` y métricas.
- 🔑 Configuración compleja de MLflow con DAGsHub.
- ⏳ Entrenamiento lento → uso de `n_jobs=-1` y `n_iter` reducido.

## 🏆 Logros

- 📈 **Precisión**: accuracy = 0.832, f1_macro = 0.819, ROC AUC = 0.935
- 🤖 **Pipeline automático** con un solo comando.
- 🧾 **Informes detallados** para entender el rendimiento.
- ☁️ **Integración total** con MLflow y DAGsHub.

## 🚀 ¿Cómo usar este proyecto?

1. **Clona el repositorio**
   ```bash
   git clone https://github.com/JorgeDataScientist/MLOps_CreditScore.git
   cd MLOps_CreditScore
   ```

2. **Instala dependencias**
   ```bash
   python -m venv env_pipeline
   .\env_pipeline\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Configura DAGsHub (opcional)**
   - Edita `config/versioning_dagshub.yaml` con tu usuario y token.

4. **Ejecuta el pipeline**
   ```bash
   python src/run_pipeline.py
   ```

5. **Explora los resultados**
   - 📊 **Gráficas**: `graphics/model_1/`
   - 📈 **Métricas**: `metrics/model_1/`
   - 📄 **EDA**: `informe/model_1/informe.html`
   - 🌐 **MLflow**: Enlace a tu DAGsHub

## 🎨 ¿Qué hace especial a este pipeline?

- 🔁 **Automatización con MLOps**
- 📊 **Visualizaciones claras e informativas**
- ☁️ **Herramientas modernas**: MLflow, DAGsHub, Hydra
- 🔧 **Configuración flexible** para múltiples modelos


# 🌱 Próximos pasos

- 🆚 Agregar modelos como XGBoost y LightGBM
- 📉 Nuevas métricas como Precision-Recall curves
- 🖥️ Dashboard interactivo con Streamlit
- 🐳 Despliegue con Docker o Kubernetes

## 🙌 Agradecimientos

Gracias a la comunidad de código abierto por herramientas como scikit-learn, Hydra, MLflow y ydata-profiling. ¡Y gracias a ti por llegar hasta aquí! ¿Tienes ideas o quieres colaborar? 📬 ¡Contáctame en GitHub!

# 🐳 Desplegar la API con BentoML y Docker

Esta sección detalla cómo desplegar el modelo de scoring crediticio como una API usando BentoML, empaquetarlo en un contenedor Docker, y subirlo a Docker Hub para compartirlo o desplegarlo en otros entornos. También incluye cómo registrar el modelo, listar los modelos guardados, y probar la API localmente.

## 1. Prerrequisitos

Asegúrate de tener lo siguiente antes de proceder:

- **Python 3.8+**: Instalado y configurado en el entorno virtual `env_pipeline`.
- **Docker Desktop**: Instalado y ejecutándose en Windows. Verifica con:
  ```bash
  docker version
  ```

### 7. Levantar la API Localmente

Antes de construir el contenedor Docker, es importante probar la API localmente para asegurarse de que todo funciona correctamente. Aquí están los pasos para hacerlo:

1. Navega al directorio de despliegue:
    ```bash
    cd G:\MLOps Proyecto End_to_End\IntelligentCreditScoringPipeline\deployment
    ```

2. Levanta el servicio localmente con BentoML:
    ```bash
    bentoml serve service.py:CreditScoringService --reload
    ```
   - **bentoml serve**: Esto inicia un servidor de desarrollo usando FastAPI.
   - **--reload**: Activa la recarga automática si haces cambios en el archivo `service.py`.

3. Una vez que el servidor esté en funcionamiento, la API estará disponible en:  
    [http://127.0.0.1:3000](http://127.0.0.1:3000).

### Probar la API

Accede a la documentación interactiva de la API a través de FastAPI en el navegador:

[http://127.0.0.1:3000/docs](http://127.0.0.1:3000/docs)

Para probar el endpoint `/predict`, puedes enviar una solicitud POST con el siguiente JSON:

```json
{
  "input_data": {
    "Edad": 40,
    "Salario_Mensual": 16000,
    "Num_Tarjetas_Credito": 2,
    "Tasa_Interes": 3,
    "Retraso_Pago": 1,
    "Num_Pagos_Retrasados": 2,
    "Cambio_Limite_Credito": 0,
    "Num_Consultas_Credito": 2,
    "Deuda_Pendiente": 1000,
    "Edad_Historial_Credito": 5,
    "Total_Cuota_Mensual": 200,
    "Inversion_Mensual": 100,
    "Saldo_Mensual": 1000,
    "Comportamiento_de_Pago_High_spent_Large_value_payments": 0,
    "Comportamiento_de_Pago_High_spent_Medium_value_payments": 0,
    "Comportamiento_de_Pago_High_spent_Small_value_payments": 0,
    "Comportamiento_de_Pago_Low_spent_Large_value_payments": 0,
    "Comportamiento_de_Pago_Low_spent_Medium_value_payments": 1,
    "Comportamiento_de_Pago_Low_spent_Small_value_payments": 0,
    "Mezcla_Crediticia_Bad": 0,
    "Mezcla_Crediticia_Good": 1,
    "Mezcla_Crediticia_Standard": 0,
    "Pago_Minimo_No": 1,
    "Pago_Minimo_Yes": 0,
    "Ocupacion_Architect": 0,
    "Ocupacion_Developer": 1,
    "Ocupacion_Doctor": 0,
    "Ocupacion_Engineer": 0,
    "Ocupacion_Entrepreneur": 0,
    "Ocupacion_Journalist": 0,
    "Ocupacion_Lawyer": 0,
    "Ocupacion_Manager": 0,
    "Ocupacion_Mechanic": 0,
    "Ocupacion_Media_Manager": 0,
    "Ocupacion_Musician": 0,
    "Ocupacion_Scientist": 0,
    "Ocupacion_Teacher": 0,
    "Ocupacion_Writer": 0,
    "debt_to_income": 0.833,
    "payment_to_income": 0.033,
    "credit_history_ratio": 0.167
  }
}
```

O usa cURL para probarlo:

```bash
curl -X POST -H "Content-Type: application/json" -d "{\"input_data\": {\"Edad\": 40, ...}}" http://127.0.0.1:3000/predict
```

**Salida esperada**: Un JSON con la predicción, como:
```json
{"predictions": ["Good"]}
```

### Detener el Servidor

Para detener el servidor, presiona `Ctrl+C` en la terminal.

#### En caso de que algo falle:

- Revisa los logs de la terminal para identificar posibles errores, como un modelo no encontrado.
- Asegúrate de que el modelo esté registrado correctamente con `bentoml models list`.

---

### 8. Construir el Bento

Una vez que hayas probado la API localmente, es hora de construir el Bento que empaquetará el modelo, el código y las dependencias. Aquí están los pasos:

1. Entra al directorio de despliegue:
    ```bash
    cd G:\MLOps Proyecto End_to_End\IntelligentCreditScoringPipeline\deployment
    ```

2. Configura la codificación UTF-8 en Windows:
    ```bash
    set PYTHONUTF8=1
    ```

3. Ejecuta el comando para construir el Bento:
    ```bash
    bentoml build
    ```

**Salida esperada**: Un mensaje que confirma que el Bento se construyó correctamente, como:

```bash
Successfully built Bento(tag="credit_scoring_service_model_1:mrsbs4i4acecsaib").
```

### Detalles

- **set PYTHONUTF8=1**: Esto asegura que no haya problemas con caracteres especiales en Windows.
- **bentoml build**: Este comando empaqueta el modelo y el servicio en un contenedor Bento, listo para ser convertido en una imagen Docker.

Si falla, revisa los logs para identificar el error, como un modelo no encontrado o un conflicto de dependencias. Asegúrate de que el modelo esté registrado correctamente.

---

### 9. Crear la Imagen Docker

Ahora, crea una imagen Docker a partir del Bento generado. Ejecuta:

```bash
bentoml containerize credit_scoring_service_model_1:mrsbs4i4acecsaib
```

**Salida esperada**:
```bash
Successfully built Bento container for "credit_scoring_service_model_1:latest" with tag(s) "credit_scoring_service_model_1:mrsbs4i4acecsaib".
```

### 10. Verificar la Imagen Docker

Para verificar que la imagen Docker se ha creado correctamente, ejecuta:

```bash
docker images
```

**Salida esperada**:
```bash
REPOSITORY                     TAG                IMAGE ID       CREATED        SIZE
credit_scoring_service_model_1 mrsbs4i4acecsaib   abc123def456   5 minutes ago  1.2GB
```

Si la imagen no aparece, asegúrate de que Docker esté corriendo y vuelve a ejecutar `bentoml containerize`.

---

### 11. Probar el Contenedor Localmente

Ahora que tienes la imagen Docker, puedes ejecutar el contenedor y verificar que la API funcione correctamente:

```bash
docker run --rm -p 3000:3000 credit_scoring_service_model_1:mrsbs4i4acecsaib
```

**Salida esperada**:

```bash
2025-04-18TXX:XX:XX+0000 [INFO] [cli] Starting production HTTP BentoServer from "/home/bentoml/bento" listening on http://localhost:3000 (Press CTRL+C to quit)
2025-04-18TXX:XX:XX+0000 [INFO] [:1] Service credit_scoring_service_model_1 initialized
```

Luego, prueba la API en [http://localhost:3000/docs](http://localhost:3000/docs) o con cURL.

---

### 12. Subir la Imagen a Docker Hub

Una vez que todo esté funcionando, puedes subir la imagen a Docker Hub para compartirla o desplegarla en otros entornos. Aquí están los pasos:

1. Etiqueta la imagen para tu repositorio en Docker Hub:
    ```bash
    docker tag credit_scoring_service_model_1:mrsbs4i4acecsaib <tu_usuario_dockerhub>/credit_scoring_service_model_1:mrsbs4i4acecsaib
    ```

2. Inicia sesión en Docker Hub:
    ```bash
    docker login
    ```

3. Sube la imagen:
    ```bash
    docker push <tu_usuario_dockerhub>/credit_scoring_service_model_1:mrsbs4i4acecsaib
    ```

**Salida esperada**:
```bash
The push refers to repository [docker.io/<tu_usuario_dockerhub>/credit_scoring_service_model_1]
abc123def456: Pushed
mrsbs4i4acecsaib: digest: sha256:xyz789... size: 1234
```

Verifica en Docker Hub que la imagen se haya subido correctamente.


---

### 13. (Opcional) Probar la Imagen desde Docker Hub

Si deseas verificar que la imagen funcione correctamente, puedes descargarla y ejecutarla en otro entorno:

1. Descarga la imagen:
    ```bash
    docker pull <tu_usuario_dockerhub>/credit_scoring_service_model_1:mrsbs4i4acecsaib
    ```

2. Ejecuta el contenedor:
    ```bash
    docker run --rm -p 3000:3000 <tu_usuario_dockerhub>/credit_scoring_service_model_1:mrsbs4i4acecsaib
    ```

¡Y listo! Ahora tienes la API en producción y accesible desde cualquier lugar.


---

### Docker Hub Repository

Puedes acceder a mi repositorio del contenedor de tu servicio de scoring crediticio en Docker Hub desde el siguiente enlace:

[**credit_scoring_service_model_1 en Docker Hub**](https://hub.docker.com/repository/docker/jorgedatascientist/credit_scoring_service_model_1/general)


---

### 14. Solución de Problemas para el Despliegue

| **Problema**                  | **Qué Hago**                                                                                                                                      |
|-------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------|
| ❌ **Docker Desktop no responde** | Inicia Docker Desktop y verifica con `docker version`.                                                                                           |
| ❌ **Modelo no encontrado**     | Ejecuta `bentoml models list` para confirmar que `credit_scoring_model_1` existe. Si no, re-ejecuta `save_model_to_bentoml.py`.                    |
| ❌ **Error en bentoml build**   | Verifica `requirements_clean.txt` por conflictos. Asegúrate de que `bentofile.yaml` incluya el modelo correcto.                                  |
| ❌ **Error en bentoml containerize** | Confirma que Docker Desktop está corriendo y que el Bento se construyó (`bentoml list`).                                                           |
| ❌ **Contenedor no inicia**     | Revisa los logs (`docker logs <container_id>`) para errores como modelo faltante o puertos ocupados.                                               |
| ❌ **Error al subir a Docker Hub** | Asegúrate de estar autenticado (`docker login`) y que el nombre del repositorio sea correcto.                                                      |
| ❌ **Terminal bloqueada por contenedor** | Abre una nueva terminal o detén el contenedor con `docker stop <container_id>`.                                                                   |
| ❌ **API no responde**          | Verifica los logs del contenedor o del servidor (`bentoml serve`). Prueba desde `http://localhost:3000/docs`.                                      |

---

### 15. Notas Adicionales para el Despliegue

- **Ubicaciones:**
  - **Bentos:** `C:\Users\<mi_usuario>\.bentoml\bentos\credit_scoring_service_model_1\mrsbs4i4acecsaib\`
  - **Modelos:** `C:\Users\<mi_usuario>\.bentoml\models\credit_scoring_model_1\`

- **Tags Dinámicos:**
  - Reemplaza `mrsbs4i4acecsaib` con el tag generado por `bentoml build`.
  - Reemplaza `pkfagxy35cdvwaib` con el tag del modelo de `bentoml models list`.

- **API en Producción:**
  - Usa **FastAPI** y **BentoML**, accesible en `http://localhost:3000` (o `http://127.0.0.1:3000` para pruebas locales).
  
- **Escalabilidad:**
  - La imagen Docker puede desplegarse en plataformas como **AWS ECS**, **GCP Cloud Run**, **Azure Container Instances**, o **Kubernetes**.
  
- **Siguientes Pasos:**
  - Considera configurar **CI/CD con GitHub Actions** para automatizar builds y despliegues, o integrar un orquestador como **Kubernetes**.

---

🧡 ¡Con este pipeline, puedo entrenar, evaluar y desplegar un modelo de scoring crediticio de manera automatizada, escalable y profesional! 🚀✨

---

Con esto completamos la guía de despliegue, ¡estás listo para poner en marcha tu modelo y escalarlo!








Nos falta:

Punto 9: Crear visualización con Streamlit (sin app.py para visualización).
Punto 11: Automatizar con GitHub Actions (probablemente falta, ligado a las pruebas).