

---

# 🚀✨ Intelligent Credit Scoring Pipeline - Deployment ✨🚀

¡Hola! 👋 Esta carpeta contiene todo lo necesario para desplegar mi modelo de scoring crediticio como una API usando **BentoML**. Después de entrenar y registrar el modelo, ahora puedo servirlo para hacer predicciones en tiempo real 📡💡.

---

## 🧰 Requisitos previos

Antes de ejecutar la API, asegúrate de tener lo siguiente:

- 🐍 **Python 3.8+**
- ⚙️ Entorno virtual `env_pipeline` activado
- 📦 Dependencias instaladas desde `requirements.txt`:

```bash
pip install -r ../requirements.txt
```

- 🧠 Modelo entrenado guardado en `models/model_1/rf_model.pkl` y registrado en BentoML ✅

---

## 📁 Archivos importantes

| Archivo                  | ¿Para qué lo uso?                            |
|--------------------------|----------------------------------------------|
| `bentofile.yaml`         | Configura el servicio con BentoML ⚙️         |
| `service.py`             | Define el endpoint `/predict` 🚀             |
| `save_model_to_bentoml.py` | Registra el modelo en BentoML 📥           |

---

## ▶️ Cómo levantar la API

1️⃣ Activar entorno virtual:

```bash
cd G:\MLOps Proyecto End_to_End\IntelligentCreditScoringPipeline\deployment
.\env_pipeline\Scripts\activate
```

2️⃣ Registrar el modelo (si hubo cambios):

```bash
python save_model_to_bentoml.py
```

3️⃣ Iniciar el servicio de la API:

```bash
bentoml serve service.py:CreditScoringService --reload
```

📍 La API estará disponible en: [http://127.0.0.1:3000](http://127.0.0.1:3000)

---

## 🧪 Cómo probar la API

### 🌐 Desde Swagger UI

1. Abre [http://127.0.0.1:3000/docs](http://127.0.0.1:3000/docs)
2. Busca el endpoint `/predict`
3. Verás un JSON con valores de prueba
4. Haz clic en **Execute** para obtener tu predicción 🎉

### 💻 Usando cURL

✅ Valores por defecto:

```bash
curl -X POST http://127.0.0.1:3000/predict
```

✍️ Enviar tus propios valores:

```bash
curl -X POST -H "Content-Type: application/json" ^
     -d "[{...mis datos...}]" ^
     http://127.0.0.1:3000/predict
```

---

## 🔄 ¿Qué devuelve la API?

```json
{"predictions": ["Good"]}
```

🔹 Los valores posibles: `"Good"`, `"Bad"` o `"Standard"`.

---

## 📌 Notas útiles

- Modelo guardado en: `C:\Users\<mi_usuario>\.bentoml\models\`
- Verifica la etiqueta del modelo: `credit_scoring_model_1:<última_tag>`

✅ Para empaquetar todo:

```bash
bentoml build
```

---

## 🧾 Ejemplo de entrada para el modelo

```json
{
  "input_data": {
    "Edad": 40,
    "Salario_Mensual": 16000,
    ...
    "credit_history_ratio": 0.167
  }
}
```

🔍 Representa a una persona con buen historial y comportamiento financiero. Ideal para pruebas reales.

---

## ❗ Qué hacer si algo falla

| Problema | Qué hacer |
|----------|-----------|
| ❌ No encuentra el modelo | Verifica con `bentoml models list` y ejecuta `save_model_to_bentoml.py` |
| ⚠️ Error en la API | Revisa los logs en consola o usa Swagger UI |

---

# 🐳 Crear Contenedor Docker y Subir a Docker Hub

## 1. Prerrequisitos

- Docker Desktop instalado y corriendo (`docker version`)
- Cuenta en Docker Hub
- Modelo registrado: `credit_scoring_model_1`

## 2. Preparar dependencias

Crear `requirements_clean.txt` con:

```txt
bentoml==1.4.10
pandas==2.2.3
pydantic==2.11.2
scikit-learn==1.6.1
python-dotenv==1.1.0
```

Instalarlas (opcional):

```bash
pip install -r ../requirements_clean.txt
```

## 3. Registrar el modelo

```bash
cd G:\MLOps Proyecto End_to_End\IntelligentCreditScoringPipeline\deployment
python save_model_to_bentoml.py
```

## 4. Verificar modelos

```bash
bentoml models list
```

## 5. Configurar `bentofile.yaml`

```yaml
service: "service.py:CreditScoringService"
include:
  - "*.py"
python:
  requirements_txt: "../requirements_clean.txt"
models:
  - credit_scoring_model_1:latest
```

## 6. Construir el Bento

```bash
cd G:\MLOps Proyecto End_to_End\IntelligentCreditScoringPipeline\deployment
set PYTHONUTF8=1
bentoml build
```

---

## 7. Crear la Imagen Docker

```bash
bentoml containerize credit_scoring_service_model_1:mrsbs4i4acecsaib
```

## 8. Verificar la Imagen

```bash
docker images
```

---

## 9. Probar Contenedor Localmente

```bash
docker run --rm -p 3000:3000 credit_scoring_service_model_1:mrsbs4i4acecsaib
```

📍 Prueba en [http://localhost:3000/docs](http://localhost:3000/docs)

Detener contenedor:

```bash
docker ps
docker stop <container_id>
```

---

## 10. Subir Imagen a Docker Hub

```bash
docker tag credit_scoring_service_model_1:mrsbs4i4acecsaib jorgedatascientist/credit_scoring_service_model_1:mrsbs4i4acecsaib
docker login
docker push jorgedatascientist/credit_scoring_service_model_1:mrsbs4i4acecsaib
```

✅ **La imagen del contenedor ha sido publicada exitosamente en Docker Hub** y puedes consultarla aquí:

🔗 [https://hub.docker.com/repository/docker/jorgedatascientist/credit_scoring_service_model_1/general](https://hub.docker.com/repository/docker/jorgedatascientist/credit_scoring_service_model_1/general)

---

## 11. (Opcional) Probar desde Docker Hub

```bash
docker pull jorgedatascientist/credit_scoring_service_model_1:mrsbs4i4acecsaib
docker run --rm -p 3000:3000 jorgedatascientist/credit_scoring_service_model_1:mrsbs4i4acecsaib
```

---

## 🛠️ Solución de Problemas

| Problema | Qué hacer |
|----------|-----------|
| ❌ Docker no responde | Inicia Docker Desktop y verifica con `docker version` |
| ❌ Modelo no en BentoML | Ejecuta `save_model_to_bentoml.py` |
| ❌ Error en `bentoml build` | Verifica `requirements_clean.txt` y `bentofile.yaml` |
| ❌ Error en `bentoml containerize` | Confirma que Docker esté activo |
| ❌ Contenedor no inicia | Usa `docker logs <container_id>` para más información |
| ❌ Error al subir | Asegúrate de estar autenticado en Docker Hub |

---