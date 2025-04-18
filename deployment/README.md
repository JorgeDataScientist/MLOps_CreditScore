
# 🚀✨ Intelligent Credit Scoring Pipeline - Deployment ✨🚀

¡Hola! 👋 Esta carpeta contiene todo lo necesario para **desplegar mi modelo de scoring crediticio** como una API usando **BentoML**. Después de entrenar y registrar el modelo, ahora puedo servirlo para hacer predicciones en tiempo real 📡💡.

---

## 🧰 Lo que necesito antes de comenzar

Antes de ejecutar la API, me aseguro de tener todo listo:

- 🐍 **Python 3.8+**
- ⚙️ El entorno virtual `env_pipeline` activado.
- 📦 Las dependencias instaladas desde `requirements.txt`, ejecutando:

```bash
pip install -r ../requirements.txt
```

- 🧠 Mi modelo entrenado guardado en:  
  `models/model_1/rf_model.pkl`  
  (Y ya registrado en BentoML con su etiqueta correspondiente) ✅

---

## 📁 Archivos importantes de este directorio

| Archivo | ¿Para qué lo uso? |
|--------|-------------------|
| `bentofile.yaml` | Configura el servicio con BentoML ⚙️ |
| `service.py` | Define el endpoint `/predict` de mi API 🚀 |
| `save_model_to_bentoml.py` | Me permite registrar el modelo en BentoML 📥 |

---

## ▶️ Cómo levanto la API

### 1️⃣ Activo el entorno virtual

```bash
cd G:\MLOps Proyecto End_to_End\IntelligentCreditScoringPipeline\deployment
.\env_pipeline\Scripts\activate
```

---

### 2️⃣ Registro el modelo (si hubo cambios)

```bash
python save_model_to_bentoml.py
```

---

### 3️⃣ Inicio el servicio de la API

```bash
bentoml serve service.py:CreditScoringService --reload
```

Una vez hecho esto, la API queda disponible en:  
🌐 **http://127.0.0.1:3000**

---

## 🧪 Cómo pruebo la API

### 🌐 Desde Swagger UI

1. Abro el navegador en: [http://127.0.0.1:3000/docs](http://127.0.0.1:3000/docs)
2. Busco el endpoint `/predict`
3. Puedo ver un JSON con valores de prueba
4. Hago clic en **Execute** y obtengo mi predicción 🎉

---

### 💻 Usando cURL

✅ Si quiero probar con los valores por defecto:

```bash
curl -X POST http://127.0.0.1:3000/predict
```

✍️ O puedo enviar mis propios valores:

```bash
curl -X POST -H "Content-Type: application/json" ^
     -d "[{...mis datos...}]" ^
     http://127.0.0.1:3000/predict
```

---

## 🔄 ¿Qué devuelve la API?

Me devuelve una respuesta como esta:

```json
{"predictions": ["Good"]}
```

Los posibles valores que puedo recibir son `"Good"`, `"Bad"` o `"Standard"` según cómo fue entrenado el modelo.

---

## 📌 Algunas notas útiles

- El modelo está guardado en:  
  `C:\Users\<mi_usuario>\.bentoml\models\`

- Me aseguro de que la etiqueta del modelo sea algo como:  
  `credit_scoring_model_1:<última_tag>`

- Si quiero empaquetar todo para producción o Docker, simplemente corro:

```bash
bentoml build
```

---

## 🧾 Ejemplo de entrada para el modelo

Aquí te muestro un ejemplo completo del **JSON que puedo enviar al endpoint `/predict`** para obtener una predicción:

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

🔍 Este JSON representa a una persona de 40 años, con buen historial crediticio y comportamiento financiero saludable. ¡Perfecto para probar la precisión del modelo en situaciones reales!

---

## ❗ Qué hago si algo falla

| Problema | Qué hago |
|----------|----------|
| ❌ No encuentra el modelo | Verifico con `bentoml models list` y vuelvo a ejecutar `save_model_to_bentoml.py` |
| ⚠️ Error en la API | Reviso los logs en consola o pruebo desde Swagger UI |

---

🧡 ¡Ya puedo servir predicciones de manera elegante, escalable y rápida! 🚀✨