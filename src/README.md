

```markdown
# 🌟 MLOps_CreditScore Pipeline 🚀

¡Bienvenido al pipeline de Credit Scoring! Este proyecto predice puntajes crediticios usando un modelo `RandomForestClassifier`. Aquí te explicamos cómo probar el pipeline paso a paso. 🎉

---

## 🛠️ Requisitos

- Python 3.8+ 🐍  
- Dependencias instaladas:  
  ```bash
  pip install -r requirements.txt 📦
  ```

---

## 📋 Procedimiento para Probar el Pipeline

Sigue estos pasos para ejecutar el pipeline completo:

---

### 1. 📊 Preprocesar los Datos

Ejecuta el script de preprocesamiento para limpiar y transformar los datos de `data/raw/train.csv`.

```bash
python src/preprocess.py
```

**Resultado:**  
Se generan `X_train.csv`, `X_test.csv`, `y_train.csv`, y `y_test.csv` en `data/processed/`. ✅

---

### 2. 🧠 Entrenar el Modelo

Entrena el modelo `RandomForestClassifier` con la configuración especificada (por defecto, `model_1.yaml`).

```bash
python src/train.py
```

**Resultado:**  
El modelo se guarda en `models/model_1/rf_model.pkl` y los parámetros en `models/model_1/params.json`. 📈

**Nota:** Para usar `model_2.yaml`, ejecuta:

```bash
python src/train.py model=model_2
```

Esto guarda el modelo en `models/model_2/`.

---

### 3. 🔍 Evaluar el Modelo

Evalúa el modelo entrenado calculando métricas y generando visualizaciones.

```bash
python src/evaluate.py
```

**Resultado:**

- Métricas en `metrics/metrics_model_1.csv` 📊  
- Matriz de confusión en `graphics/confusion_matrix_model_1.png` 🖼️

**Nota:** Para evaluar `model_2`, usa:

```bash
python src/evaluate.py model=model_2
```

---

### 4. 🎯 Realizar Predicciones

Predice puntajes crediticios sobre nuevos datos en `data/external/new_data.csv`.

```bash
python src/predict.py
```

**Resultado:**  
Las predicciones se guardan en `data/processed/predictions_model_1.csv` 🚀

**Nota:** Para `model_2`, usa:

```bash
python src/predict.py model=model_2
```

---

## 💡 Consejos

- Asegúrate de que `data/raw/train.csv` exista antes de empezar 📂  
- Revisa los logs en la consola para confirmar que cada paso se ejecuta correctamente 👀  
- Cambia entre configuraciones (`model_1.yaml`, `model_2.yaml`) usando el parámetro `model` en los comandos ⚙️

---

¡Disfruta probando el pipeline! 🎈
