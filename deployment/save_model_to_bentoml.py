import os
import bentoml
import joblib
from pathlib import Path

# Obtiene el nombre del modelo desde la variable de entorno o usa el predeterminado
model_name = os.getenv("MODEL_NAME", "model_1")

# Construye la ruta al modelo
model_path = Path(f"../models/{model_name}/rf_model.pkl")

# Verifica que el archivo exista
if not model_path.exists():
    raise FileNotFoundError(f"Modelo no encontrado en {model_path}")

# Carga el modelo
model = joblib.load(model_path)

# Guarda el modelo en BentoML
bentoml.sklearn.save_model(f"credit_scoring_{model_name}", model)
print(f"Modelo {model_name} registrado en BentoML como credit_scoring_{model_name}")