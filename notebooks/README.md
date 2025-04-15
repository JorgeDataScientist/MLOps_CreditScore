

---

# 📊 Proyecto de Clasificación de Puntaje Crediticio

¡Hola! Este es mi informe sobre el proceso completo que seguí para analizar, limpiar, mejorar y modelar un dataset de puntajes crediticios. Desde el análisis exploratorio (EDA) hasta la selección de modelos, te cuento cada paso que di para predecir **`Puntaje_Credito_Num`** con la mayor precisión posible. Vamos allá.

---

## 🌟 Objetivo
Mi meta fue construir un modelo que prediga **`Puntaje_Credito_Num`** (0: Poor, 1: Standard, 2: Good) usando características financieras y demográficas, alcanzando una precisión cercana al 89-90%. Trabajé con RandomForestClassifier y XGBoost, ajustando datos y parámetros en el camino.

---

## 🛠️ Proceso Paso a Paso

Aquí está la lista detallada de todo lo que hice, desde el EDA hasta el modelado final:

### 1. 📈 Análisis Exploratorio de Datos (EDA)
- **Cargué el Dataset**: Importé mi dataset con pandas para verlo en acción.
- **Exploré Columnas**: Identifiqué variables numéricas (como `Salario_Mensual`, `Deuda_Pendiente`) y categóricas (como `Ocupacion`, `Comportamiento_Pago`).
- **Detecté Nulos**: Revisé valores faltantes y decidí cómo manejarlos (relleno o eliminación según el caso).
- **Distribuciones**: Grafiqué histogramas y boxplots para entender la dispersión y detectar outliers.
- **Correlaciones**: Usé un mapa de calor para ver relaciones entre variables numéricas.

### 2. 🧹 Limpieza de Datos
- **Eliminé Duplicados**: Me aseguré de que no hubiera filas repetidas.
- **Manejé Nulos**: Rellené valores faltantes en columnas numéricas con la mediana (ej. `Salario_Mensual`) y en categóricas con la moda (ej. `Ocupacion`).
- **Outliers**: Apliqué recorte (clipping) a valores extremos en columnas como `Deuda_Pendiente` usando percentiles 1% y 99%.

### 3. 🔧 Mejoramiento de Datos (Feature Engineering)
- **Creé Nuevas Características**:
  - `credit_history_ratio`: Relación entre historial crediticio y edad.
  - `credit_usage_to_limit`: Uso del crédito respecto al límite.
  - `debt_to_income`: Deuda respecto al ingreso.
  - `payment_to_income`: Pagos respecto al ingreso.
  - `delay_ratio`: Proporción de retrasos en pagos.
- **Codifiqué Variables**:
  - **OneHotEncoder**: Transformé `Ocupacion` (ej. Doctor, Engineer) y `Comportamiento_Pago` (Bajo Impacto, Intermedio, Responsable) en columnas binarias.
  - **Codificación Manual**: Convertí `Mezcla_Crediticia` y `Pago_Minimo` en valores numéricos (ej. `Mezcla_Crediticia_Cod`, `Pago_Minimo_Cod`).

### 4. 🗑️ Eliminación de Columnas
- **Columnas Redundantes**: Eliminé columnas originales tras crear nuevas características (ej. componentes de `credit_usage_to_limit`).
- **Baja Importancia**: Descarté variables con poca relevancia inicial (basado en EDA), como algunas ocupaciones poco frecuentes si no aportaban.

### 5. 📏 Escalado de Columnas Numéricas
- **StandardScaler**: Escalo columnas numéricas (ej. `Salario_Mensual`, `Deuda_Pendiente`, `Tasa_Interes`) para que tengan media 0 y desviación 1, mejorando el rendimiento del modelo.

### 6. 🔍 División y Optimización
- **Train-Test Split**: Dividí el dataset en 80% entrenamiento y 20% prueba con `stratify=y` para mantener las proporciones de `Puntaje_Credito_Num`.
- **RandomForestClassifier**: Probé múltiples configuraciones con `RandomizedSearchCV`, ajustando `n_estimators`, `max_depth`, `min_samples_split`, etc.
- **XGBoost**: Exploré este modelo para buscar mayor precisión, pero no cumplió expectativas.

### 7. 📊 Evaluación y Selección
- **Métricas Iniciales**: Usé `accuracy`, `f1_macro`, y `f1_per_class` para evaluar.
- **Métricas Finales**: Añadí `precision_macro`, `recall_macro`, `balanced_accuracy`, `roc_auc_ovr`, y `confusion_matrix` para un análisis completo.
- **Resultados**: RandomForest alcanzó un máximo de 83.1%, mientras XGBoost se quedó en 81.96%.

---

## 🎯 Selección de los Dos Mejores Modelos RandomForestClassifier

Escogí dos configuraciones de RandomForestClassifier porque dieron los mejores resultados tras muchas iteraciones:

1. **Modelo 1 (83.01%)**:
   - `n_estimators: 600`, `max_depth: 30`, `min_samples_split: 5`, `class_weight: 'balanced_subsample'`.
   - F1 Macro: 0.823, con buen balance entre clases.
2. **Modelo 2 (83.1%)**:
   - `n_estimators: 800`, `max_depth: 30`, `min_samples_split: 5`, `class_weight: 'balanced_subsample'`.
   - F1 Macro: 0.824, mi mejor marca.

Los seleccioné por su alta precisión, balance entre clases (especialmente en "Poor"), y estabilidad en las métricas. RandomForest superó consistentemente otras opciones.

---

## 😕 XGBoost: No Cumplió Expectativas

Probé XGBoost esperando un salto hacia 89-90%, pero se quedó en 81.96% de precisión y 0.808 de F1 Macro. Aunque optimicé parámetros como `n_estimators` y `max_depth`, no capturó las relaciones tan bien como RandomForest con este dataset. Creo que el desbalance o la complejidad de las características favorecieron más a RandomForest.

---

## 📜 Creación de Archivos de Configuración Simplificados

Finalmente, creé dos archivos `.yaml` simplificados para documentar mis mejores modelos:
- **`rf_config_1_simplified.yaml`**: Refleja el Modelo 1 con sus parámetros óptimos.
- **`rf_config_2_simplified.yaml`**: Refleja el Modelo 2, mi campeón.
- **Por Qué Simplificados**: Quité el `search_space` y `optimization` porque ya los probé en los notebooks, dejando solo los parámetros finales y métricas clave para uso directo.

Estos archivos son mi resumen final, listos para implementar o compartir.

---

## 🚀 Conclusión

Llegué a un tope de 83.1% con RandomForest, y aunque no alcancé un poco mas, estoy satisfecho con el proceso: limpié, mejoré y optimicé al máximo este dataset. Si tuviera más datos o nuevas características, podría ir más lejos, pero por ahora, ¡mis dos modelos son aceptables! ¿Qué te parecen?