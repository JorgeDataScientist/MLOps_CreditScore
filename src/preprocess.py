import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from omegaconf import DictConfig
from pathlib import Path
import pandera as pa
from typing import Optional
from src.utils import logger, ensure_dir, validate_data

def load_data(file_path: str) -> pd.DataFrame:
    """Carga datos desde un CSV."""
    logger.info(f"Cargando datos desde {file_path}")
    return pd.read_csv(file_path)

def rename_columns(df: pd.DataFrame, translations: dict) -> pd.DataFrame:
    """Renombra columnas según traducciones."""
    return df.rename(columns=translations)

def clean_data(df: pd.DataFrame, config: DictConfig) -> pd.DataFrame:
    """Aplica reglas de limpieza."""
    df_clean = df.copy()
    df_clean = df_clean[df_clean["Age"] >= config.cleaning.min_age]
    df_clean["Credit_History_Age"] = pd.to_numeric(
        df_clean["Credit_History_Age"], errors="coerce"
    ).fillna(0)
    df_clean["age_credit_ratio"] = df_clean["Age"] / (
        df_clean["Credit_History_Age"] / 12
    )
    df_clean = df_clean[df_clean["age_credit_ratio"] <= config.cleaning.max_age_credit_ratio]
    df_clean = df_clean.drop(columns=config.cleaning.drop_columns, errors="ignore")
    df_clean = df_clean.drop(columns=["age_credit_ratio"])
    return df_clean

def encode_target(df: pd.DataFrame, target_col: str, encoding: dict) -> pd.DataFrame:
    """Codifica la columna objetivo a valores numéricos."""
    df_encoded = df.copy()
    df_encoded[target_col] = df_encoded[target_col].map(encoding)
    return df_encoded.dropna(subset=[target_col])

def encode_categorical(df: pd.DataFrame, column: str, drop: Optional[str]) -> pd.DataFrame:
    """Codifica una columna categórica con OneHotEncoding."""
    ohe = OneHotEncoder(sparse_output=False, drop=drop, handle_unknown="ignore")
    encoded_data = ohe.fit_transform(df[[column]])
    encoded_cols = (
        [f"{column}_{cat}" for cat in ohe.categories_[0]]
        if drop is None
        else [f"{column}_{cat}" for i, cat in enumerate(ohe.categories_[0]) if i != ohe.drop_idx_[0]]
    )
    encoded_df = pd.DataFrame(encoded_data, columns=encoded_cols, index=df.index)
    return pd.concat([df.drop(columns=[column]), encoded_df], axis=1)

def create_new_features(df: pd.DataFrame, new_features: list) -> pd.DataFrame:
    """Crea nuevas features según fórmulas."""
    df_new = df.copy()
    for feature in new_features:
        try:
            df_new[feature.name] = df_new.eval(feature.formula).replace([np.inf, -np.inf], np.nan).fillna(0)
        except Exception as e:
            logger.warning(f"Error en {feature.name}: {e}")
            df_new[feature.name] = 0
    return df_new

def define_schema(config: DictConfig) -> pa.DataFrameSchema:
    """Define un esquema para validar datos."""
    schema_dict = {col: pa.Column(float, nullable=True) for col in config.features if col not in config.encoding}
    schema_dict.update(
        {
            col: pa.Column(float, nullable=True)
            for col in [f.name for f in config.new_features]
        }
    )
    for col in config.encoding:
        schema_dict.update({f"{col}_{cat}": pa.Column(float, nullable=True) for cat in ["dummy"]})
    if config.target in config.features:
        schema_dict[config.target] = pa.Column(int, nullable=False)
    return pa.DataFrameSchema(schema_dict, strict=False)

def preprocess_data(config: DictConfig, is_train: bool = True) -> Optional[tuple]:
    """Procesa datos crudos o experimentales."""
    input_path = config.data.raw.train if is_train else config.data.raw.experimental
    if not Path(input_path).exists():
        logger.error(f"Archivo no encontrado: {input_path}")
        return None

    df = load_data(input_path)
    translations = {
        "Age": "Edad",
        "Monthly_Inhand_Salary": "Salario_Mensual",
        "Num_Credit_Card": "Num_Tarjetas_Credito",
        "Interest_Rate": "Tasa_Interes",
        "Delay_from_due_date": "Retraso_Pago",
        "Num_of_Delayed_Payment": "Num_Pagos_Retrasados",
        "Changed_Credit_Limit": "Cambio_Limite_Credito",
        "Num_Credit_Inquiries": "Num_Consultas_Credito",
        "Outstanding_Debt": "Deuda_Pendiente",
        "Credit_History_Age": "Edad_Historial_Credito",
        "Total_EMI_per_month": "Total_Cuota_Mensual",
        "Amount_invested_monthly": "Inversion_Mensual",
        "Monthly_Balance": "Saldo_Mensual",
        "Payment_Behaviour": "Comportamiento_de_Pago",
        "Credit_Mix": "Mezcla_Crediticia",
        "Payment_of_Min_Amount": "Pago_Minimo",
        "Occupation": "Ocupacion",
        "Credit_Score": "Puntaje_Credito",
    }
    df = rename_columns(df, translations)

    if is_train:
        df = encode_target(df, config.target, config.target_encoding)
    df = clean_data(df, config)
    for col, params in config.encoding.items():
        if col in df.columns:
            df = encode_categorical(df, col, params.drop)
    df = create_new_features(df, config.new_features)

    # Validar esquema
    schema = define_schema(config)
    try:
        schema.validate(df, lazy=True)
    except pa.errors.SchemaError as e:
        logger.error(f"Validación fallida: {e}")
        return None

    if is_train:
        X = df.drop(columns=[config.target], errors="ignore")
        y = df[config.target]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.test_size, random_state=config.random_state, stratify=y
        )
        ensure_dir(config.data.processed.train.X_train)
        X_train.to_csv(config.data.processed.train.X_train, index=False)
        X_test.to_csv(config.data.processed.train.X_test, index=False)
        y_train.to_csv(config.data.processed.train.y_train, index=False)
        y_test.to_csv(config.data.processed.train.y_test, index=False)
        logger.info("Datos procesados guardados en data/processed/train/")
        return X_train, X_test, y_train, y_test
    else:
        ensure_dir(config.data.processed.experimental.processed)
        df.to_csv(config.data.processed.experimental.processed, index=False)
        logger.info(f"Datos procesados guardados en {config.data.processed.experimental.processed}")
        return df

if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig

    @hydra.main(config_path="../config", config_name="main", version_base="1.3")
    def main(config: DictConfig):
        if config.stage in ["all", "preprocess"]:
            preprocess_data(config, is_train=True)
            preprocess_data(config, is_train=False)

    main()
