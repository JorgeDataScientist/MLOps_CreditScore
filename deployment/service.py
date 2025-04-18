import os
import json
import bentoml
import pandas as pd
from pydantic import BaseModel, Field

# Obtiene el nombre del modelo y el input ficticio
model_name = os.getenv("MODEL_NAME", "model_1")
test_input = os.getenv("TEST_INPUT")

# Definir esquema Pydantic para las columnas
class CreditScoringInput(BaseModel):
    Edad: int = Field(30, description="Edad del cliente")
    Salario_Mensual: float = Field(6000, description="Salario mensual")
    Num_Tarjetas_Credito: int = Field(3, description="Número de tarjetas de crédito")
    Tasa_Interes: float = Field(15, description="Tasa de interés promedio")
    Retraso_Pago: int = Field(0, description="Días de retraso en pagos")
    Num_Pagos_Retrasados: int = Field(0, description="Número de pagos retrasados")
    Cambio_Limite_Credito: float = Field(0, description="Cambio en el límite de crédito")
    Num_Consultas_Credito: int = Field(2, description="Número de consultas de crédito")
    Deuda_Pendiente: float = Field(5000, description="Deuda total pendiente")
    Edad_Historial_Credito: int = Field(5, description="Años de historial crediticio")
    Total_Cuota_Mensual: float = Field(200, description="Cuota mensual total")
    Inversion_Mensual: float = Field(100, description="Inversión mensual")
    Saldo_Mensual: float = Field(1000, description="Saldo mensual")
    Comportamiento_de_Pago_High_spent_Large_value_payments: int = Field(0, description="Comportamiento de pago: Alto gasto, pagos grandes")
    Comportamiento_de_Pago_High_spent_Medium_value_payments: int = Field(0, description="Comportamiento de pago: Alto gasto, pagos medianos")
    Comportamiento_de_Pago_High_spent_Small_value_payments: int = Field(0, description="Comportamiento de pago: Alto gasto, pagos pequeños")
    Comportamiento_de_Pago_Low_spent_Large_value_payments: int = Field(0, description="Comportamiento de pago: Bajo gasto, pagos grandes")
    Comportamiento_de_Pago_Low_spent_Medium_value_payments: int = Field(1, description="Comportamiento de pago: Bajo gasto, pagos medianos")
    Comportamiento_de_Pago_Low_spent_Small_value_payments: int = Field(0, description="Comportamiento de pago: Bajo gasto, pagos pequeños")
    Mezcla_Crediticia_Bad: int = Field(0, description="Mezcla crediticia: Mala")
    Mezcla_Crediticia_Good: int = Field(1, description="Mezcla crediticia: Buena")
    Mezcla_Crediticia_Standard: int = Field(0, description="Mezcla crediticia: Estándar")
    Pago_Minimo_No: int = Field(1, description="Pago mínimo: No")
    Pago_Minimo_Yes: int = Field(0, description="Pago mínimo: Sí")
    Ocupacion_Architect: int = Field(0, description="Ocupación: Arquitecto")
    Ocupacion_Developer: int = Field(1, description="Ocupación: Desarrollador")
    Ocupacion_Doctor: int = Field(0, description="Ocupación: Doctor")
    Ocupacion_Engineer: int = Field(0, description="Ocupación: Ingeniero")
    Ocupacion_Entrepreneur: int = Field(0, description="Ocupación: Emprendedor")
    Ocupacion_Journalist: int = Field(0, description="Ocupación: Periodista")
    Ocupacion_Lawyer: int = Field(0, description="Ocupación: Abogado")
    Ocupacion_Manager: int = Field(0, description="Ocupación: Gerente")
    Ocupacion_Mechanic: int = Field(0, description="Ocupación: Mecánico")
    Ocupacion_Media_Manager: int = Field(0, description="Ocupación: Gerente de medios")
    Ocupacion_Musician: int = Field(0, description="Ocupación: Músico")
    Ocupacion_Scientist: int = Field(0, description="Ocupación: Científico")
    Ocupacion_Teacher: int = Field(0, description="Ocupación: Profesor")
    Ocupacion_Writer: int = Field(0, description="Ocupación: Escritor")
    debt_to_income: float = Field(0.833, description="Relación deuda-ingreso")
    payment_to_income: float = Field(0.033, description="Relación pago-ingreso")
    credit_history_ratio: float = Field(0.167, description="Relación historial crediticio")

# Define el servicio BentoML
@bentoml.service(
    name=f"credit_scoring_service_{model_name}",
    resources={"cpu": "2"}
)
class CreditScoringService:
    def __init__(self):
        # Carga el modelo
        self.model = bentoml.sklearn.load_model(f"credit_scoring_{model_name}:latest")

    @bentoml.api
    async def predict(self, input_data: CreditScoringInput = None) -> dict:
        """
        Endpoint para realizar predicciones con el modelo seleccionado.
        
        Args:
            input_data: Objeto con columnas esperadas. Si None, usa TEST_INPUT.
        
        Returns:
            Dict con las predicciones.
        """
        if input_data is None:
            input_df = pd.DataFrame(json.loads(test_input))
        else:
            input_df = pd.DataFrame([input_data.dict()])
        
        predictions = self.model.predict(input_df)
        return {"predictions": predictions.tolist()}