# config.py
"""
Configuración central del proyecto de red neuronal para diabetes.
Define:
- Rutas
- Hiperparámetros de entrenamiento
- Columnas de características
"""

from dataclasses import dataclass
import os

# Columnas de entrada (en este orden)
FEATURE_COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

# Nombre de la columna objetivo
TARGET_COLUMN = "Outcome"


@dataclass
class TrainingConfig:
    # Fuente de datos (URL pública del dataset Pima Indians Diabetes)
    data_url: str = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv"

    # Rutas de archivos y carpetas
    data_dir: str = "data"
    data_filename: str = "diabetes.csv"

    model_dir: str = "models"
    model_name: str = "modelo_diabetes.keras"
    scaler_name: str = "scaler_diabetes.pkl"
    calibrator_name: str = "calibrador_diabetes.pkl"

    reports_dir: str = "reports"

    # Splits
    test_size: float = 0.20   # 20% test
    val_size: float = 0.10    # 10% de TODO (se saca del train)

    # Entrenamiento
    random_state: int = 42
    batch_size: int = 32
    epochs: int = 80
    learning_rate: float = 1e-3

    # Búsqueda de arquitecturas con cross-validation
    use_cross_validation: bool = True
    n_splits_cv: int = 5
    model_variant: str = "base"   # "small", "base" o "wide"

    # Métrica para EarlyStopping y ModelCheckpoint
    monitor_metric: str = "val_auc"
    monitor_mode: str = "max"

    @property
    def data_path(self) -> str:
        return os.path.join(self.data_dir, self.data_filename)

    @property
    def model_path(self) -> str:
        return os.path.join(self.model_dir, self.model_name)

    @property
    def scaler_path(self) -> str:
        return os.path.join(self.model_dir, self.scaler_name)

    @property
    def calibrator_path(self) -> str:
        return os.path.join(self.model_dir, self.calibrator_name)
