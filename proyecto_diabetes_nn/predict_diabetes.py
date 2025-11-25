# predict_diabetes.py
"""
Uso del modelo entrenado para predecir diabetes en nuevos pacientes.

Ejecución típica:
    python predict_diabetes.py --input data/mis_pacientes.csv

El CSV de entrada debe tener estas columnas (en este orden o al menos con estos nombres):
    Pregnancies, Glucose, BloodPressure, SkinThickness,
    Insulin, BMI, DiabetesPedigreeFunction, Age

Salida:
    - Crea data/mis_pacientes_con_prediccion.csv con:
      prob_raw (sin calibrar), prob_diabetes (calibrada si hay calibrador) y prediccion (1 = diabético, 0 = no)
    - Si solo hay 1 paciente:
      * Muestra explicación detallada en consola
      * Genera reports/explicacion_paciente.png (z-scores de cada variable)
"""

import argparse
import os
import logging

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import tensorflow as tf

from config import TrainingConfig, FEATURE_COLUMNS


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
    )


def load_model_scaler_calibrator(cfg: TrainingConfig):
    if not os.path.exists(cfg.model_path):
        raise FileNotFoundError(f"No se encontró el modelo: {cfg.model_path}")
    if not os.path.exists(cfg.scaler_path):
        raise FileNotFoundError(f"No se encontró el scaler: {cfg.scaler_path}")

    model = tf.keras.models.load_model(cfg.model_path)
    scaler = joblib.load(cfg.scaler_path)

    calibrator = None
    if os.path.exists(cfg.calibrator_path):
        calibrator = joblib.load(cfg.calibrator_path)
        logging.info("Calibrador de probabilidades cargado desde: %s", cfg.calibrator_path)
    else:
        logging.info("No se encontró calibrador, se usarán probabilidades sin calibrar.")

    return model, scaler, calibrator


def explain_single_patient(row: pd.Series,
                           prob_raw: float,
                           prob_cal: float,
                           pred: int,
                           scaler,
                           cfg: TrainingConfig):
    """
    Explica un solo paciente:
    - Imprime valores
    - Probabilidad de diabetes / no diabetes (cruda y calibrada)
    - Factores de riesgo (z-scores más altos)
    - Genera gráfico de barras de z-scores
    """
    logging.info("===== EXPLICACIÓN DETALLADA DEL PACIENTE =====")

    # Asegurar orden de columnas
    row = row[FEATURE_COLUMNS]
    values = row.values.astype(float)

    mean = scaler.mean_
    scale = scaler.scale_

    # z-score: (x - media) / desviación
    z_scores = (values - mean) / scale

    logging.info("Valores del paciente:")
    for name, val in zip(FEATURE_COLUMNS, values):
        logging.info("  %s = %.3f", name, val)

    logging.info("Probabilidad de DIABETES (cruda): %.3f", prob_raw)
    if prob_cal is not None:
        logging.info("Probabilidad de DIABETES (calibrada): %.3f", prob_cal)
        logging.info("Probabilidad de NO diabetes (calibrada): %.3f", 1 - prob_cal)
        prob_to_report = prob_cal
    else:
        logging.info("Probabilidad de NO diabetes (cruda): %.3f", 1 - prob_raw)
        prob_to_report = prob_raw

    logging.info("Clasificación del modelo: %s", "DIABÉTICO" if pred == 1 else "NO DIABÉTICO")

    # Principales factores de riesgo (valor absoluto de z-score)
    idx_sorted = np.argsort(-np.abs(z_scores))  # descendente
    top_n = min(3, len(FEATURE_COLUMNS))
    logging.info("Principales factores (z-scores más alejados de la media):")
    for i in range(top_n):
        idx = idx_sorted[i]
        feature = FEATURE_COLUMNS[idx]
        z = z_scores[idx]
        if z > 0:
            interpretacion = "por ENCIMA de la media del dataset"
        else:
            interpretacion = "por DEBAJO de la media del dataset"
        logging.info("  %s: z=%.2f (%s)", feature, z, interpretacion)

    # Generar gráfico de barras de z-scores
    os.makedirs(cfg.reports_dir, exist_ok=True)

    plt.figure(figsize=(8, 4))
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.bar(FEATURE_COLUMNS, z_scores)
    plt.title(f"Paciente vs. media del dataset (z-scores)\nProb. diabetes ~ {prob_to_report:.2f}")
    plt.ylabel("z-score")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    fig_path = os.path.join(cfg.reports_dir, "explicacion_paciente.png")
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()

    logging.info("Gráfico de explicación guardado en: %s", fig_path)
    logging.info("===== FIN DE EXPLICACIÓN =====")


def main():
    setup_logging()
    cfg = TrainingConfig()

    parser = argparse.ArgumentParser(
        description="Predicción de diabetes usando el modelo entrenado."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Ruta al archivo CSV con pacientes a evaluar.",
    )

    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"No se encontró el archivo: {args.input}")

    logging.info("Cargando modelo, scaler y (si existe) calibrador...")
    model, scaler, calibrator = load_model_scaler_calibrator(cfg)

    logging.info("Cargando datos de entrada desde: %s", args.input)
    df_new = pd.read_csv(args.input)

    # Validar columnas
    missing = [c for c in FEATURE_COLUMNS if c not in df_new.columns]
    if missing:
        raise ValueError(
            f"Faltan columnas en el CSV de entrada: {missing}\n"
            f"Se esperan las columnas: {FEATURE_COLUMNS}"
        )

    # Reordenar columnas según FEATURE_COLUMNS
    df_new = df_new[FEATURE_COLUMNS]

    # Transformar con el mismo scaler
    X_new = scaler.transform(df_new.values)
    proba_raw = model.predict(X_new).ravel()

    if calibrator is not None:
        proba_cal = calibrator.predict(proba_raw)
    else:
        proba_cal = proba_raw

    pred = (proba_cal >= 0.5).astype(int)

    df_result = df_new.copy()
    df_result["prob_raw"] = proba_raw
    df_result["prob_diabetes"] = proba_cal
    df_result["prediccion"] = pred  # 1 = diabético, 0 = no

    base, ext = os.path.splitext(args.input)
    out_path = base + "_con_prediccion.csv"
    df_result.to_csv(out_path, index=False)

    logging.info("Predicciones guardadas en: %s", out_path)
    logging.info("Primeras filas con predicción:\n%s", df_result.head().to_string())

    # Si solo hay 1 paciente, damos una explicación visual y textual
    if len(df_new) == 1:
        row = df_new.iloc[0]
        explain_single_patient(
            row=row,
            prob_raw=float(proba_raw[0]),
            prob_cal=float(proba_cal[0]) if calibrator is not None else None,
            pred=int(pred[0]),
            scaler=scaler,
            cfg=cfg,
        )
    else:
        logging.info(
            "Archivo contiene %d pacientes. Para explicación detallada usa un CSV con 1 sola fila.",
            len(df_new),
        )


if __name__ == "__main__":
    main()
