# train_diabetes.py
"""
Entrenamiento de una red neuronal para predicción de diabetes (Pima Indians).

✔ Descarga el dataset automáticamente si no existe (desde GitHub).
✔ Búsqueda de mejor arquitectura con cross-validation (CV) usando AUC.
✔ Separa train / valid / test.
✔ Escala los datos (StandardScaler).
✔ Entrena una red neuronal con:
    - BatchNormalization
    - Dropout
    - EarlyStopping
    - ModelCheckpoint
    - class_weight para desbalance 0/1
✔ Guarda:
    - Modelo entrenado    → models/modelo_diabetes.keras
    - Scaler (StandardScaler) → models/scaler_diabetes.pkl
    - Calibrador de probabilidades → models/calibrador_diabetes.pkl
✔ Genera gráficas:
    - reports/training_curves.png
    - reports/roc_curve.png
    - reports/pr_curve.png
    - reports/model_comparison_cv.png

Ejecución:
    python train_diabetes.py
"""

import os
import logging
import urllib.request
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
    brier_score_loss,
    auc,
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.isotonic import IsotonicRegression

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers

from config import TrainingConfig, FEATURE_COLUMNS, TARGET_COLUMN


# ---------------------------
# 1) Logging
# ---------------------------

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
    )


# ---------------------------
# 2) Dataset: descarga y carga
# ---------------------------

def ensure_dataset(cfg: TrainingConfig):
    """
    Crea la carpeta data/ y descarga el CSV de diabetes desde GitHub si no existe.
    """
    os.makedirs(cfg.data_dir, exist_ok=True)

    if not os.path.exists(cfg.data_path):
        logging.info("📥 Descargando dataset de diabetes desde GitHub...")
        urllib.request.urlretrieve(cfg.data_url, cfg.data_path)
        logging.info("✅ Dataset descargado en: %s", cfg.data_path)
    else:
        logging.info("✅ Dataset ya existe en: %s", cfg.data_path)


def load_data(cfg: TrainingConfig) -> pd.DataFrame:
    """
    Carga el CSV en un DataFrame y asigna nombres de columnas.
    El CSV original NO trae encabezados.
    """
    columnas = FEATURE_COLUMNS + [TARGET_COLUMN]
    df = pd.read_csv(cfg.data_path, header=None, names=columnas)

    logging.info("Primeras filas del dataset:\n%s", df.head().to_string())
    logging.info("Distribución de Outcome:\n%s", df[TARGET_COLUMN].value_counts(normalize=True))

    return df


# ---------------------------
# 3) Preprocesamiento train/val/test
# ---------------------------

def split_and_scale(
    df: pd.DataFrame, cfg: TrainingConfig
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Separa en X (features) e y (target), hace split train/val/test y aplica StandardScaler.
    """
    X = df[FEATURE_COLUMNS].values
    y = df[TARGET_COLUMN].astype(int).values

    # Train / Test
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y,
    )

    # De lo que queda para train sacamos validación
    val_ratio = cfg.val_size / (1.0 - cfg.test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=val_ratio,
        random_state=cfg.random_state,
        stratify=y_train_full,
    )

    logging.info("Tamaño X_train: %s", X_train.shape)
    logging.info("Tamaño X_val:   %s", X_val.shape)
    logging.info("Tamaño X_test:  %s", X_test.shape)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, scaler


# ---------------------------
# 4) Arquitecturas y construcción de modelo
# ---------------------------

def build_model_variant(input_dim: int, cfg: TrainingConfig, variant: str) -> tf.keras.Model:
    """
    Crea una red neuronal según la variante:
      - "small": 32-16
      - "base":  64-32
      - "wide":  128-64-32
    Todas con BatchNormalization + Dropout.
    """
    if variant == "small":
        hidden_units: List[int] = [32, 16]
        dropouts: List[float] = [0.2, 0.2]
    elif variant == "wide":
        hidden_units = [128, 64, 32]
        dropouts = [0.4, 0.3, 0.2]
    else:  # "base"
        hidden_units = [64, 32]
        dropouts = [0.3, 0.2]

    model = models.Sequential()
    model.add(layers.Input(shape=(input_dim,)))
    for units, d in zip(hidden_units, dropouts):
        model.add(layers.Dense(units, activation="relu"))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(d))
    model.add(layers.Dense(1, activation="sigmoid"))

    optimizer = optimizers.Adam(learning_rate=cfg.learning_rate)

    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )

    return model


def build_model(input_dim: int, cfg: TrainingConfig) -> tf.keras.Model:
    model = build_model_variant(input_dim, cfg, cfg.model_variant)
    logging.info("Resumen del modelo (%s):", cfg.model_variant)
    model.summary(print_fn=logging.info)
    return model


# ---------------------------
# 5) Cross-validation para elegir arquitectura
# ---------------------------

def cross_validate_architectures(df: pd.DataFrame, cfg: TrainingConfig) -> str:
    """
    Compara varias arquitecturas con StratifiedKFold CV usando AUC.
    Devuelve el nombre de la mejor variante.
    """
    X = df[FEATURE_COLUMNS].values
    y = df[TARGET_COLUMN].astype(int).values

    variants = ["small", "base", "wide"]
    results: List[Dict] = []

    skf = StratifiedKFold(
        n_splits=cfg.n_splits_cv,
        shuffle=True,
        random_state=cfg.random_state,
    )

    logging.info("===== INICIO CROSS-VALIDATION PARA ELEGIR ARQUITECTURA =====")

    for variant in variants:
        fold_aucs: List[float] = []
        fold_id = 1

        for train_idx, val_idx in skf.split(X, y):
            X_tr, X_va = X[train_idx], X[val_idx]
            y_tr, y_va = y[train_idx], y[val_idx]

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_va_s = scaler.transform(X_va)

            model = build_model_variant(input_dim=X.shape[1], cfg=cfg, variant=variant)

            es = callbacks.EarlyStopping(
                monitor="val_auc",
                mode="max",
                patience=10,
                restore_best_weights=True,
                verbose=0,
            )

            model.fit(
                X_tr_s,
                y_tr,
                validation_data=(X_va_s, y_va),
                epochs=min(40, cfg.epochs),
                batch_size=cfg.batch_size,
                callbacks=[es],
                verbose=0,
            )

            y_va_proba = model.predict(X_va_s).ravel()
            fold_auc = roc_auc_score(y_va, y_va_proba)
            fold_aucs.append(fold_auc)
            logging.info("[CV] Variante=%s | Fold %d/%d | AUC=%.4f", variant, fold_id, cfg.n_splits_cv, fold_auc)
            fold_id += 1

        mean_auc = float(np.mean(fold_aucs))
        std_auc = float(np.std(fold_aucs))
        logging.info("[CV] Variante=%s | AUC medio=%.4f ± %.4f", variant, mean_auc, std_auc)

        results.append({"variant": variant, "mean_auc": mean_auc, "std_auc": std_auc})

    os.makedirs(cfg.reports_dir, exist_ok=True)
    df_res = pd.DataFrame(results)
    cv_path = os.path.join(cfg.reports_dir, "model_comparison_cv.csv")
    df_res.to_csv(cv_path, index=False)
    logging.info("Resultados de CV guardados en: %s", cv_path)

    # Gráfico de comparación
    plt.figure(figsize=(6, 4))
    plt.bar(df_res["variant"], df_res["mean_auc"])
    plt.ylabel("AUC medio (CV)")
    plt.title("Comparación de arquitecturas (Cross-Validation)")
    plt.tight_layout()
    plot_path = os.path.join(cfg.reports_dir, "model_comparison_cv.png")
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    logging.info("Gráfico de comparación de arquitecturas guardado en: %s", plot_path)

    best_row = max(results, key=lambda r: r["mean_auc"])
    logging.info("Mejor arquitectura según CV: %s (AUC=%.4f)", best_row["variant"], best_row["mean_auc"])
    return best_row["variant"]


# ---------------------------
# 6) Entrenamiento final con class_weight
# ---------------------------

def train_model(
    model: tf.keras.Model,
    cfg: TrainingConfig,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> tf.keras.callbacks.History:

    os.makedirs(cfg.model_dir, exist_ok=True)
    os.makedirs(cfg.reports_dir, exist_ok=True)

    # Pesos de clase para desbalance (0 / 1)
    classes = np.unique(y_train)
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train,
    )
    class_weight_dict = {int(cls): float(w) for cls, w in zip(classes, class_weights)}
    logging.info("Pesos de clase usados en el entrenamiento: %s", class_weight_dict)

    early_stop = callbacks.EarlyStopping(
        monitor=cfg.monitor_metric,
        mode=cfg.monitor_mode,
        patience=15,
        restore_best_weights=True,
        verbose=1,
    )

    checkpoint = callbacks.ModelCheckpoint(
        filepath=cfg.model_path,
        monitor=cfg.monitor_metric,
        mode=cfg.monitor_mode,
        save_best_only=True,
        verbose=1,
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        callbacks=[early_stop, checkpoint],
        verbose=1,
        class_weight=class_weight_dict,
    )

    return history


# ---------------------------
# 7) Evaluación + curvas
# ---------------------------

def evaluate_model(
    model: tf.keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
):
    logging.info("Evaluando en conjunto de prueba...")

    loss, acc, auc_val = model.evaluate(X_test, y_test, verbose=0)
    logging.info("Loss test: %.4f", loss)
    logging.info("Accuracy test: %.4f", acc)
    logging.info("AUC test: %.4f", auc_val)

    y_proba = model.predict(X_test).ravel()
    y_pred = (y_proba >= 0.5).astype(int)

    logging.info("Reporte de clasificación:\n%s", classification_report(y_test, y_pred))
    logging.info("Matriz de confusión:\n%s", confusion_matrix(y_test, y_pred))

    roc_auc = roc_auc_score(y_test, y_proba)
    logging.info("ROC-AUC test (probabilidades crudas): %.4f", roc_auc)

    return y_proba, y_pred


def plot_history(history: tf.keras.callbacks.History, cfg: TrainingConfig):
    plt.figure(figsize=(10, 4))

    # Pérdida
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Train")
    plt.plot(history.history["val_loss"], label="Val")
    plt.title("Pérdida (loss)")
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.legend()

    # Exactitud
    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"], label="Train")
    plt.plot(history.history["val_accuracy"], label="Val")
    plt.title("Exactitud (accuracy)")
    plt.xlabel("Época")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    path = os.path.join(cfg.reports_dir, "training_curves.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    logging.info("Curvas de entrenamiento guardadas en: %s", path)


def plot_roc_pr_curves(y_test: np.ndarray, y_proba: np.ndarray, cfg: TrainingConfig):
    os.makedirs(cfg.reports_dir, exist_ok=True)

    # ROC
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--", label="Azar")
    plt.xlabel("Tasa de falsos positivos (1 - Especificidad)")
    plt.ylabel("Tasa de verdaderos positivos (Sensibilidad)")
    plt.title("Curva ROC - Modelo Diabetes")
    plt.legend()
    plt.tight_layout()
    roc_path = os.path.join(cfg.reports_dir, "roc_curve.png")
    plt.savefig(roc_path, bbox_inches="tight")
    plt.close()
    logging.info("Curva ROC guardada en: %s", roc_path)

    # Precision-Recall
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f"AP = {pr_auc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Curva Precision-Recall - Modelo Diabetes")
    plt.legend()
    plt.tight_layout()
    pr_path = os.path.join(cfg.reports_dir, "pr_curve.png")
    plt.savefig(pr_path, bbox_inches="tight")
    plt.close()
    logging.info("Curva Precision-Recall guardada en: %s", pr_path)


# ---------------------------
# 8) Calibración de probabilidades
# ---------------------------

def calibrate_probabilities(model: tf.keras.Model, X_val: np.ndarray, y_val: np.ndarray, cfg: TrainingConfig):
    """
    Ajusta las probabilidades del modelo usando Isotonic Regression
    sobre el conjunto de validación y guarda el calibrador.
    """
    logging.info("Calibrando probabilidades con IsotonicRegression (usando conjunto de validación)...")
    y_val_proba = model.predict(X_val).ravel()

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(y_val_proba, y_val)

    os.makedirs(cfg.model_dir, exist_ok=True)
    joblib.dump(iso, cfg.calibrator_path)
    logging.info("Calibrador guardado en: %s", cfg.calibrator_path)

    raw_brier = brier_score_loss(y_val, y_val_proba)
    cal_proba = iso.predict(y_val_proba)
    cal_brier = brier_score_loss(y_val, cal_proba)

    logging.info("Brier (val) sin calibrar: %.4f | calibrado: %.4f", raw_brier, cal_brier)


def evaluate_calibrated(y_test: np.ndarray, y_proba_raw: np.ndarray, cfg: TrainingConfig):
    """
    Evalúa el efecto de la calibración en el conjunto de test (solo para reporte).
    """
    if not os.path.exists(cfg.calibrator_path):
        logging.info("No se encontró calibrador, se omite evaluación calibrada.")
        return

    iso: IsotonicRegression = joblib.load(cfg.calibrator_path)
    y_proba_cal = iso.predict(y_proba_raw)

    auc_raw = roc_auc_score(y_test, y_proba_raw)
    auc_cal = roc_auc_score(y_test, y_proba_cal)

    brier_raw = brier_score_loss(y_test, y_proba_raw)
    brier_cal = brier_score_loss(y_test, y_proba_cal)

    logging.info("AUC test sin calibrar: %.4f | calibrado: %.4f", auc_raw, auc_cal)
    logging.info("Brier test sin calibrar: %.4f | calibrado: %.4f", brier_raw, brier_cal)


# ---------------------------
# 9) Guardar scaler y demo de paciente
# ---------------------------

def save_scaler(scaler: StandardScaler, cfg: TrainingConfig):
    os.makedirs(cfg.model_dir, exist_ok=True)
    joblib.dump(scaler, cfg.scaler_path)
    logging.info("Scaler guardado en: %s", cfg.scaler_path)


def example_patient(model: tf.keras.Model, scaler: StandardScaler):
    ejemplo = np.array([[6, 148, 72, 35, 0, 33.6, 0.627, 50]])
    ejemplo_scaled = scaler.transform(ejemplo)
    proba = model.predict(ejemplo_scaled)[0, 0]

    logging.info("Paciente de ejemplo:")
    logging.info("Pregnancies=6, Glucose=148, BloodPressure=72, SkinThickness=35, Insulin=0, BMI=33.6, DPF=0.627, Age=50")
    logging.info("Probabilidad de diabetes (sin calibrar): %.3f", proba)
    logging.info("Probabilidad de NO diabetes: %.3f", 1 - proba)
    logging.info("Predicción: %s", "DIABÉTICO" if proba >= 0.5 else "NO DIABÉTICO")


# ---------------------------
# 10) main
# ---------------------------

def main():
    setup_logging()
    cfg = TrainingConfig()

    np.random.seed(cfg.random_state)
    tf.random.set_seed(cfg.random_state)

    logging.info("===== INICIO ENTRENAMIENTO MODELO DIABETES =====")

    ensure_dataset(cfg)
    df = load_data(cfg)

    # 1) Cross-validation para elegir arquitectura (opcional)
    if cfg.use_cross_validation:
        best_variant = cross_validate_architectures(df, cfg)
        cfg.model_variant = best_variant
    else:
        logging.info("Saltando cross-validation, usando arquitectura por defecto: %s", cfg.model_variant)

    # 2) Split train/val/test + scaler
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = split_and_scale(df, cfg)

    # 3) Construir modelo final con la mejor arquitectura
    model = build_model(input_dim=X_train.shape[1], cfg=cfg)

    # 4) Entrenar con EarlyStopping + ModelCheckpoint + class_weight
    history = train_model(model, cfg, X_train, y_train, X_val, y_val)

    # 5) Cargar mejor modelo desde disco
    if os.path.exists(cfg.model_path):
        logging.info("Cargando mejor modelo desde disco: %s", cfg.model_path)
        model = tf.keras.models.load_model(cfg.model_path)

    # 6) Evaluación en test
    y_proba, y_pred = evaluate_model(model, X_test, y_test)

    # 7) Gráficas de entrenamiento y ROC/PR
    plot_history(history, cfg)
    plot_roc_pr_curves(y_test, y_proba, cfg)

    # 8) Guardar scaler
    save_scaler(scaler, cfg)

    # 9) Calibración de probabilidades y evaluación calibrada
    calibrate_probabilities(model, X_val, y_val, cfg)
    evaluate_calibrated(y_test, y_proba, cfg)

    # 10) Ejemplo rápido de paciente
    example_patient(model, scaler)

    logging.info("===== FIN ENTRENAMIENTO =====")


if __name__ == "__main__":
    main()
