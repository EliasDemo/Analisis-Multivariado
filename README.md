<div align="center">

# 🧠 Predicción de Diabetes con Redes Neuronales  
### (Pima Indians Diabetes Project)

Modelado, entrenamiento y explicación de riesgo de diabetes tipo 2 con **Python + TensorFlow + scikit-learn**.

> ⚠️ Proyecto con fines **académicos y educativos**.  
> No debe usarse para diagnóstico médico real.

---

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![scikit--learn](https://img.shields.io/badge/scikit--learn-ML-yellow?logo=scikitlearn)
![Status](https://img.shields.io/badge/Status-Experimental-informational)

</div>

---

## 📚 Tabla de contenidos

1. [🎯 Objetivo del proyecto](#-objetivo-del-proyecto)  
2. [📂 Estructura del repositorio](#-estructura-del-repositorio)  
3. [🧩 Dataset y variables](#-dataset-y-variables)  
4. [⚙️ Instalación y configuración](#️-instalación-y-configuración)  
5. [🧪 Flujo de entrenamiento](#-flujo-de-entrenamiento)  
6. [🔮 Predicción sobre nuevos pacientes](#-predicción-sobre-nuevos-pacientes)  
7. [📊 Resultados típicos](#-resultados-típicos)  
8. [🧠 Por qué funciona este enfoque](#-por-qué-funciona-este-enfoque)  
9. [⚠️ Limitaciones y advertencias](#️-limitaciones-y-advertencias)  
10. [🚀 Ideas de mejora futura](#-ideas-de-mejora-futura)  
11. [🎤 Cómo presentar este proyecto](#-cómo-presentar-este-proyecto)

---

## 🎯 Objetivo del proyecto

Este proyecto busca construir un **pipeline completo de Machine Learning** que:

- Prediga la probabilidad de que una persona tenga **diabetes tipo 2** (0 = no, 1 = sí).
- Siga **buenas prácticas** de ingeniería de ML:
  - Preprocesamiento y estandarización correctos.
  - División clara en train / validación / test.
  - Comparación de arquitecturas con **validación cruzada (CV)**.
  - Uso de **BatchNormalization** y **Dropout** para regularizar.
  - Manejo de **desbalance de clases** con `class_weight`.
  - **Calibración de probabilidades** (Isotonic Regression).
- Permita:
  - Entrenar el modelo desde cero (`train_diabetes.py`).
  - Predecir el riesgo de nuevos pacientes desde CSV (`predict_diabetes.py`).
  - Generar **gráficas e interpretaciones** a nivel de paciente.

---

## 📂 Estructura del repositorio

```bash
proyecto_diabetes_nn/
├─ data/
│  ├─ diabetes.csv                # Dataset original (descarga automática)
│  └─ mis_pacientes.csv           # Ejemplo de pacientes nuevos
├─ models/
│  ├─ modelo_diabetes.keras       # Red neuronal entrenada
│  ├─ scaler_diabetes.pkl         # StandardScaler (normalización)
│  └─ calibrador_diabetes.pkl     # Calibrador de probabilidades
├─ reports/
│  ├─ training_curves.png         # Curvas de entrenamiento (loss / accuracy)
│  ├─ roc_curve.png               # Curva ROC
│  ├─ pr_curve.png                # Curva Precision–Recall
│  ├─ model_comparison_cv.png     # Comparación de arquitecturas (CV)
│  └─ explicacion_paciente.png    # Factores de riesgo de un paciente
├─ config.py                      # Configuración global del proyecto
├─ train_diabetes.py              # Entrenamiento + evaluación del modelo
├─ predict_diabetes.py            # Predicción para nuevos pacientes
└─ requirements.txt               # Dependencias del entorno
