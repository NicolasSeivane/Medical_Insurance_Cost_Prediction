[![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/)

# 🤖Model Selection and Evaluation

---

## 🌎 Language / Idioma
- [English Version](#english)
- [Versión en Español](#español)

---

<a name="español"></a>
## 🇪🇸 Versión en Español

Este documento detalla el proceso de selección de modelos, comparando diferentes algoritmos de regresión para encontrar el mejor ajuste para predecir los cargos de seguros médicos.

---

## 🛠️ Stack Técnico
Este análisis fue realizado utilizando el lenguaje **Python** y las siguientes librerías:

*   **Pandas**: Manipulación de datos.
*   **Scikit-Learn**: Procesamiento de datos, creación de `Pipeline` y entrenamiento de modelos.
*   **Joblib**: Guardado del modelo en formato `.pkl`.

---

## 🔍 Metodología

Realizamos una búsqueda exhaustiva utilizando `GridSearchCV` y Validación Cruzada (K-Fold, k=5) para asegurar métricas de rendimiento robustas.

---

## 📋 Grillas Utilizadas para `GridSearchCV`

| Modelo | Parámetros / Configuración |
| :--- | :--- |
| **Regresión Lineal** | K-folds: 5 |
| **Regresión Polinómica** | K-folds: 5 \| Grado: 2, 3, 4 |
| **Árbol de Decisión** | K-folds: 5 \| Criterion: squared_error, friedman_mse, absolute_error \| Splitter: best, random \| Max depth: None, 10, 20, 30 \| Min samples split: 2, 5, 10 \| Min samples leaf: 1, 2, 4 \| Random state: 42 |
| **Gradient Boosting** | K-folds: 5 \| N_estimators: 100, 200, 300, 500 \| Learning rate: 0.001, 0.01, 0.1, 0.2 \| Max depth: 1, 3, 5, 7 \| Min samples split: 2, 5, 10 \| Min samples leaf: 1, 2, 4 \| Random state: 42 |

---

## 📊 Modelos Explorados y Resultados de Entrenamiento

| Modelo | R² Score (Test) | MAE | Tiempo (s) |
| :--- | :---: | :---: | :---: |
| **Regresión Lineal** | 0.7427 | 4227.84 | 0.0143 |
| **Regresión Polinómica (Grado 2)** | 0.8189 | 2766.92 | 19.2019 |
| **Árbol de Decisión** | 0.8577 | 1498.77 | 10.6018 |
| **Gradient Boosting** | 0.8834 | 2084.93 | 504.8198 |

---

## 🏆 Selección del Modelo Final

> Se seleccionó el **Árbol de Decisión (Decision Tree Regressor)** con ajuste de hiperparámetros para el pipeline final.

### Razones de la elección:
*   **Equilibrio Excelente**: Un alto puntaje R² (~0.85) con un Error Absoluto Medio (MAE) significativamente menor.
*   **Eficiencia**: Tiempos de entrenamiento y predicción mucho más rápidos que Gradient Boosting.
*   **Interpretabilidad**: Caminos de decisión más claros para entender incrementos de costos.

> **Mejores Parámetros (Árbol de Decisión):**
> *   `criterion`: 'absolute_error'
> *   `max_depth`: None
> *   `min_samples_split`: 10
> *   `splitter`: 'random'

---

## ⚙️ Integración del Pipeline
El modelo seleccionado se integra en un pipeline de extremo a extremo que incluye:
1. `ColumnTransformer` para escalados numéricos y codificación OneHot categórica.
2. El `DecisionTreeRegressor` ya entrenado.
3. Modelo exportado como archivo `.pkl`.

---

<a name="english"></a>
## 🇬🇧 English Version

This document details the model selection process, comparing different regression algorithms to find the best fit for predicting medical insurance charges.

---

## 🛠️ Technical Stack
This analysis was performed using **Python** and the following libraries:

*   **Pandas**: Data manipulation.
*   **Scikit-Learn**: Data preprocessing, `Pipeline` creation and model training.
*   **Joblib**: Model saving in `.pkl` format.

---

## 🔍 Methodology
We performed an extensive search using `GridSearchCV` and Cross-Validation (K-Fold, k=5) to ensure robust performance metrics.

---

## 📋 Grids Used for `GridSearchCV`

| Model | Parameters / Configuration |
| :--- | :--- |
| **Linear Regression** | K-folds: 5 |
| **Polynomial Regression** | K-folds: 5 \| Degree: 2, 3, 4 |
| **Decision Tree** | K-folds: 5 \| Criterion: squared_error, friedman_mse, absolute_error \| Splitter: best, random \| Max depth: None, 10, 20, 30 \| Min samples split: 2, 5, 10 \| Min samples leaf: 1, 2, 4 \| Random state: 42 |
| **Gradient Boosting** | K-folds: 5 \| N_estimators: 100, 200, 300, 500 \| Learning rate: 0.001, 0.01, 0.1, 0.2 \| Max depth: 1, 3, 5, 7 \| Min samples split: 2, 5, 10 \| Min samples leaf: 1, 2, 4 \| Random state: 42 |

---

## 📊 Models Explored and Training Results

| Model | R² Score (Test) | MAE | Training Time (s) |
| :--- | :---: | :---: | :---: |
| **Linear Regression** | 0.7427 | 4227.84 | 0.0143 |
| **Polynomial Regression (Deg 2)** | 0.8189 | 2766.92 | 19.2019 |
| **Step Tree / Decision Tree** | 0.8577 | 1498.77 | 10.6018 |
| **Gradient Boosting** | 0.8834 | 2084.93 | 504.8198 |

---

## 🏆 Final Model Selection

> [!IMPORTANT]
> The **Decision Tree Regressor** (with hyperparameter tuning) was selected for the final pipeline.

### Why this model?
*   **Excellent Balance**: A high R² score (~0.85) with a significantly lower Mean Absolute Error (MAE).
*   **Efficiency**: Much faster training and prediction times compared to Gradient Boosting.
*   **Interpretability**: Easier to understand the decision paths for cost increments.

> [!TIP]
> **Best Parameters (Decision Tree):**
> *   `criterion`: 'absolute_error'
> *   `max_depth`: None
> *   `min_samples_split`: 10
> *   `splitter`: 'random'

---

## ⚙️ Pipeline Integration
The selected model is integrated into an end-to-end pipeline that includes:
1. `ColumnTransformer` for numerical scaling and categorical OneHot encoding.
2. The trained `DecisionTreeRegressor`.
3. Model saved in `.pkl` format.