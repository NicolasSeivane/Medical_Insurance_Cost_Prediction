# Medical Insurance Cost Prediction - End-to-End ML Pipeline

[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-316192?style=for-the-badge&logo=postgresql&logoColor=white)](https://www.postgresql.org/)

This repository contains a professional End-to-End Machine Learning pipeline designed to predict medical insurance costs. The project covers the entire lifecycle: from data ingestion into a relational database to automated PDF report generation.

---

## 🌎 Language / Idioma
- [English Version](#english)
- [Versión en Español](#español)

---

<a name="english"></a>
## 🇬🇧 English Version

### 🚀 Overview
The goal of this project is to provide a robust and scalable solution for predicting insurance charges based on demographic and health data. It leverages a containerized environment to ensure consistency across different systems.

### ✨ Key Features
- **Automated Data Ingestion**: Seamlessly loads raw CSV data into a PostgreSQL database with duplicate prevention.
- **Robust Training**: Uses `DecisionTreeRegressor` with `GridSearchCV` and Cross-Validation (K-Fold with Shuffle) for optimized performance ($R^2 \approx 0.85$).
- **Automated Reporting**: Generates visual PNG summaries and a professional LaTeX-based PDF report upon completion.
- **Dockerized Environment**: Fully containerized with custom Docker volumes for data persistence (Models, Reports, and SQL Schemas).

### 🛠 Tech Stack
- **Languages**: Python (Pandas, Scikit-Learn, Matplotlib, Jinja2).
- **Database**: PostgreSQL 15.
- **Reporting**: LaTeX (pdflatex).
- **DevOps**: Docker & Docker Compose.

---

<a name="español"></a>
## 🇪🇸 Versión en Español

### 🚀 Resumen
El objetivo de este proyecto es proporcionar una solución robusta y escalable para predecir costos de seguros médicos basados en datos demográficos y de salud. Utiliza un entorno contenedorizado para asegurar la consistencia entre diferentes sistemas.

### ✨ Características Principales
- **Ingesta Automatizada**: Carga datos desde CSV a una base de datos PostgreSQL con mecanismos de prevención de duplicados.
- **Entrenamiento Robusto**: Utiliza `DecisionTreeRegressor` con `GridSearchCV` y Validación Cruzada con barajado (Shuffle) para un rendimiento óptimo ($R^2 \approx 0.85$).
- **Informes Automatizados**: Genera resúmenes visuales en PNG y un reporte profesional en PDF basado en LaTeX al finalizar el pipeline.
- **Entorno Dockerizado**: Totalmente contenedorizado con volúmenes personalizados para la persistencia de datos (Modelos, Reportes y Esquemas SQL).

### 🛠 Tecnologías Utilizadas
- **Lenguajes**: Python (Pandas, Scikit-Learn, Matplotlib, Jinja2).
- **Base de Datos**: PostgreSQL 15.
- **Reportes**: LaTeX (pdflatex).
- **DevOps**: Docker y Docker Compose.

---

## 📁 Project Structure / Estructura del Proyecto

```text
├── app/               # Source code (Training, Ingestion, Scoring)
├── data/              # Raw datasets (CSV)
├── db/                # SQL scripts and persisted schemas
├── models/            # Trained models (.pkl)
├── reports/           # LaTeX templates and final PDF reports
├── notebooks/         # Exploratory Data Analysis (EDA)
├── Dockerfile         # Container definition
└── docker-compose.yml # Orchestration of app and database
```

## ⚙️ How to Run / Cómo Ejecutar

For detailed commands, please check the [DOCKER_GUIDE.md](./DOCKER_GUIDE.md).

1. **Clone the repo.**
2. **Execute the pipeline:**
   ```bash
   docker-compose up --build
   ```
3. **Check results:**
   - Reports in `reports/outputs/`
   - SQL schema in `db/schema.sql`
   - Models in `models/`
