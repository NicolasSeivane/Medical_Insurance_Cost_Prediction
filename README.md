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
- **Robust Training**: Uses `DecisionTreeRegressor` (optimized via GridSearchCV) for predictive performance. See [Model Selection Details](./Model_selection.md).
- **Exploratory Analysis**: Comprehensive data insights extracted from raw datasets. See [EDA Details](./EDA_EN.md).
- **Automated Reporting**: Generates visual PNG summaries and a professional LaTeX-based PDF report.
- **Dockerized Environment**: Fully containerized with custom Docker volumes for data persistence.
- **Interactive Dashboard**: Streamlit interface for real-time predictions. See [Dashboard Info](./Streamlit_Info.md).

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
- **Ingesta Automatizada**: Carga datos desde CSV a una base de datos PostgreSQL.
- **Entrenamiento Robusto**: Optimizado con una selección de modelos rigurosa. Ver [Detalles de Selección de Modelos](./Model_selection.md).
- **Análisis Exploratorio**: Insights detallados sobre el dataset. Ver [Detalles del EDA](./EDA.md).
- **Informes Automatizados**: Genera resúmenes visuales y reportes profesionales en PDF.
- **Entorno Dockerizado**: Totalmente contenedorizado para asegurar consistencia.
- **Dashboard Interactivo**: Interfaz Streamlit para predicciones en tiempo real. Ver [Info del Dashboard](./Streamlit_Info.md).

### 🛠 Tecnologías Utilizadas
- **Lenguajes**: Python (Pandas, Scikit-Learn, Matplotlib, Jinja2).
- **Base de Datos**: PostgreSQL 15.
- **Reportes**: LaTeX (pdflatex).
- **DevOps**: Docker y Docker Compose.

---

## 📁 Project Structure / Estructura del Proyecto

```text
├── app/               # Source code (DB creation, Ingestion, Training, Scoring)
├── streamlit/         # Streamlit App
├── data/              # Raw datasets (CSV)
├── db/                # SQL schema
├── models/            # Trained models (.pkl)
├── reports/           # LaTeX templates and reports
│   ├── figures/       # Pipeline manual figures
│   ├── outputs/       # Pipeline manual PDF reports
│   └── streamlit_figures/ # Dashboard specific figures
├── Dockerfile         # Container definition
└── docker-compose.yml # Orchestration (DB, Pipeline, Dashboard)
```

## ⚙️ How to Run / Cómo Ejecutar

For detailed commands, please check the [DOCKER_GUIDE.md](./DOCKER_GUIDE.md).

1. **Clone the repo.**
2. **Execute everything:**
   ```bash
   docker-compose up --build
   ```
   - **Service `db`**: Starts the database.
   - **Service `app`**: Runs the ML Pipeline automatically and then launches a Dashboard at [http://localhost:8502](http://localhost:8502).
   - **Service `streamlit`**: Dedicated Dashboard available instantly at [http://localhost:8501](http://localhost:8501).

> [!TIP]
> Use **localhost** or **127.0.0.1** in your browser. Do not use 0.0.0.0.

3. **Check results:**
   - Visual results in the Dashboard.
   - Files in `reports/`, `models/` and `db/`.

---

## 📚 References & Credits / Referencias y Créditos

### 🔗 Documentation / Documentación
- **Python**: [Official Documentation](https://docs.python.org/3/)
- **Pandas**: [API Reference](https://pandas.pydata.org/docs/)
- **Scikit-Learn**: [User Guide](https://scikit-learn.org/stable/documentation.html)
- **Matplotlib**: [Usage Guide](https://matplotlib.org/stable/contents.html)
- **Seaborn**: [Statistical Visualization](https://seaborn.pydata.org/)
- **Streamlit**: [Cloud & Library Docs](https://docs.streamlit.io/)
- **Docker**: [Containerization Docs](https://docs.docker.com/)
- **PostgreSQL**: [SQL & DB Docs](https://www.postgresql.org/docs/)

### 📂 Base Repositories / Repositorios Base
- [Markdown Repo](https://github.com/drklis/Learning-Markdown) - Used for [Markdown]
- [[Streamlit Repo](https://github.com/siddhardhan23/deploy-streamlit-app-as-docker-container)] - Inspiration for [Streamlit part]

- [[Streamlit Repo 2](https://github.com/siddhardhan23/no-code-ml-mpodel-training-app)] - Also for [Streamlit part]
