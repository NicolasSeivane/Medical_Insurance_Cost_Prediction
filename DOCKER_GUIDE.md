# Guía de Comandos Docker - ML Challenge

Este documento contiene los comandos esenciales para administrar el entorno de contenedores y ejecutar el pipeline de Machine Learning.

## 1. Arquitectura de 3 Servicios

El sistema está diseñado para que todo funcione automáticamente:

- **`db`**: Base de datos PostgreSQL.
- **`app`**: Ejecuta el pipeline de entrenamiento/scoring y, al terminar, inicia automáticamente un dashboard en el puerto **8502**.
- **`streamlit`**: Un dashboard dedicado que está disponible inmediatamente en el puerto **8501**.

```bash
# Levantar todo (DB, Pipeline y Dashboard)
docker-compose up --build
```

Una vez que termine el pipeline en el contenedor `app`, tendrás dos formas de ver los resultados:
1. **[http://localhost:8501](http://localhost:8501)**: Dashboard principal.
2. **[http://localhost:8502](http://localhost:8502)**: Dashboard automático del pipeline.

**Nota importante**: Usa siempre `localhost`. La dirección `0.0.0.0` que muestra Streamlit es interna del contenedor y no funcionará en el navegador de Windows.

---

## 2. Comandos Útiles

```bash
# Ver logs de todo para monitorear el progreso del pipeline
docker-compose logs -f

# Si solo quieres ver el progreso del entrenamiento:
docker-compose logs -f app

# Apagar y limpiar
docker-compose down
```

---

## 3. Estructura de Salida
Los resultados se guardan permanentemente en:
- `reports/figures/`: Imágenes del pipeline automático.
- `reports/streamlit_figures/`: Imágenes generadas desde el Dashboard.
- `reports/outputs/`: Reportes PDF.
- `models/`: Modelos entrenados (.pkl).
