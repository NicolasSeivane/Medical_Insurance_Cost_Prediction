# Guía de Comandos Docker - ML Challenge

Este documento contiene los comandos esenciales para administrar el entorno de contenedores y ejecutar el pipeline de Machine Learning.

## 1. Pipeline Completo (Recomendado)

Para levantar la base de datos, entrenar el modelo, realizar el scoring y generar el reporte final:

```bash
# Construir y levantar todo el sistema (primera vez o cambios en Dockerfile/requirements)
docker-compose up --build

# Levantar normalmente (si ya está construido)
docker-compose up
```

---

## 2. Ejecutar Scripts Individuales

Si el contenedor ya está corriendo (vía `docker-compose up`), puedes ejecutar scripts específicos dentro del servicio `app`:

```bash
# Ejecutar solo entrenamiento
docker-compose exec app python training.py

# Ejecutar solo scoring y reportes PNG
docker-compose exec app python scoring.py

# Generar el reporte PDF (LaTeX) explicitamente
docker-compose exec app python ../reports/generate_report.py

# Acceder a la consola interactiva (bash) del contenedor
docker-compose exec app bash
```

---

## 3. Administración y Mantenimiento

```bash
# Apagar contenedores y limpiar recursos
docker-compose down

# Limpiar volúmenes (borra la base de datos de Postgres)
docker-compose down -v

# Ver logs del contenedor en tiempo real
docker-compose logs -f app
```

---

## 4. Estructura de Salida
Los resultados aparecerán automáticamente en tu máquina en:
- `db/schema.sql`: Esquema generado.
- `reports/`: Imágenes PNG de resultados.
- `reports/outputs/`: Reporte PDF final.
- `models/`: Archivos .pkl del modelo entrenado.
