#!/bin/bash
# run_pipeline.sh

set -e

# Esperar a que la base de datos esté lista
./wait-for-postgres.sh db 5432

echo "🚀 Iniciando pipeline de ML..."

# create_database.py ya no se ejecuta al ser importado gracias al bloque __main__
# Lo corremos una vez explícitamente para asegurar que la DB esté lista y con datos.
echo "📦 Paso 1: Configurando base de datos..."
python create_database.py

echo "🧠 Paso 2: Ejecutando entrenamiento..."
python training.py

echo "📊 Paso 3: Ejecutando scoring..."
python scoring.py

echo "📄 Paso 4: Generando reporte LaTeX..."
python ../reports/generate_report.py

echo "✅ Pipeline completado. Resultados en carpeta 'reports'."
