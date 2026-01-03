#!/bin/bash
# run_pipeline.sh

set -e

# NS: Wait until db is ready
./wait-for-postgres.sh db 5432

echo "🚀 Iniciando pipeline de ML..."


echo "📦 Paso 1: Configurando base de datos..."
python create_database.py

echo "🧠 Paso 2: Ejecutando entrenamiento..."
python training.py

echo "📊 Paso 3: Ejecutando scoring..."
python scoring.py

echo "📄 Paso 4: Generando reporte LaTeX..."
python ../reports/generate_report.py

echo "✅ Pipeline completado. Resultados en carpeta 'reports'."
