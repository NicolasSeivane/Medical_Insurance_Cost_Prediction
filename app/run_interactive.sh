#!/bin/bash
# run_interactive.sh

set -e

# Esperar a que la base de datos esté lista
./wait-for-postgres.sh db 5432

# Ejecutar el pipeline normal
./run_pipeline.sh

echo ""
echo "----------------------------------------------------------"
echo "✅ Pipeline ML completado exitosamente."
echo "🚀 Iniciando dashboard de Streamlit..."
echo "----------------------------------------------------------"
echo ""

# Iniciar Streamlit (sin preguntar)
# El puerto se puede configurar vía comandos o env var, pero acá lo dejamos por defecto
# para que el docker-compose lo mapee.
streamlit run ../streamlit/streamlit_app.py --server.port 8501 --server.address 0.0.0.0
