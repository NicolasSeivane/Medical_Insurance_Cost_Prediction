#!/bin/bash
# run_interactive.sh

set -e

# NS: Wait until the db is ready
./wait-for-postgres.sh db 5432

# NS: Run the main pipeline
./run_pipeline.sh

echo ""
echo "----------------------------------------------------------"
echo "✅ Pipeline ML completado exitosamente."
echo "🚀 Iniciando dashboard de Streamlit..."
echo "----------------------------------------------------------"
echo ""

# NS: Start streamlit without question, port can be changed here

streamlit run ../streamlit/streamlit_app.py --server.port 8501 --server.address 0.0.0.0
