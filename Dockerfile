FROM python:3.10-slim

EXPOSE 8501

WORKDIR /app

# Instalar netcat, sed y pdflatex
RUN apt-get update && apt-get install -y \
    netcat-openbsd \
    sed \
    texlive-latex-base \
    texlive-latex-extra \
    texlive-fonts-recommended \
    && rm -rf /var/lib/apt/lists/*

# Copia todo el contenido del proyecto
COPY . .

# Instalar dependencias
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r app/requirements.txt

# Cambiamos al directorio app para que las rutas relativas funcionen
WORKDIR /app/app

# Corregir fines de línea CRLF -> LF para los scripts de bash
RUN sed -i 's/\r$//' wait-for-postgres.sh && \
    sed -i 's/\r$//' run_pipeline.sh && \
    sed -i 's/\r$//' run_interactive.sh && \
    chmod +x wait-for-postgres.sh run_pipeline.sh run_interactive.sh

# Crear carpetas de salida para persistencia
RUN mkdir -p /app/reports/outputs /app/db /app/models && \
    chmod -R 777 /app/reports /app/db /app/models

# Comando por defecto
CMD ["./run_pipeline.sh"]