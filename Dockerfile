# Imagen base pequeña con Python
FROM python:3.12-slim

# Directorio de trabajo dentro de la imagen
WORKDIR /app

# Copiar primero las dependencias para aprovechar el cache de capas
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el proyecto (app.py, model.joblib, etc.; ver .dockerignore)
COPY . .

# Puerto que usa Streamlit
EXPOSE 8501

# Proceso que se ejecuta al iniciar el contenedor
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
