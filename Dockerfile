# Usar una imagen de base
FROM python:3.11-slim

#Estabecer el directorio de trabajo en app
WORKDIR /app

RUN pip install pandas
RUN pip install -U sentence-transformers

COPY src/ ./src/

#Comando para ejecutar e iniciar el contendor
CMD ["python3", "src/main_students.py"]
#CMD ["python3", "/app/src/main_scripts.py"]

