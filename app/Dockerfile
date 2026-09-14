# Imagen minima para correr la app de Gradio en Koyeb (o cualquier PaaS
# con soporte Docker). Build context = esta carpeta (app/).
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Koyeb inyecta $PORT en runtime; 8000 es el valor por default que asigna
# el asistente al crear el servicio. app.py ya lee $PORT via os.environ.
ENV PORT=8000
EXPOSE 8000

CMD ["python", "app.py"]
