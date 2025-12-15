FROM python:3.11-slim
WORKDIR /app

# copy only what is necessary
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy the app main only
COPY main.py .

ENV MODEL_DIR=/models
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# ============================
# File: `.dockerignore`
