# Image de base Python
FROM python:3.10-slim

# Répertoire de travail dans le conteneur
WORKDIR /app

# Copier les fichiers du projet
COPY requirements.txt .
COPY model_pipeline.py .
COPY main.py .
COPY app.py .
COPY titanic.csv .

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Entraîner le modèle au build (génère model.pkl, scaler.pkl, label_encoder.pkl)
RUN python main.py --step train --data titanic.csv --model model.pkl

# Exposer le port de l'API
EXPOSE 8000

# Lancer l'API FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]