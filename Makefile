PYTHON      = python
DATA        = titanic.csv
MODEL       = model.pkl
MLFLOW_DB   = sqlite:///mlflow.db
MLFLOW_PORT = 5000
API_PORT    = 8000

# ── Docker ────────────────────────────────────────────────────
DOCKER_USER  = omaima123456789
IMAGE_NAME   = oumaima_bouhlel_mlops
TAG          = latest
FULL_IMAGE   = $(DOCKER_USER)/$(IMAGE_NAME):$(TAG)

.PHONY: all install format lint security prepare train evaluate test api mlflow clean docker-build docker-run docker-push docker-stop

# ── Pipeline complet CI/CD ────────────────────────────────────
all: install format lint security prepare train evaluate test
	@echo ========================================
	@echo  PIPELINE COMPLET TERMINE AVEC SUCCES
	@echo ========================================

# ── 1. Installation ───────────────────────────────────────────
install:
	pip install -r requirements.txt

# ── 2. Formatage ──────────────────────────────────────────────
format:
	black model_pipeline.py main.py app.py

# ── 3. Qualité du code ────────────────────────────────────────
lint:
	flake8 model_pipeline.py main.py app.py --max-line-length=100 --exclude=venv

# ── 4. Sécurité ───────────────────────────────────────────────
security:
	bandit -r model_pipeline.py main.py app.py -ll -x venv

# ── 5. Préparation des données ────────────────────────────────
prepare:
	$(PYTHON) main.py --step prepare --data $(DATA)

# ── 6. Entraînement ───────────────────────────────────────────
train:
	$(PYTHON) main.py --step train --data $(DATA) --model $(MODEL)

# ── 7. Évaluation ─────────────────────────────────────────────
evaluate:
	$(PYTHON) main.py --step evaluate --data $(DATA) --model $(MODEL)

# ── 8. Tests ──────────────────────────────────────────────────
test:
	pytest tests/ -v

# ── 9. API FastAPI ────────────────────────────────────────────
api:
	uvicorn app:app --reload --host 0.0.0.0 --port $(API_PORT)

# ── 10. MLflow UI avec SQLite ─────────────────────────────────
mlflow:
	mlflow ui --backend-store-uri $(MLFLOW_DB) --host 0.0.0.0 --port $(MLFLOW_PORT)

# ── 11. Docker : construire l'image ───────────────────────────
docker-build:
	@echo --- Construction de l image Docker ---
	docker build -t $(FULL_IMAGE) .
	docker images

# ── 12. Docker : lancer le conteneur ─────────────────────────
docker-run:
	@echo --- Lancement du conteneur Docker ---
	docker run -d --name mlops-api -p $(API_PORT):8000 $(FULL_IMAGE)
	@echo API disponible sur http://localhost:$(API_PORT)

# ── 13. Docker : arrêter le conteneur ────────────────────────
docker-stop:
	-docker stop mlops-api
	-docker rm mlops-api

# ── 14. Docker : push sur Docker Hub ─────────────────────────
docker-push:
	@echo --- Push de l image sur Docker Hub ---
	docker login
	docker push $(FULL_IMAGE)

# ── 15. Nettoyage ─────────────────────────────────────────────
clean:
	-del /Q $(MODEL) scaler.pkl label_encoder.pkl 2>nul
	-rmdir /s /q __pycache__ 2>nul