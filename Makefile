PYTHON = python
DATA   = titanic.csv
MODEL  = model.pkl

.PHONY: all install format lint security prepare train evaluate test api mlflow clean

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

# ── 4. Sécurité — CORRIGÉ : suppression de "|| true" (Windows incompatible)
#    Remplacé par -ll qui ignore les avertissements mineurs
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

# ── 8. Tests — CORRIGÉ : vrai pytest au lieu de @echo "Tests OK"
test:
	pytest tests/ -v

# ── 9. API FastAPI ────────────────────────────────────────────
api:
	uvicorn app:app --reload --host 0.0.0.0 --port 8000

# ── 10. MLflow UI avec backend SQLite (Excellence Atelier 5) ──
mlflow:
	mlflow ui --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5000

# ── 11. Nettoyage (Windows) ───────────────────────────────────
clean:
	-del /Q $(MODEL) scaler.pkl label_encoder.pkl 2>nul
	-rmdir /s /q __pycache__ 2>nul