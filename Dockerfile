FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
COPY model_pipeline.py .
COPY main.py .
COPY app.py .
COPY titanic.csv .

RUN pip install --no-cache-dir -r requirements.txt

RUN python main.py --step train --data titanic.csv --model model.pkl

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
