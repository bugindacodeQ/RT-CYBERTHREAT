FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p models data/raw data/processed

# Build-time training with output
RUN echo "Generating synthetic data..." && python src/data_generation.py && \
    echo "Preprocessing data..." && python src/preprocess.py && \
    echo "Training model..." && python src/train.py && \
    echo "Model training completed successfully!"

EXPOSE 8000

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]