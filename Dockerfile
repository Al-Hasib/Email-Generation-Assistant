FROM python:3.11-slim AS base

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p results

FROM base AS evaluation

CMD ["python", "run_evaluation.py"]
