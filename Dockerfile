# Dockerfile (FastAPI + XGBoost + Scikit‑Learn + Uvicorn)
# -----------------------------
# 1. Base image (Python 3.10)
# -----------------------------
FROM python:3.10-slim

# -----------------------------
# 2. Set working directory
# -----------------------------
WORKDIR /app

# -----------------------------
# 3. Install system dependencies
# -----------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------
# 4. Copy requirements first (for caching)
# -----------------------------
COPY requirements.txt .

# -----------------------------
# 5. Install Python dependencies
# -----------------------------
RUN pip install --no-cache-dir -r requirements.txt

# -----------------------------
# 6. Copy application code
# -----------------------------
COPY src/ ./src/
COPY artifacts/ ./artifacts/

# -----------------------------
# 7. Expose API port
# -----------------------------
EXPOSE 8000

# -----------------------------
# 8. Start FastAPI app
# -----------------------------
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8080"]
