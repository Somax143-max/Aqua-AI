# AquaProtect-AI: Multi-target Container
# National Institute of Ocean Technology (NIOT) / MoES
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    BACKEND_HOST=0.0.0.0

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install unified requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy complete project
COPY . .

# Ensure working directories
RUN mkdir -p models_output reports_output digital_twin_data feedback_data benchmark_reports

EXPOSE 8000 8501

# Default start backend (can be overridden with CMD ["python", "frontend/run_frontend.py"])
CMD ["python", "backend/run_backend.py"]
