# 🚀 AquaProtect-AI: Deployment Guide (Render & Vercel)

This repository is fully configured for automated cloud deployment on **Render** and **Vercel**, as well as Docker and standard container platforms.

---

## 🌊 Architecture Overview

| Component | Platform | Tech Stack | Entry Point |
|---|---|---|---|
| **Backend REST API** | Render / Vercel / Docker | FastAPI, PyTorch, ONNX, OpenCV-Headless | `backend/api.py` or `api/index.py` |
| **Command Center Dashboard** | Render / Docker / Streamlit Cloud | Streamlit, Plotly, Folium, Leaflet | `frontend/app.py` |

---

## 1. 🟣 Deploying to Render (Recommended)

Render can deploy both the **FastAPI Backend** and the **Streamlit Frontend** simultaneously using the included [render.yaml](render.yaml) Blueprint.

### Option A: 1-Click Blueprint (Fastest)
1. Push your repository to GitHub: `https://github.com/Somax143-max/Aqua-AI`.
2. Log into [Render Dashboard](https://dashboard.render.com/).
3. Click **New +** → **Blueprint**.
4. Connect your GitHub repository `Aqua-AI`.
5. Render will automatically read `render.yaml` and configure two Web Services:
   - **`aqua-ai-backend`**: FastAPI REST microservice (`/docs`, `/health`, `/api/*`).
   - **`aqua-ai-frontend`**: Streamlit Command Center UI (automatically wired to the backend URL via `BACKEND_URL`).
6. Click **Apply**.

---

### Option B: Manual Service Creation on Render

#### 1. Backend Web Service:
- **Environment**: `Python 3`
- **Root Directory**: `backend`
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `python -m uvicorn api:app --host 0.0.0.0 --port $PORT`
- **Health Check Path**: `/health`
- **Environment Variables**:
  - `PYTHON_VERSION`: `3.11.9`
  - `BACKEND_HOST`: `0.0.0.0`

#### 2. Frontend Web Service:
- **Environment**: `Python 3`
- **Root Directory**: `frontend`
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `python -m streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true --server.enableCORS false --server.enableXsrfProtection false`
- **Health Check Path**: `/_stcore/health`
- **Environment Variables**:
  - `PYTHON_VERSION`: `3.11.9`
  - `BACKEND_URL`: `https://<your-backend-render-subdomain>.onrender.com`

---

## 2. ▲ Deploying to Vercel

The FastAPI backend can be deployed directly as a Python Serverless Function on Vercel using the configured [vercel.json](vercel.json) and [api/index.py](api/index.py).

### Quick Steps:
1. Install Vercel CLI (or import via the [Vercel Web Dashboard](https://vercel.com/new)):
   ```bash
   npm i -g vercel
   ```
2. In the project root directory, run:
   ```bash
   vercel
   ```
   Or for production:
   ```bash
   vercel --prod
   ```
3. Vercel automatically routes all HTTP traffic to `api/index.py`.
4. Your endpoints will be available at:
   - Interactive Swagger API: `https://<your-project>.vercel.app/docs`
   - ReDoc Documentation: `https://<your-project>.vercel.app/redoc`
   - Health Check: `https://<your-project>.vercel.app/health`
   - Ingestion & Missions: `https://<your-project>.vercel.app/api/missions`

---

## 3. 🐳 Deploying with Docker

You can also run both services locally or on any cloud container host using Docker Compose:

```bash
# Build and run both backend and frontend containers
docker-compose up --build -d
```

- **Backend REST Service**: `http://localhost:8000/docs`
- **Streamlit Frontend**: `http://localhost:8501`

---

## 🔧 Environment Variables Reference

| Variable | Default | Description |
|---|---|---|
| `PORT` | Dynamic (`$PORT`) | Allocated automatically by Render / Cloud Hosts |
| `BACKEND_HOST` | `0.0.0.0` | Binding IP address for the FastAPI backend |
| `BACKEND_URL` | Auto-detected | Backend REST endpoint for the Frontend Client |
| `PYTHON_VERSION` | `3.11.9` | Preferred Python runtime version |
