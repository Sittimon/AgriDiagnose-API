Corn Disease Prediction API
===========================

This repository contains a FastAPI application that performs inference on corn leaf images to predict diseases using a PyTorch model.

Files of interest
-----------------
- `api.py` - Main FastAPI application.
- `best_jitter.pth` / `best_cosine.pth` - Pretrained PyTorch model(s) used for inference. (download is available at https://disk.yandex.com/d/BYixpO12d4KH0w)
- `requirements.txt` - Python dependencies (trimmed to essentials for Docker builds).
- `Dockerfile` - Dockerfile to build the API container.
- `.env` - Optional environment file for sensitive keys (not included in repo).

Quickstart — Docker (recommended)
---------------------------------
Build the Docker image from the project root (Windows PowerShell):

```powershell
cd "C:\Users\sitth\Downloads\api_jj\Archive"
docker build -t corn-model-api:latest .
```

Run a container (map port 8010):

```powershell
docker run --rm -p 8010:8010 --env-file .env corn-model-api:latest
```

After the container starts, the API should be available at `http://localhost:8010`.

Endpoints
---------
- `GET /` - Basic info about the API.
- `GET /health` - Health check (returns model_loaded and device info).
- `POST /predict` - Predict disease from an uploaded image. Multipart form with field `file`.
- `POST /predict_with_advice` - Predict + call GPT for advice (if configured).
- `POST /predict_cv` - Predict via Azure Custom Vision pipeline (if configured).

Example — curl
---------------
```bash
curl -X POST "http://localhost:8010/predict" -F "file=@/path/to/leaf.jpg"
```

Notes & Troubleshooting
-----------------------
- If `docker build` fails during `pip install`, check your `requirements.txt`. The project supplies a trimmed `requirements.txt` with only essential packages (FastAPI, uvicorn, torch, torchvision, pillow, requests, numpy, pydantic, python-multipart).
- If your model filename differs (e.g. `best_cosine.pth` vs `best_jitter.pth`), update `api.py` or replace the file copied into the image.
- GPU support: The provided Dockerfile uses the `python:3.11-slim` base image and CPU PyTorch. For GPU support, see the official PyTorch Docker images and replace the base image accordingly.
- Secrets: Do not hard-code API keys in `api.py`. Use `.env` and environment variables.

Development (local without Docker)
----------------------------------
1. Create and activate a virtual environment.
2. Install dependencies: `pip install -r requirements.txt`.
3. Run:

```powershell
uvicorn api:app --host 0.0.0.0 --port 8010 --reload
```

Security & Production Notes
---------------------------
- Run the app behind a production server (e.g., Gunicorn + Uvicorn workers) for production throughput.
- Restrict CORS origins instead of `*` in production.
- Monitor memory and CPU usage — model inference can be memory-intensive depending on `torch` and `torchvision` versions.

Contact
-------
If you need further help building or deploying, share the full `docker build` output and I will help debug specific failures.
