# 3D Pose Analyzer UI

This Vite-powered React client makes it easy to call the FastAPI `/analyze` endpoint defined in `api.py`. Paste a JSON payload (or upload one of the samples from `pose_examples.json`), toggle the transpose flag if needed, and review the prediction, confidence, and performance metrics the backend returns.

## Quick start

```bash
# from the repo root
cd frontend
npm install      # first run only
# optionally create .env and set VITE_API_BASE_URL if needed
npm run dev
```

If you set `VITE_API_BASE_URL`, populate `frontend/.env` with e.g.:

```
VITE_API_BASE_URL=http://localhost:8000
```

The dev server runs at `http://localhost:5173`. Make sure your FastAPI server is running and accessible from the browser (e.g., `uvicorn api:app --reload --host 0.0.0.0 --port 8000`).

## Production build

```bash
npm run build
npm run preview   # optional smoke test
```

You can host the contents of `frontend/dist` behind any static-file server or CDN. Remember to set `VITE_API_BASE_URL` during the build if the backend lives elsewhere.
