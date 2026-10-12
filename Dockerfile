# syntax=docker/dockerfile:1.7

# ---- stage 1: build the SPA -------------------------------------------------
FROM node:20-alpine AS ui
WORKDIR /build/frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci --no-audit --no-fund || npm install --no-audit --no-fund
COPY frontend/ ./
RUN npm run build          # vite outDir is ../backend/app/static -> /build/backend/app/static


# ---- stage 2: python app + runtime deps -------------------------------------
FROM python:3.12-slim AS app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /srv
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/app/ ./app/
COPY fixtures/ /srv/fixtures/
COPY assets/ /srv/assets/
# The full GarmageSet library is deliberately not copied: it is 2.9 GB and only needed
# for ad-hoc template selection. Mount it at /srv/templates to enable that.
COPY --from=ui /build/backend/app/static /srv/app/static


# ---- stage 3: test image ----------------------------------------------------
# docker build --target test -t base-block-transform:test . && docker run --rm base-block-transform:test
FROM app AS test
RUN pip install --no-cache-dir pytest==8.3.4 httpx==0.28.1
COPY backend/tests/ /srv/backend/tests/
COPY backend/pytest.ini /srv/backend/pytest.ini
COPY scripts/ /srv/scripts/
WORKDIR /srv/backend
CMD ["python", "-m", "pytest"]


# ---- stage 4: server image, the default target ------------------------------
FROM app AS runtime
RUN useradd --create-home --uid 10001 appuser && chown -R appuser /srv
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=4s --start-period=10s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz').status==200 else 1)"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
