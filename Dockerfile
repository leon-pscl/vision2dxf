FROM python:3.11-slim

WORKDIR /app

# Dependencies go in a venv, so they never mix with the image's system python
# and the layout matches a developer's local env.
ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/app/src:/app \
    PYTHONUNBUFFERED=1
RUN python -m venv "$VIRTUAL_ENV"

# The wheels here are large (torch, opencv) and a slow link makes pip give up
# mid-download; these two flags are what make the build survive it.
ENV PIP_RETRIES=10 PIP_TIMEOUT=120

# CPU wheels only: ultralytics would otherwise pull the CUDA build of torch
# (~2.5 GB) from PyPI, and the station has no GPU.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    torch torchvision

# Native libraries the python packages bind to: the first three for the
# opencv-python GUI build (ultralytics depends on it by name, so the headless
# package cannot replace it), libcairo2 for cairosvg, which the bench's
# pygarment import needs.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libxcb1 libcairo2 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-bench.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# The M4b bench too, so everything runs in here. It costs little next to torch,
# and the GarmentCode clone it reads is bind-mounted from the host.
RUN pip install --no-cache-dir -r requirements-bench.txt

# The YOLO26 weights are git-ignored, so a fresh clone has none and every capture
# used to fail with "model weights missing". The entrypoint fetches whatever
# models/ is missing on start-up; nothing to bake into the image.
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN sed -i 's/\r$//' /usr/local/bin/entrypoint.sh && chmod +x /usr/local/bin/entrypoint.sh

COPY . .

EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["uvicorn", "bodymeasure.api:app", "--host", "0.0.0.0", "--port", "8080"]
