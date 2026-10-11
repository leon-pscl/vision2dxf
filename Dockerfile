FROM python:3.13-slim

# OpenPattern is only on GitHub, so git is needed to install it.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLBACKEND=Agg

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY pattern_drafter/ ./pattern_drafter/
COPY web.py index.html ./

RUN useradd --create-home --uid 10001 drafter
USER drafter

EXPOSE 8000
CMD ["python", "web.py", "--host", "0.0.0.0", "--port", "8000"]