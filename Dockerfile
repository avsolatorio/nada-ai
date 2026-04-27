# Build from the `nada-ai` repo root (`docker compose` uses context `.`).
# `ai4data` is installed via `[tool.uv.sources]` git pin in `pyproject.toml` / `uv.lock`.

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    ca-certificates \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv

WORKDIR /workspace

COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src

ENV UV_COMPILE_BYTECODE=1
RUN uv sync --frozen --no-dev --extra local

ENV PATH="/workspace/.venv/bin:$PATH"
ENV AI4DATA_DISCOVERY_DATA_PATH=/workspace/data/nada-discovery

RUN mkdir -p /workspace/data/nada-discovery

CMD ["uvicorn", "nada_ai.app.main:app", "--host", "0.0.0.0", "--port", "8020"]
