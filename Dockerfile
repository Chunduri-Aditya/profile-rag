FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 FASTEMBED_CACHE_PATH=/app/.fastembed
WORKDIR /app
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
# pystemmer (via llama-index-retrievers-bm25) compiles from source.
RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY . .
RUN uv sync --frozen --no-dev
# Build the index and warm the ONNX model caches into the image.
RUN uv run python -m profile_rag.build && uv run python -c "from profile_rag.retrieve import warm; print(warm())"
EXPOSE 8000
CMD ["uv", "run", "uvicorn", "profile_rag.api:app", "--host", "0.0.0.0", "--port", "8000"]
