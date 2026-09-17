FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    GIT_PYTHON_REFRESH=quiet \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY mlops_demo ./mlops_demo
RUN uv sync --frozen --no-dev

EXPOSE 8000

CMD ["python", "-m", "mlops_demo.api"]
