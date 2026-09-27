# syntax=docker/dockerfile:1

# aqdas-rag — the Kitab-i-Aqdas MCP server, containerised.
#
# Shape: inbound-serving (FastMCP HTTP transport), stateless and strictly
# read-only — no volume, no database, nothing to back up.
#
# The corpus is NOT in git: `data/` is gitignored, so it is derived at build
# time — fetch the Reference Library fragments, parse them into citable
# records, and warm the embedding matrix. The runtime image therefore carries
# the corpus, the vectors and the ONNX model, so its first query needs no
# network and does not re-embed the whole book.

# ---- Build stage ----
FROM python:3.12.14-slim-bookworm AS build

# Pinned to match uv.lock (revision 3); a newer uv may want to rewrite the
# lock, which --locked would then reject.
COPY --from=ghcr.io/astral-sh/uv:0.9.28 /uv /bin/uv

ENV UV_CACHE_DIR=/opt/uv-cache \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    FASTEMBED_CACHE_PATH=/app/models

WORKDIR /app

# Dependencies before source, so this layer survives every code edit.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/opt/uv-cache \
    uv sync --locked --no-dev --no-install-project

# pyproject declares `readme` and `license-files`, so both must exist before
# the project itself can be built.
COPY README.md LICENSE ./
COPY src ./src
RUN --mount=type=cache,target=/opt/uv-cache \
    uv sync --locked --no-dev

# Derive the corpus (one network call to bahai.org) and warm the embedding
# matrix, so the running container neither fetches text nor embeds 487 records
# on its first query.
RUN /app/.venv/bin/python -m aqdas_rag.fetch \
 && /app/.venv/bin/python -m aqdas_rag.parse \
 && /app/.venv/bin/python -c "from aqdas_rag.retrieve import Corpus; from aqdas_rag.hybrid import HybridRetriever; HybridRetriever(Corpus())"

# ---- Runtime stage ----
FROM python:3.12.14-slim-bookworm AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    FASTEMBED_CACHE_PATH=/app/models \
    AQDAS_TRANSPORT=http \
    AQDAS_HOST=0.0.0.0 \
    AQDAS_PORT=8300

WORKDIR /app

RUN groupadd --system --gid 10001 aqdas \
 && useradd --system --uid 10001 --gid aqdas --no-create-home --shell /usr/sbin/nologin aqdas

# The venv, then the source it points at. uv installs the project editable, so
# /app/src must sit at the same path it did at build time. data/ and models/
# are the build-derived corpus, vectors and ONNX model.
COPY --from=build --chown=aqdas:aqdas /app/.venv /app/.venv
COPY --from=build --chown=aqdas:aqdas /app/data ./data
COPY --from=build --chown=aqdas:aqdas /app/models ./models
COPY --chown=aqdas:aqdas src ./src

USER aqdas

EXPOSE 8300

# curl is not in the slim image, and a HEALTHCHECK invoking a missing binary
# reports unhealthy forever. FastMCP serves GET /health.
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD ["python", "-c", "import os,sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('AQDAS_PORT','8300')+'/health', timeout=2).status==200 else 1)"]

ENTRYPOINT ["/app/.venv/bin/python", "-m", "aqdas_rag.server"]
