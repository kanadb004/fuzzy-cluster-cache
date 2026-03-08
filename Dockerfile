# ──────────────────────────────────────────────────────────────────────────────
# Multi-stage Dockerfile for the fuzzy-cluster-cache semantic search service
# ──────────────────────────────────────────────────────────────────────────────
#
# Stage layout
# ─────────────
# Stage 1 (builder): install ALL Python dependencies into a virtual-env
#   isolated from the system Python.  This stage carries build tools (gcc,
#   pkg-config) that are NOT needed at runtime and would bloat the final image.
#
# Stage 2 (runtime): copy only the virtual-env and application code from the
#   builder.  Result is a lean image with no compiler toolchain.
#
# Layer caching strategy
# ──────────────────────
# Copy requirements.txt BEFORE copying the application source:
#
#   COPY requirements.txt .
#   RUN  pip install -r requirements.txt      ← cached until requirements change
#   COPY app/ .                               ← invalidates only this layer
#
# If only Python files change, Docker reuses the cached pip layer and the
# rebuild takes seconds instead of minutes.  This is the single most impactful
# Docker best-practice for ML services with heavy dependencies.
#
# Base image choice: python:3.10-slim
# ─────────────────────────────────────
# • Debian-based, minimal OS packages (~150 MB vs ~900 MB for -full).
# • python:3.10-slim-bullseye ships glibc, which FAISS wheels require.
# • Alpine would need manual FAISS compilation (C++ build chain); skip it.

# ─── Stage 1: dependency builder ────────────────────────────────────────────
FROM python:3.10-slim AS builder

# Install OS-level build dependencies that Python packages compile against.
# --no-install-recommends keeps the layer lean; packages are removed after
# build artefacts are copied to the runtime stage.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create and activate a virtual environment so the runtime stage can copy a
# self-contained site-packages directory without system Python interference.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build

# ── critical layer-cache step ────────────────────────────────────────────────
# Copy only the dependency manifest first.  Docker caches this layer until
# requirements.txt changes.  Heavy wheel downloads (torch, faiss, etc.) only
# re-run when you add/upgrade a package, not when you edit app/ files.
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


# ─── Stage 2: runtime image ─────────────────────────────────────────────────
FROM python:3.10-slim AS runtime

# libgomp1 is required by the FAISS C++ shared library at runtime.
# It is NOT a build tool, so it belongs in the runtime stage.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the isolated virtual environment from the builder stage.
# This is the entire pip ecosystem, ~1.5 GB for ML packages, copied as a
# single Docker layer without re-installing.
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create a non-root user.  Running as root inside a container is a security
# anti-pattern; if the process is compromised, it has root on the host via
# mounted volumes.
RUN useradd --create-home --shell /bin/bash appuser
WORKDIR /home/appuser/app
USER appuser

# Copy application source after dependencies so edits to Python files do not
# invalidate the (expensive) pip install layer above.
COPY --chown=appuser:appuser app/ ./app/

# Expose the Uvicorn port.  Not strictly required (docker run -p still works
# without it), but documents the contract and is required by `docker compose`.
EXPOSE 8000

# Health-check:  gives orchestrators (Kubernetes, ECS) a signal that the
# service is ready.  The 60-second start-period covers the ML startup time
# (embedding model download + GMM fitting).
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Run Uvicorn in production mode:
#   --host 0.0.0.0  – bind to all interfaces (required inside Docker).
#   --workers 1     – single worker; the in-memory cache is not shared across
#                     workers, so multiple workers would each maintain an
#                     independent cache, defeating the purpose.
#   --log-level info – structured access logs to stdout for collection by
#                      container log drivers (CloudWatch, Datadog, etc.).
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", \
    "--workers", "1", "--log-level", "info"]
