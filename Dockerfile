# ollamaq daemon in a container.
#
# The daemon drains the file-backed job queue and talks to an Ollama server that
# runs OUTSIDE this container (on the host, or another container). It reaches
# Ollama via OLLAMAQ_BASE_URL — see docker-compose.yml for the host wiring.
#
# Build:  docker build -t ollamaq:latest .
# Run:    docker compose up          (preferred — mounts queue/results/config)
#
# This image does NOT bundle Ollama or any models; it is a lightweight worker.

FROM python:3.13-slim AS build
WORKDIR /src
# Build the wheel from source so the runtime image installs a real artifact
# (identical to `pip install ollamaq`) rather than an editable tree.
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir build && python -m build --wheel --outdir /dist

FROM python:3.13-slim AS runtime
# Non-root: the daemon only needs to read config and read/write the mounted
# queue/results/cache volumes, all owned by this user.
RUN useradd --create-home --uid 10001 ollamaq
WORKDIR /home/ollamaq

COPY --from=build /dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && rm /tmp/*.whl

# Queue/results/cache live on mounted volumes (see compose); config is mounted
# read-only. OLLAMAQ_HOME anchors the relative queue/ results/ cache/ paths.
ENV OLLAMAQ_HOME=/data \
    OLLAMAQ_CONFIG=/config/models.yaml \
    OLLAMAQ_QUEUE=/data/queue \
    OLLAMAQ_RESULTS=/data/results \
    OLLAMAQ_CACHE=/data/cache \
    PYTHONUNBUFFERED=1
RUN mkdir -p /data/queue /data/results /data/cache && chown -R ollamaq:ollamaq /data

USER ollamaq

# A one-shot health probe: config parses and Ollama is reachable. `doctor`
# exits 0 only when healthy, which is exactly the signal Docker wants.
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD ollamaq doctor || exit 1

# Foreground daemon (run ONE per queue). Compose overrides --poll if needed.
ENTRYPOINT ["ollamaq", "daemon"]
CMD ["--queue", "/data/queue", "--results", "/data/results"]
