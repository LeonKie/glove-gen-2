# syntax=docker/dockerfile:1

# --------------------------------------------------------------------------
# builder — resolve the dependency tree into a self-contained venv
# --------------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

RUN python -m venv "$VIRTUAL_ENV"

WORKDIR /app

# Dependencies first, read straight out of pyproject.toml so there is no second
# list to keep in sync. Source is copied *after* this layer: editing geometry
# code then costs a few seconds instead of re-resolving numpy/scipy/manifold3d.
COPY pyproject.toml README.md ./
RUN python - <<'PY' > /tmp/requirements.txt
import pathlib, tomllib
p = tomllib.loads(pathlib.Path("pyproject.toml").read_text())["project"]
print("\n".join(p["dependencies"] + p["optional-dependencies"]["fast"]))
PY

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /tmp/requirements.txt

COPY glovegen/ ./glovegen/
COPY server/ ./server/

# Editable, deliberately: server/static/ is not a Python package, so a regular
# install would leave the viewer's assets behind. This keeps one copy of the
# tree at /app and still puts the `glovegen` CLI on PATH.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-deps -e .

# --------------------------------------------------------------------------
# runtime
# --------------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    GLOVEGEN_STORE=/data/store \
    PORT=8111

RUN groupadd --gid 10001 glovegen \
 && useradd --uid 10001 --gid 10001 --home-dir /app --no-create-home glovegen

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /app /app
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

WORKDIR /app
EXPOSE 8111

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import os,urllib.request;urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','8111')+'/api/status',timeout=4)"]

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
# One uvicorn process, never --workers: the app owns a process pool and an
# in-process mesh cache, and a second copy of either would double the 4 GB peak.
CMD ["sh", "-c", "exec uvicorn server.app:app --host 0.0.0.0 --port ${PORT:-8111} --forwarded-allow-ips '*'"]
