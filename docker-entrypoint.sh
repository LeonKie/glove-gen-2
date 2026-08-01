#!/bin/sh
# Prepare the store, then drop to an unprivileged user.
#
# Mounted volumes (Fly volumes, fresh docker volumes, bind mounts) arrive owned
# by root, so the container has to start as root to fix that up and only then
# hand off to the app user.
set -eu

APP_UID=10001
APP_GID=10001
STORE="${GLOVEGEN_STORE:-/data/store}"

if [ "$(id -u)" = "0" ]; then
    mkdir -p "$STORE"
    # Recursive only when the ownership is actually wrong — first boot, or a
    # volume carried over from a root-run container. Steady state costs nothing.
    if [ "$(stat -c %u "$STORE")" != "$APP_UID" ]; then
        chown -R "$APP_UID:$APP_GID" "$STORE"
    fi
    exec setpriv --reuid="$APP_UID" --regid="$APP_GID" --init-groups "$@"
fi

# Started with an explicit --user: no privileges to fix anything up, and the CLI
# does not touch the store at all, so a store we cannot create is not fatal here.
# The server would fail loudly on its own if it needed one.
mkdir -p "$STORE" 2>/dev/null || true
exec "$@"
