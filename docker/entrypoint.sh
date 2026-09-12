#!/usr/bin/env bash
# ── AuK Gradio entrypoint ────────────────────────────────────────────
# 1. Source .env if present (LLM_API_KEY, LLM_BASE_URL, …).
# 2. Auto-download weights when ckpts/ looks empty.
# 3. Export DEVICE hint so the app can adapt at runtime.
# 4. Exec the CMD (defaults to auk-gradio).
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Load environment variables from .env ──────────────────────────────
if [ -f /app/.env ]; then
    echo "[entrypoint] Loading /app/.env"
    set -a
    # shellcheck disable=SC1091
    source /app/.env
    set +a
fi

# ── Auto-download weights on first run ────────────────────────────────
# Skip if the user bind-mounted a populated ckpts/ volume.
AUK_BASE="/app/ckpts/AuK/auk_base.safetensors"

if [ "${AUTO_DOWNLOAD_WEIGHTS:-true}" = "true" ] && [ ! -f "${AUK_BASE}" ]; then
    echo "[entrypoint] No AuK weights found – starting auto-download …"
    python /app/docker/download_weights.py
fi

# ── DEVICE auto-detect hint for the application ───────────────────────
# If DEVICE was baked in at build time, re-export it so the app can
# branch on it.  At runtime you can override via environment.
export DEVICE="${DEVICE:-cpu}"

echo "[entrypoint] DEVICE=${DEVICE}"

# ── Launch the application ────────────────────────────────────────────
exec "$@"
