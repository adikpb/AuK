# ── AuK Gradio Docker Image (Hybrid Auto-Tier) ─────────────────────
# CPU by default, optional CUDA/ROCm via build args.
# Build:  docker compose build                       (CPU)
#         DEVICE=cuda docker compose --profile gpu build  (NVIDIA)
# Run:    docker compose up --build                  (CPU)
#         DEVICE=cuda docker compose --profile gpu up --build
# ─────────────────────────────────────────────────────────────────────
FROM python:3.10-slim

LABEL maintainer="AuK Team" \
      description="AuK Gradio demo – CPU/CUDA/ROCm hybrid build"

# ── Build-time knobs ────────────────────────────────────────────────
ARG DEVICE=cpu
ARG CUDA_VERSION_MM=124

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_SYSTEM_PYTHON=1 \
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
    DEVICE=${DEVICE}

# ── uv binary ───────────────────────────────────────────────────────
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# ── System packages (Debian slim: Python already present, no PPA) ────
# g++ + libfst-dev are required to build pynini (via WeTextProcessing in the gradio extra).
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        git \
        g++ \
        libfst-dev \
    && rm -rf /var/lib/apt/lists/*

# ── Conditional PyTorch install ─────────────────────────────────────
# CPU  (~200 MB wheel)  |  CUDA cu${CUDA_VERSION_MM}  |  ROCm (skip – install manually)
# Uses shell expansion inside RUN so the ARG is evaluated at build time.
RUN case "${DEVICE}" in \
      cuda) \
        uv pip install --system --no-cache \
          torch==2.7.0 torchaudio==2.7.0 torchvision==0.22.0 \
          --index-url "https://download.pytorch.org/whl/cu${CUDA_VERSION_MM}" \
        ;; \
      rocm) \
        echo "[dockerfile] ROCm PyTorch install is not bundled – install manually at runtime." ;; \
      *) \
        uv pip install --system --no-cache \
          torch==2.7.0 torchaudio==2.7.0 torchvision==0.22.0 \
          --index-url https://download.pytorch.org/whl/cpu \
        ;; \
    esac

# ── Application ─────────────────────────────────────────────────────
WORKDIR /app

# Copy build metadata first (layer cache for dependency install).
COPY pyproject.toml uv.lock README.md LICENSE .env.example ./
COPY src/ src/
COPY assets/demo-input-audio/ assets/demo-input-audio/

# Install AuK with the gradio extras + huggingface_hub for weight downloads.
RUN uv pip install --system --no-cache -e ".[gradio]" \
    && uv pip install --system --no-cache "huggingface_hub[cli]>=0.23"

# Prepare weight & output directories.
RUN mkdir -p ckpts outputs

# Copy Docker helper scripts.
COPY docker/entrypoint.sh docker/download_weights.py docker/
RUN chmod +x docker/entrypoint.sh

EXPOSE 7860

ENTRYPOINT ["/app/docker/entrypoint.sh"]
CMD ["auk-gradio", "--host", "0.0.0.0", "--port", "7860"]
