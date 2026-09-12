# AuK Docker — One-Command Gradio Playground

Spin up the full AuK test UI without local setup.

## Prereqs

- Docker (Docker Desktop or equivalent)
- ~20GB disk (CUDA layers + ~12GB weights: AuK + Qwen2.5-Omni-3B, + Flash unless skipped)
- NVIDIA Container Toolkit – only needed for GPU builds

## Run — CPU (default)

Works everywhere, including macOS ARM (Apple Silicon) with no GPU passthrough.

```bash
cp .env.example .env   # optional: LLM keys for Prompt Enhancer
docker compose up --build
```

Open http://localhost:7860

## Run — NVIDIA GPU

Requires the NVIDIA Container Toolkit on the host.
See <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/>.

```bash
DEVICE=cuda docker compose --profile gpu up --build
```

The `gradio-cuda` service extends the base `gradio` build with a CUDA
PyTorch wheel and an NVIDIA device reservation.

## macOS / ARM note

macOS ARM (Apple Silicon) runs the **CPU** build by default.
There is no NVIDIA CUDA passthrough on macOS Docker; use the default
`docker compose up --build` command.

## Weights

Weights auto-download on first run into the `auk-ckpts` volume
(`tencent/AuK`, `Qwen/Qwen2.5-Omni-3B`, + `tencent/AuK-Flash`).
To skip Flash: `SKIP_AUK_FLASH=1 docker compose up --build`.
To use local weights: `AUTO_DOWNLOAD_WEIGHTS=false` with a bind-mount of `./ckpts`.

## Config

- Port: `AUK_PORT=7860` (default)
- Device: `DEVICE=cpu` (default) or `DEVICE=cuda`
- CUDA wheel tag: `CUDA_VERSION_MM=124` (default, matches CUDA 12.4)
- Volumes: `auk-ckpts` → `/app/ckpts`, `auk-outputs` → `/app/outputs`
- Env file: `.env` (auto-loaded by `docker/entrypoint.sh`)
- Image uses `uv` (`UV_SYSTEM_PYTHON=1`) on `python:3.10-slim` with
  conditional PyTorch (CPU wheel ~200 MB, or CUDA cu124 wheel).

## UI Coverage

22 tabs: zero-shot/instruct TTS, content replace/insert/delete, lyric edit,
pitch/speed/volume, emotion/timbre/de-accent, nonverbal add/remove,
whisper both directions, enhancement, separation by-order + by-content
(target speaker), vocal singing-only + all-voices, quality restoration.
