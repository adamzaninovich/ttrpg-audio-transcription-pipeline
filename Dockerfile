FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04 AS base

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 python3-pip ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages \
    faster-whisper==1.1.0 \
    fastapi "uvicorn[standard]" requests python-multipart

# Suppress NVIDIA banners/notices on container start, keep GPU driver check
RUN find /opt/nvidia/entrypoint.d/ -maxdepth 1 ! -name '*gpu*' -type f -delete

WORKDIR /app
COPY transcribe.py server.py /app/

# Fail the build if any import is missing
RUN python3 -c "from faster_whisper import WhisperModel; from fastapi import FastAPI, File"

CMD ["python3", "/app/transcribe.py"]

FROM base AS full

RUN pip3 install --no-cache-dir --break-system-packages \
    pyannote.audio
