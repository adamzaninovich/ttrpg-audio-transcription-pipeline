FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 python3-pip ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages \
    faster-whisper==1.1.0 \
    pyannote.audio

# Suppress NVIDIA banners/notices on container start, keep GPU driver check
RUN find /opt/nvidia/entrypoint.d/ -maxdepth 1 ! -name '*gpu*' -type f -delete

WORKDIR /app
COPY transcribe.py /app/transcribe.py

CMD ["python3", "/app/transcribe.py"]
