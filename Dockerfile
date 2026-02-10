FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1

RUN apt-get update &&  apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libfontconfig1

COPY . /app

# Pillow is used for image I/O; torchvision is used by Sam3Processor.
RUN python -m pip install --upgrade pip && \
    python -m pip install pillow einops && \
    python -m pip install -e '.[dev,notebooks,train]'

ENTRYPOINT ["python", "-m", "sam3.cli.segment_image"]
